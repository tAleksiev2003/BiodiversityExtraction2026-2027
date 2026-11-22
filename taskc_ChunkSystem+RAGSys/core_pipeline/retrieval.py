"""Paper-local dense retrieval with bounded embedding inputs and disk caching."""
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path
import httpx

GROUPS = {
    "habitat_ecosystem": {"fields": ["habitat", "ecosystem_type"], "query":
        "What habitats, study sites and ecosystem types does this study investigate? "
        "Find descriptions of the study area, land use, vegetation and aquatic or terrestrial environments."},
    "taxa": {"fields": ["taxa"], "query":
        "Which organisms and taxonomic groups are the subjects of this study? "
        "Find focal sampled or analysed species, scientific names and common names."},
    "biodiversity": {"fields": ["biodiversity_level", "biodiversity_variable"], "query":
        "Which levels of biodiversity does this study investigate and which biodiversity variables "
        "does it measure or analyse? Find genetic, species, community or ecosystem diversity, "
        "abundance, richness, biomass, composition and diversity indices."},
    "ecosystem_service": {"fields": ["ecosystem_service"], "query":
        "Which ecosystem services does this study investigate? Find provisioning, regulation, "
        "maintenance and cultural services, ecosystem contributions and benefits to people."},
}
QUERY_PREFIX = "Instruct: Retrieve passages from a scientific paper that answer the question.\nQuery: "


def digest(value):
    """Hash JSON-serializable content deterministically."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def page_units(paper_id, pages):
    """Preserve canonical page text and numbering while excluding empty index entries."""
    return [{"id": f"{paper_id}:p{int(number):03d}", "paper_id": paper_id,
             "section": f"Page {number}", "section_index": int(number) - 1,
             "page_number": int(number), "start": 0, "end": len(text),
             "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
            for number, text in pages if text.strip()]


def embedding_slices(text, max_bytes):
    """Partition text without loss into UTF-8-byte-bounded embedding windows."""
    if max_bytes < 4:
        raise ValueError("Embedding window must allow at least four UTF-8 bytes")
    start, size = 0, 0
    for i, char in enumerate(text):
        width = len(char.encode("utf-8"))
        if size + width > max_bytes:
            yield start, i, text[start:i]
            start, size = i, 0
        size += width
    if start < len(text):
        yield start, len(text), text[start:]


def unit_vector(values):
    """Validate and normalize a finite, nonzero embedding vector."""
    if not isinstance(values, list) or not values:
        raise ValueError("Missing embedding vector")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise ValueError("Invalid embedding values")
    norm = math.sqrt(sum(x * x for x in values))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Invalid embedding norm")
    return [x / norm for x in values]


class EmbeddingClient:
    """Use an OpenAI-compatible embeddings endpoint without storing credentials."""

    def __init__(self, base_url, model, cache, query_prefix=QUERY_PREFIX,
                 document_prefix="", max_bytes=6000, batch_size=16, timeout=180):
        """Store explicit model, prefix, cache and input-size configuration."""
        if not model or max_bytes < 128 or batch_size < 1:
            raise ValueError("Invalid embedding configuration")
        self.base_url, self.model = base_url.rstrip("/"), model
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.query_prefix, self.document_prefix = query_prefix, document_prefix
        self.max_bytes, self.batch_size, self.timeout = max_bytes, batch_size, timeout
        self.calls = []

    def embed(self, texts, kind):
        """Return normalized embeddings, reusing content-keyed cached vectors."""
        if kind not in {"query", "document"}:
            raise ValueError("Unknown embedding role")
        prefix = self.query_prefix if kind == "query" else self.document_prefix
        result = [None] * len(texts)
        missing = []
        for index, text in enumerate(texts):
            prepared = prefix + text
            if len(prepared.encode()) > self.max_bytes:
                raise ValueError("Embedding input exceeds byte guard; shorten query or window")
            key = digest(["embedding-v1", self.base_url, self.model, kind, prepared])
            path = self.cache / (key + ".json")
            if path.exists():
                result[index] = unit_vector(json.loads(path.read_text())["vector"])
                self.calls.append({"cache_hit": True, "items": 1, "usage": {}})
            else:
                missing.append((index, prepared, path))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            vectors, metadata = self._request([item[1] for item in batch])
            self.calls.append(metadata)
            for (index, _, path), vector in zip(batch, vectors):
                result[index] = vector
                temp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
                temp.write_text(json.dumps({"vector": vector}), encoding="utf-8")
                temp.replace(path)
        if result and len({len(v) for v in result}) != 1:
            raise ValueError("Embedding dimensions differ; clear stale cache or fix endpoint")
        return result

    def _request(self, texts):
        """Retry transient failures and validate response indices and vector dimensions."""
        key = os.environ.get("EMBEDDING_API_KEY") or os.environ.get("LLM_API_KEY")
        headers = {"Authorization": "Bearer " + key} if key else {}
        request = {"model": self.model, "input": texts, "encoding_format": "float"}
        started = time.monotonic()
        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self.base_url + "/embeddings", json=request, headers=headers)
                    response.raise_for_status()
                body = response.json()
                data = body["data"]
                if len(data) != len(texts) or sorted(x["index"] for x in data) != list(range(len(texts))):
                    raise ValueError("Embedding response indices do not match inputs")
                vectors = [unit_vector(x["embedding"]) for x in sorted(data, key=lambda x: x["index"])]
                if len({len(v) for v in vectors}) != 1:
                    raise ValueError("Inconsistent embedding dimensions")
                return vectors, {"cache_hit": False, "items": len(texts),
                                 "usage": body.get("usage", {}), "attempts": attempt + 1,
                                 "elapsed_s": time.monotonic() - started}
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise
            except (httpx.TransportError, ValueError, KeyError, TypeError):
                if attempt == 2:
                    raise
            time.sleep(2 ** attempt)
        raise RuntimeError("Unreachable retry state")


def rank_pages(units, embedder, groups=GROUPS):
    """Rank pages per query by maximum window cosine similarity within this paper."""
    if not units or len({u["paper_id"] for u in units}) != 1:
        raise ValueError("Retrieval requires nonempty units from exactly one paper")
    limit = embedder.max_bytes - len(embedder.document_prefix.encode())
    windows = [(u, a, b, text) for u in units
               for a, b, text in embedding_slices(u["text"], limit)]
    docs = embedder.embed([w[3] for w in windows], "document")
    queries = embedder.embed([g["query"] for g in groups.values()], "query")
    rankings = {}
    for (name, _), query in zip(groups.items(), queries):
        scores = {}
        for (unit, a, b, _), vector in zip(windows, docs):
            if len(query) != len(vector):
                raise ValueError("Query and document embedding dimensions differ")
            score = sum(x * y for x, y in zip(query, vector))
            old = scores.get(unit["id"])
            if old is None or score > old["score"]:
                scores[unit["id"]] = {"unit_id": unit["id"], "score": score,
                                      "page_number": unit["page_number"], "best_window": [a, b]}
        rankings[name] = sorted(scores.values(), key=lambda x: (-x["score"], x["page_number"]))
    return rankings, [{"unit_id": u["id"], "start": a, "end": b} for u, a, b, _ in windows]
