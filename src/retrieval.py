"""Find the most similar TRAIN papers for every dev or test paper (prompt v4).

Prompt v4 shows the database labels of similar training papers, so the model can see the
database's labelling conventions by example (SPEC.md, section 13, 2026-11-19). Similarity
is computed on the first pages of each PDF (src/first_pages.py) with one of two methods:

- `embedding`: a multilingual sentence-embedding model (`retrieval.model`), because the
  papers mix German and English and TF-IDF cannot match a German paper to an English one;
- `tfidf`: TF-IDF with cosine similarity, for comparison (a colleague's system uses it).

Neighbours are always TRAIN papers that have a reference, never papers of the evaluated
split, so no gold label of an evaluated paper can leak into its prompt. A train id that
also occurs in the evaluated split is left out as well.

Usage (run by the team, it reads course data):
    python3 src/retrieval.py --split dev
    python3 src/retrieval.py --split test
    python3 src/retrieval.py --split dev --method tfidf

Writes `paths.processed/neighbours_<split>.json` with `retrieval.k` neighbours per paper
(method and k are stored in the file and checked when prompt v4 reads it). The embeddings
are cached per split in `paths.embeddings`; only new or changed texts are embedded again.
"""
import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from data import load_references
from first_pages import TEXT_SUFFIX, first_pages_dir
from settings import load_settings

TRAIN_SPLIT = "train"
QUERY_SPLITS = ("dev", "test")
METHOD_EMBEDDING = "embedding"
METHOD_TFIDF = "tfidf"
METHODS = (METHOD_EMBEDDING, METHOD_TFIDF)
NEIGHBOURS_FILE_PATTERN = "neighbours_{split}.json"

# Turns a list of texts into one row vector per text (normalised to length 1).
Encoder = Callable[[list[str]], np.ndarray]
# {query id: [(train id, similarity), ...]}, most similar first
NeighbourLists = dict[str, list[tuple[str, float]]]


def neighbours_path(settings: dict[str, Any], split: str) -> Path:
    return settings["paths"]["processed"] / NEIGHBOURS_FILE_PATTERN.format(split=split)


def load_texts(text_dir: Path) -> dict[str, str]:
    """{id: first-pages text} of one split; documents without text have no file."""
    if not text_dir.exists():
        return {}
    return {path.stem: path.read_text(encoding="utf-8")
            for path in sorted(text_dir.glob(f"*{TEXT_SUFFIX}"))}


def train_candidates(train_texts: dict[str, str], train_reference_ids: set[str],
                     query_ids: set[str]) -> dict[str, str]:
    """Train papers that may be shown as examples: with a reference (their labels are the
    examples) and not part of the evaluated split (no gold of an evaluated paper)."""
    return {document_id: text for document_id, text in train_texts.items()
            if document_id in train_reference_ids and document_id not in query_ids}


def rank_by_similarity(query_ids: list[str], query_vectors: np.ndarray,
                       train_ids: list[str], train_vectors: np.ndarray, n: int) -> NeighbourLists:
    """The `n` most similar train papers per query. The vectors have length 1, so the dot
    product is the cosine similarity. Ties are broken by id, so the result is stable."""
    similarities = query_vectors @ train_vectors.T
    ranked: NeighbourLists = {}
    for row, query_id in enumerate(query_ids):
        scored = sorted(zip(train_ids, similarities[row].tolist()), key=lambda pair: (-pair[1], pair[0]))
        ranked[query_id] = [(train_id, round(score, 4)) for train_id, score in scored[:n]]
    return ranked


# --- sentence embeddings ---------------------------------------------------------------------

def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_sentence_encoder(settings: dict[str, Any]) -> Encoder:
    """The model of `retrieval.model`; imported here because loading it takes seconds."""
    from sentence_transformers import SentenceTransformer

    retrieval = settings["retrieval"]
    model = SentenceTransformer(retrieval["model"])
    prefix = retrieval["text_prefix"]

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode([prefix + text for text in texts], batch_size=retrieval["batch_size"],
                            normalize_embeddings=True, show_progress_bar=True)

    return encode


def embedding_cache_path(settings: dict[str, Any], split: str) -> Path:
    model_name = settings["retrieval"]["model"].replace("/", "__")
    return settings["paths"]["embeddings"] / f"{model_name}_{split}.npz"


def cached_embeddings(texts: dict[str, str], encode: Encoder, cache_path: Path) -> dict[str, np.ndarray]:
    """{id: vector}. Texts whose hash is in the cache are not embedded again, so a second
    run (or a run after more first pages were read) only embeds what is new."""
    cached: dict[str, tuple[str, np.ndarray]] = {}
    if cache_path.exists():
        with np.load(cache_path) as stored:
            for document_id, digest, vector in zip(stored["ids"], stored["hashes"], stored["vectors"]):
                cached[str(document_id)] = (str(digest), vector)
    vectors = {document_id: cached[document_id][1] for document_id, text in texts.items()
               if document_id in cached and cached[document_id][0] == text_hash(text)}
    missing = [document_id for document_id in texts if document_id not in vectors]
    if missing:
        print(f"Embedding {len(missing)} texts ({len(vectors)} from the cache)")
        for document_id, vector in zip(missing, encode([texts[document_id] for document_id in missing])):
            vectors[document_id] = np.asarray(vector)
            cached[document_id] = (text_hash(texts[document_id]), vectors[document_id])
        # Entries not asked for this time stay in the cache.
        ids = sorted(cached)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, ids=np.array(ids), hashes=np.array([cached[i][0] for i in ids]),
                 vectors=np.stack([cached[i][1] for i in ids]))
    return vectors


def embedding_neighbours(query_vectors: dict[str, np.ndarray], train_vectors: dict[str, np.ndarray],
                         n: int) -> NeighbourLists:
    query_ids, train_ids = sorted(query_vectors), sorted(train_vectors)
    return rank_by_similarity(query_ids, np.stack([query_vectors[i] for i in query_ids]),
                              train_ids, np.stack([train_vectors[i] for i in train_ids]), n)


# --- TF-IDF ----------------------------------------------------------------------------------

def tfidf_neighbours(query_texts: dict[str, str], train_texts: dict[str, str], n: int,
                     retrieval_settings: dict[str, Any]) -> NeighbourLists:
    """Vocabulary and weights are learned from the train texts only, so the result for
    one paper does not depend on the other papers of its split."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=retrieval_settings["tfidf_min_df"],
                                 max_df=retrieval_settings["tfidf_max_df"])
    train_ids, query_ids = sorted(train_texts), sorted(query_texts)
    train_matrix = vectorizer.fit_transform([train_texts[i] for i in train_ids])
    query_matrix = vectorizer.transform([query_texts[i] for i in query_ids])
    # TfidfVectorizer normalises every row to length 1 (norm="l2").
    return rank_by_similarity(query_ids, query_matrix.toarray(), train_ids, train_matrix.toarray(), n)


# --- both methods ----------------------------------------------------------------------------

def find_neighbours(settings: dict[str, Any], split: str, method: str, n: int,
                    encoder: Encoder | None = None) -> tuple[NeighbourLists, list[str]]:
    """`n` train neighbours for every paper of `split` that has a first-pages text.

    Returns the neighbour lists and the ids of the split without a text (no neighbours).
    `encoder`: only for `embedding`; default is the model of `retrieval.model`.
    """
    if split not in QUERY_SPLITS:
        raise ValueError(f"neighbours are found for {QUERY_SPLITS}, not for {split!r}")
    from pdf_to_text import document_ids_for_split  # imports Docling, so only when needed

    split_ids = document_ids_for_split(settings, split)
    split_texts = load_texts(first_pages_dir(settings, split))
    query_texts = {document_id: split_texts[document_id] for document_id in split_ids
                   if document_id in split_texts}
    without_text = sorted(set(split_ids) - set(query_texts))
    train_reference_ids = set(load_references(settings["paths"]["references_train"]))
    train_texts = train_candidates(load_texts(first_pages_dir(settings, TRAIN_SPLIT)),
                                   train_reference_ids, set(split_ids))
    if not train_texts:
        raise ValueError("no train first pages found: run src/first_pages.py --split train first")

    if method == METHOD_TFIDF:
        return tfidf_neighbours(query_texts, train_texts, n, settings["retrieval"]), without_text
    if method != METHOD_EMBEDDING:
        raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")
    encode = encoder or load_sentence_encoder(settings)
    query_vectors = cached_embeddings(query_texts, encode, embedding_cache_path(settings, split))
    train_vectors = cached_embeddings(train_texts, encode, embedding_cache_path(settings, TRAIN_SPLIT))
    return embedding_neighbours(query_vectors, train_vectors, n), without_text


def write_neighbours(settings: dict[str, Any], split: str, method: str, k: int,
                     neighbours: NeighbourLists, without_text: list[str]) -> Path:
    """Write the neighbour file that prompt v4 reads; method and k are stored so a file
    made with other settings is noticed (few_shot_examples.py)."""
    content = {
        "split": split,
        "method": method,
        "model": settings["retrieval"]["model"] if method == METHOD_EMBEDDING else None,
        "first_pages_chars": settings["retrieval"]["first_pages_chars"],
        "k": k,
        "without_text": without_text,
        "neighbours": {query_id: [{"id": train_id, "similarity": similarity}
                                  for train_id, similarity in ranked]
                       for query_id, ranked in neighbours.items()},
    }
    path = neighbours_path(settings, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the most similar train papers for a split.")
    parser.add_argument("--split", choices=QUERY_SPLITS, required=True)
    parser.add_argument("--method", choices=METHODS, default=None, help="default: retrieval.method")
    parser.add_argument("--experiment", type=Path, default=None)
    args = parser.parse_args()

    settings = load_settings(args.experiment)
    method = args.method or settings["retrieval"]["method"]
    k = settings["retrieval"]["k"]
    neighbours, without_text = find_neighbours(settings, args.split, method, k)
    path = write_neighbours(settings, args.split, method, k, neighbours, without_text)
    print(f"{args.split}: {len(neighbours)} papers with {k} {method} neighbours, "
          f"{len(without_text)} without first-pages text (no examples): {without_text}")
    print(f"Written: {path}")


if __name__ == "__main__":
    main()
