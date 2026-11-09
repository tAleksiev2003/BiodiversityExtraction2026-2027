#!/usr/bin/env python3
"""
Direct Single-File WebDAV Downloader for ANLP 2026 Shared Task C.

Bypasses the DFKI Nextcloud 12.3 GB on-the-fly ZIP timeout by querying the exact
list of remote PDF filenames via Nextcloud WebDAV (PROPFIND) and downloading
each PDF individually with resume support into `data/raw_pdfs/<split>/`.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]

SHARES: dict[str, str] = {
    "train": "cHAQ6Z3acdtSwie",
    "dev": "MgqMso583238C79",
}
PASSWORD: str = "sXzFps7q"
BASE_DAV_URL: str = "https://cloud.dfki.de/owncloud/public.php/dav/files"


def list_remote_pdfs(split: str) -> list[str]:
    """Queries the Nextcloud WebDAV endpoint to retrieve all available PDF IDs in the share."""
    token = SHARES[split]
    url = f"{BASE_DAV_URL}/{token}/"
    auth_str = base64.b64encode(f"{token}:{PASSWORD}".encode("utf-8")).decode("ascii")
    req = urllib.request.Request(
        url,
        method="PROPFIND",
        headers={
            "Authorization": f"Basic {auth_str}",
            "Depth": "1",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        xml_text = resp.read().decode("utf-8", errors="ignore")
    return sorted(set(re.findall(r"([A-Z0-9]{8})\.pdf", xml_text)))


def download_single_pdf(zot_id: str, split: str, out_dir: Path) -> tuple[str, str]:
    """Downloads a single PDF by its `zotitem_ptr_id` from the corresponding Nextcloud share."""
    target_path = out_dir / f"{zot_id}.pdf"
    if target_path.exists() and target_path.stat().st_size > 1000:
        return zot_id, "skipped (already exists)"

    token = SHARES[split]
    url = f"{BASE_DAV_URL}/{token}/{zot_id}.pdf"
    auth_str = base64.b64encode(f"{token}:{PASSWORD}".encode("utf-8")).decode("ascii")
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth_str}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            if data.startswith(b"%PDF"):
                target_path.write_bytes(data)
                return zot_id, f"downloaded ({len(data) // 1024} KB)"
            return zot_id, "invalid PDF header"
    except urllib.error.HTTPError as e:
        return zot_id, f"HTTP error {e.code}"
    except Exception as e:
        return zot_id, f"error: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Shared Task C PDFs individually via WebDAV."
    )
    parser.add_argument(
        "--split",
        choices=["dev", "train"],
        default="dev",
        help="Dataset split to download (default: dev)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of PDFs to download (0 = all available in split)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of concurrent download threads (default: 6)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional custom output directory (default: data/raw_pdfs/<split>)",
    )
    args = parser.parse_args()

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (REPO_ROOT / "data" / "raw_pdfs" / args.split)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ids_file = REPO_ROOT / "data" / "gold" / "dev_100_ids.txt"
    if args.split == "dev" and ids_file.exists():
        ids = sorted({l.strip() for l in ids_file.read_text().splitlines() if l.strip()})
    else:
        print(f"Fetching remote PDF list for '{args.split}' from DFKI Nextcloud WebDAV...")
        ids = list_remote_pdfs(args.split)

    if args.limit > 0:
        ids = ids[: args.limit]

    print(f"Downloading {len(ids)} '{args.split}' PDFs to {out_dir} (workers={args.workers})...")

    stats = {"downloaded": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_single_pdf, zid, args.split, out_dir): zid for zid in ids}
        for i, fut in enumerate(as_completed(futures), 1):
            zid, status = fut.result()
            if status.startswith("downloaded"):
                stats["downloaded"] += 1
            elif status.startswith("skipped"):
                stats["skipped"] += 1
            else:
                stats["failed"] += 1
            if i % 10 == 0 or i == len(ids) or "error" in status:
                print(f"[{i}/{len(ids)}] {zid}.pdf -> {status}")

    print(
        f"\nCompleted! Downloaded: {stats['downloaded']}, "
        f"Skipped: {stats['skipped']}, Failed: {stats['failed']}"
    )


if __name__ == "__main__":
    main()
