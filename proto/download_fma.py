"""
Download FMA Small dataset and metadata to data/raw/.
FMA Small: 8000 tracks, 30s clips, ~8GB
"""

import hashlib
import urllib.request
import zipfile
from pathlib import Path

from config import FMA_METADATA_DIR, FMA_SMALL_DIR, RAW_DIR

FILES = {
    "fma_small.zip": {
        "url": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
        "md5": "4edb51c99a19d31fe01a7d44d5cac19b",
    },
    "fma_metadata.zip": {
        "url": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
        "md5": None,
    },
}


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name} ...")

    def progress(block, block_size, total):
        done = block * block_size
        pct = done / total * 100 if total > 0 else 0
        mb = done / 1e6
        print(f"\r  {mb:.0f} MB  ({pct:.1f}%)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()


def extract(zip_path: Path, dest: Path) -> None:
    print(f"Extracting {zip_path.name} -> {dest} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    print("  done.")


def main() -> None:
    for filename, meta in FILES.items():
        zip_path = RAW_DIR / filename

        # Download if missing
        if not zip_path.exists():
            download(meta["url"], zip_path)
        else:
            print(f"{filename} already downloaded, skipping.")

        # Verify checksum
        if meta["md5"] is not None:
            print(f"Verifying {filename} ...")
            actual = md5(zip_path)
            if actual != meta["md5"]:
                print(f"  WARNING: MD5 mismatch (got {actual}), proceeding anyway.")
            else:
                print("  checksum OK.")

        # Extract
        dest = FMA_SMALL_DIR if "fma_small" in filename else FMA_METADATA_DIR
        if any(dest.iterdir()) if dest.exists() else False:
            print(f"{dest.name} already extracted, skipping.")
        else:
            extract(zip_path, dest)

    print("\nAll done.")
    print(f"  Audio : {FMA_SMALL_DIR}")
    print(f"  Metadata: {FMA_METADATA_DIR}")


if __name__ == "__main__":
    main()
