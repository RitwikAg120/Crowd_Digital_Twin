"""
Get the datasets the project uses — MOT17 and CrowdHuman — on the GB10.

    python scripts/fetch_datasets.py all                 # MOT17 sequences + CrowdHuman check
    python scripts/fetch_datasets.py mot17 --mot-seqs 02 04 09 11

Where they go (the paths evaluate.py / train_dense.py / scripts/run_gb10.sh use):
  mot17       dataset/MOT17/train/MOT17-XX-FRCNN   only the chosen sequences (static cameras
              02, 04, 09 by default, ≈0.5 GB) are pulled out of the 5.9 GB MOT17.zip with
              HTTP range requests, so the whole archive is never downloaded
  crowdhuman  dataset/CrowdHuman/                   annotation_train.odgt, annotation_val.odgt
              and the images (the CrowdHuman_train0*.zip / CrowdHuman_val.zip contents,
              e.g. Images/). CrowdHuman needs its licence accepted at crowdhuman.org, so it
              is not downloaded here: this checks that it is in place and complete.
"""

import argparse
import io
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"
MOT17_URL = "https://motchallenge.net/data/MOT17.zip"


def _open(url: str, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": "cdt-fetch/1.0", **(headers or {})})
    return urllib.request.urlopen(req, timeout=60)


class HTTPRangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file through HTTP range requests (for zipfile)."""
    BLOCK = 4 << 20

    def __init__(self, url: str, fetch=None):
        self.url, self.pos = url, 0
        self._fetch = fetch or self._http_range
        self.size = self._fetch(None, None)
        self._cache = {}

    def _http_range(self, start, end):
        if start is None:                                    # size probe
            with _open(self.url, {"Range": "bytes=0-0"}) as r:
                cr = r.headers.get("Content-Range", "")
                if "/" not in cr:
                    raise RuntimeError(f"{self.url} does not support range requests")
                return int(cr.split("/")[-1])
        with _open(self.url, {"Range": f"bytes={start}-{end}"}) as r:
            return r.read()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _block(self, i):
        if i not in self._cache:
            if len(self._cache) > 16:
                self._cache.pop(next(iter(self._cache)))
            start = i * self.BLOCK
            self._cache[i] = self._fetch(start, min(start + self.BLOCK, self.size) - 1)
        return self._cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            i, off = divmod(self.pos, self.BLOCK)
            b = self._block(i)[off:off + n]
            if not b:
                break
            out += b
            self.pos += len(b)
            n -= len(b)
        return bytes(out)

    def readinto(self, buf):
        data = self.read(len(buf))
        buf[:len(data)] = data
        return len(data)


def extract_members(zf: zipfile.ZipFile, part: str, dest: Path):
    """Extract the members whose path contains `part` into dest/<path from `part` on>."""
    names = [n for n in zf.namelist() if part in n and not n.endswith("/")]
    for k, name in enumerate(names, 1):
        out = dest / name[name.index(part):]
        if out.exists() and out.stat().st_size == zf.getinfo(name).file_size:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(name) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        if k % 100 == 0 or k == len(names):
            print(f"\r    {k}/{len(names)} files", end="", flush=True)
    print()
    return len(names)


# ─── Datasets ─────────────────────────────────────────────────────────────────

def fetch_mot17(args):
    seqs = [f"MOT17-{s.zfill(2)}-FRCNN" for s in args.mot_seqs]
    todo = [s for s in seqs if not (DATA / "MOT17" / "train" / s / "gt" / "gt.txt").exists()]
    for s in seqs:
        if s not in todo:
            print(f"  have dataset/MOT17/train/{s}")
    if not todo:
        return
    print(f"  reading the MOT17.zip index over HTTP ({MOT17_URL}) …")
    with zipfile.ZipFile(HTTPRangeFile(MOT17_URL)) as zf:
        for s in todo:
            n = extract_members(zf, f"train/{s}/", DATA / "MOT17")
            print(f"  {s}: {n} files → dataset/MOT17/train/{s}")


def check_crowdhuman(args):
    root = DATA / "CrowdHuman"
    if not root.exists():
        print("  dataset/CrowdHuman not found. Download it from https://www.crowdhuman.org/ "
              "(accept the licence) and unzip so that dataset/CrowdHuman holds "
              "annotation_train.odgt, annotation_val.odgt and the images.")
        return
    images = {p.stem for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    for split in ("train", "val"):
        odgt = next(iter(sorted(root.rglob(f"annotation_{split}.odgt"))), None)
        if odgt is None:
            print(f"  missing annotation_{split}.odgt under dataset/CrowdHuman")
            continue
        ids = [json.loads(l)["ID"] for l in odgt.read_text(encoding="utf-8").splitlines() if l.strip()]
        found = sum(i in images for i in ids)
        print(f"  {split}: {len(ids)} images annotated, {found} found"
              + ("" if found == len(ids) else "  ← unzip the missing image archives"))


FETCHERS = {"mot17": fetch_mot17, "crowdhuman": check_crowdhuman}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="+", choices=list(FETCHERS) + ["all"])
    ap.add_argument("--mot-seqs", nargs="+", default=["02", "04", "09"],
                    help="MOT17 sequence numbers (static cameras: 02 04 09)")
    args = ap.parse_args()
    what = list(FETCHERS) if "all" in args.what else args.what
    for w in what:
        print(f"[{w}]")
        try:
            FETCHERS[w](args)
        except (urllib.error.URLError, OSError, zipfile.BadZipFile) as e:
            print(f"  failed: {e}")


if __name__ == "__main__":
    sys.exit(main())
