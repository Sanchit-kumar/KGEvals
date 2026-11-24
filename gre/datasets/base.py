"""Dataset interface + helpers (download, dataset-spec parsing, stratified sampling)."""
from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.request
from pathlib import Path
from typing import Callable, ClassVar, Iterable

from gre.core.contracts import Sample


def data_dir() -> Path:
    p = Path(os.environ.get("GRE_DATA_DIR", Path.home() / ".cache" / "gre" / "data"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def fetch(url: str, dest: Path, *, timeout: int = 300, retries: int = 3) -> Path:
    """Download url -> dest once (atomic). Returns dest."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gre-benchmark/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                expected = r.headers.get("Content-Length")
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if expected and tmp.stat().st_size != int(expected):
                raise IOError(f"truncated download {tmp.stat().st_size}/{expected} bytes")
            tmp.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to download {url}: {last}") from last


def fetch_json(url: str, dest: Path) -> object:
    return json.loads(fetch(url, dest).read_text())


class Dataset:
    """Subclass + decorate with @dataset("name").

    Implement `download()` (idempotent) and `load(split) -> list[Sample]`.
    """
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    supported_stages: ClassVar[tuple[str, ...]] = ()
    splits: ClassVar[tuple[str, ...]] = ("test",)
    default_split: ClassVar[str] = "test"
    license: ClassVar[str] = "unknown"
    homepage: ClassVar[str] = ""
    citation: ClassVar[str] = ""
    stratify_key: ClassVar[str | None] = None   # Sample.meta key used for stratified subsampling
    quick_limit: ClassVar[int | None] = None    # override runtime.quick_limit
    gold_mentions: ClassVar[bool | None] = None  # has gold mentions for isolated resolve (default: "resolve" in stages)

    def __init__(self, **options):
        self.options = options

    @property
    def root(self) -> Path:
        return data_dir() / self.name

    def download(self) -> None:
        """Fetch raw files into self.root (no-op if present)."""

    def load(self, split: str) -> list[Sample]:
        raise NotImplementedError

    def info(self) -> dict:
        return {"name": self.name, "stages": list(self.supported_stages), "splits": list(self.splits),
                "license": self.license, "homepage": self.homepage, "description": self.description}


def stratified_subsample(samples: list[Sample], n: int, key: Callable[[Sample], str] | None,
                         seed: int = 0) -> list[Sample]:
    """Deterministic round-robin over strata so small slices stay representative."""
    if n >= len(samples):
        return samples
    rng = random.Random(seed)
    if key is None:
        idx = sorted(rng.sample(range(len(samples)), n))
        return [samples[i] for i in idx]
    strata: dict[str, list[Sample]] = {}
    for s in samples:
        strata.setdefault(key(s), []).append(s)
    for v in strata.values():
        rng.shuffle(v)
    order = sorted(strata)
    out: list[Sample] = []
    i = 0
    while len(out) < n:
        for k in order:
            if i < len(strata[k]) and len(out) < n:
                out.append(strata[k][i])
        i += 1
    return out


_SPEC = re.compile(r"^(?P<name>[A-Za-z0-9_\-]+)(?::(?P<split>[A-Za-z0-9_\-]+))?(?:\[(?P<a>\d*):(?P<b>\d*)\])?$")


def parse_dataset_spec(spec: str) -> tuple[str, str | None, int | None, int | None]:
    """`redocred`, `redocred:test`, `redocred:test[:200]`, `redocred[100:200]`"""
    m = _SPEC.match(spec.strip())
    if not m:
        raise ValueError(f"bad dataset spec {spec!r}; expected name[:split][[a:b]]")
    a = int(m["a"]) if m["a"] else None
    b = int(m["b"]) if m["b"] else None
    return m["name"], m["split"], a, b


def load_samples(spec: str, *, quick_limit: int | None = None, seed: int = 0) -> tuple[Dataset, list[Sample]]:
    from gre.core.registry import get_dataset
    name, split, a, b = parse_dataset_spec(spec)
    ds = get_dataset(name)()
    ds.download()
    samples = ds.load(split or ds.default_split)
    if a is not None or b is not None:
        samples = samples[a:b]
    if quick_limit:
        n = min(ds.quick_limit, quick_limit) if ds.quick_limit else quick_limit
        key = (lambda s: str(s.meta.get(ds.stratify_key, ""))) if ds.stratify_key else None
        samples = stratified_subsample(samples, n, key, seed)
    return ds, samples


def chunked(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
