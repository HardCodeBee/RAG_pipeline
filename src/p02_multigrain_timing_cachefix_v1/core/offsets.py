from pathlib import Path
import numpy as np
from .vendor.src.retrievers.chunk_store import JsonlOffsetChunkStore as ORIGINAL


class CompatibleOffsets(ORIGINAL):
    def __init__(self, chunks_path, offsets_path, *, expected_rows=None):
        offsets_path = Path(offsets_path).resolve()
        chunks_path = Path(chunks_path).resolve()
        if not chunks_path.is_file() or not offsets_path.is_file():
            raise FileNotFoundError("Missing frozen chunk/offset artifact")
        offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        if offsets.ndim != 1 or offsets.dtype not in {
            np.dtype("int64"),
            np.dtype("uint64"),
        }:
            raise ValueError("Offsets must be one-dimensional 64-bit integers")
        if offsets.dtype == np.dtype("int64"):
            if np.any(offsets < 0):
                raise ValueError("Negative chunk offsets are invalid")
            # Nonnegative signed values have the same uint64 bit representation.
            offsets = offsets.view(np.uint64)
        if expected_rows is not None and len(offsets) != expected_rows:
            raise ValueError("Chunk offset count differs from manifest")
        if len(offsets) and (
            int(offsets[0]) != 0
            or np.any(offsets[1:] <= offsets[:-1])
            or int(offsets[-1]) >= chunks_path.stat().st_size
        ):
            raise ValueError("Invalid ordered chunk offsets or file bounds")
        self.chunks_path, self.offsets_path, self._offsets = (
            chunks_path,
            offsets_path,
            offsets,
        )
