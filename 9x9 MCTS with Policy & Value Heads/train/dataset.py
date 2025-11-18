from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from configs.train_config import LoaderConfig


def discover_selfplay_files(data_dir: str | Path, pattern: str = "selfplay_*.npz") -> List[Path]:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No self-play files found in {data_dir} matching pattern '{pattern}'")
    return files


class SelfPlayDataset(Dataset):
    """
    Loads one or more self-play .npz files and presents them as a single dataset.

    Expected arrays per .npz:
      - 'planes': (T, C=2, H, W)
      - 'pi'    : (T, A)  where A = H*W + 1 (PASS)
      - 'z'     : (T,)    scalar outcome in [-1, 1] from the current player's perspective

    If your keys differ, pass `keys={'planes': '...', 'pi': '...', 'z': '...'}`.
    By default, arrays are fully loaded into memory for speed. Set in_memory=False to
    lazily read from disk (slower but lighter).
    """
    def __init__(self,
                 files: List[Path],
                 *,
                 in_memory: bool = True,
                 keys: Dict[str, str] | None = None,
                 verify: bool = True) -> None:
        super().__init__()
        self.files: List[Path] = list(files)
        self.in_memory: bool = in_memory
        self.keys = keys or {"planes": "planes", "pi": "pi", "z": "z"}

        # Per-file arrays or placeholders (if lazy)
        self._data: List[Optional[Dict[str, np.ndarray]]] = []
        self._handles: Dict[int, np.lib.npyio.NpzFile] = {}  # for lazy mode
        self._lengths: List[int] = []                        # samples per file
        self._cum: np.ndarray                                # prefix sums for index mapping

        # Load metadata and optionally materialize arrays
        for _, path in enumerate(self.files):
            with np.load(path, allow_pickle=False) as npz:
                kP, kPI, kZ = self.keys["planes"], self.keys["pi"], self.keys["z"]

                if kP not in npz or kPI not in npz or kZ not in npz:
                    # Try a couple of common fallbacks for 'pi'
                    if kPI not in npz:
                        for alt in ("policy", "policies"):
                            if alt in npz:
                                kPI = alt
                                break
                    if kP not in npz or kPI not in npz or kZ not in npz:
                        raise KeyError(
                            f"{path.name} must contain keys {self.keys}, "
                            f"found: {list(npz.files)}")

                planes = npz[kP]
                pi = npz[kPI]
                z = npz[kZ]

                if verify:
                    if planes.ndim != 4:
                        raise ValueError(f"{path.name}: planes expected (T,2,H,W), got {planes.shape}")
                    if pi.ndim != 2:
                        raise ValueError(f"{path.name}: pi expected (T,A), got {pi.shape}")
                    if z.ndim != 1:
                        raise ValueError(f"{path.name}: z expected (T,), got {z.shape}")
                    T = planes.shape[0]
                    if not (pi.shape[0] == T == z.shape[0]):
                        raise ValueError(
                            f"{path.name}: mismatched lengths planes={planes.shape[0]} "
                            f"pi={pi.shape[0]} z={z.shape[0]}")
                    H, W = planes.shape[-2], planes.shape[-1]
                    A = H * W + 1
                    if pi.shape[1] != A:
                        raise ValueError(
                            f"{path.name}: pi dimension {pi.shape[1]} != H*W+1 ({A})")

                self._lengths.append(planes.shape[0])

                if self.in_memory:
                    # Materialize to RAM (float32 to save memory/bw)
                    self._data.append({
                        "planes": planes.astype(np.float32, copy=False),
                        "pi":     pi.astype(np.float32, copy=False),
                        "z":      z.astype(np.float32, copy=False),
                    })
                else:
                    self._data.append(None)  # open on demand in __getitem__

        # Build prefix sums for O(log F) index routing
        self._cum = np.cumsum([0] + self._lengths, dtype=np.int64)

    def __len__(self) -> int:
        return int(self._cum[-1])

    def _locate(self, index: int) -> Tuple[int, int]:
        # Map global index -> (file_idx, local_idx)
        # np.searchsorted finds the right bin in prefix sums.
        fidx = int(np.searchsorted(self._cum, index, side="right") - 1)
        local = index - int(self._cum[fidx])
        return fidx, local

    def _get_arrays(self, fidx: int) -> Dict[str, np.ndarray]:
        if self.in_memory:
            return self._data[fidx]  # type: ignore[return-value]
        # Lazy: keep the npz handle open for this worker
        if fidx not in self._handles:
            self._handles[fidx] = np.load(self.files[fidx], allow_pickle=False)
        h = self._handles[fidx]
        kP, kPI, kZ = self.keys["planes"], self.keys["pi"], self.keys["z"]
        # resolve fallback for pi if used during __init__
        if kPI not in h:
            for alt in ("policy", "policies"):
                if alt in h:
                    kPI = alt
                    break
        return {"planes": h[kP], "pi": h[kPI], "z": h[kZ]}

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < 0:
            index = len(self) + index
        fidx, local = self._locate(index)
        arrs = self._get_arrays(fidx)

        planes = arrs["planes"][local]
        pi = arrs["pi"][local]
        z = arrs["z"][local]

        # Convert to tensors
        planes_t = torch.from_numpy(planes).to(dtype=torch.float32)   # (2,H,W)
        pi_t     = torch.from_numpy(pi).to(dtype=torch.float32)       # (A,)
        z_t      = torch.tensor(float(z), dtype=torch.float32)         # ()

        return {"planes": planes_t, "pi": pi_t, "z": z_t}

    def __del__(self):
        # Close any lazy handles
        for f in self._handles.values():
            try:
                f.close()
            except Exception:
                pass
        self._handles.clear()



def make_datasets(cfg: LoaderConfig) -> Tuple[Dataset, Dataset]:
    files = discover_selfplay_files(cfg.data_dir, cfg.pattern)

    if cfg.split_by == "file":
        # Split at game-file granularity to avoid leakage.
        g = torch.Generator().manual_seed(cfg.seed)
        perm = torch.randperm(len(files), generator=g).tolist()
        files = [files[i] for i in perm]

        n_val_files = max(1, int(round(len(files) * cfg.val_frac)))
        val_files = files[:n_val_files]
        train_files = files[n_val_files:]
        if not train_files:
            raise RuntimeError("Train split is empty after file-level split.")
        train_ds = SelfPlayDataset(train_files, in_memory=cfg.in_memory)
        val_ds = SelfPlayDataset(val_files, in_memory=cfg.in_memory)
        return train_ds, val_ds

    # Default: split by samples across the unified dataset
    full = SelfPlayDataset(files, in_memory=cfg.in_memory)
    n_total = len(full)
    n_val = max(1, int(round(n_total * cfg.val_frac)))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(full, lengths=[n_train, n_val], generator=g)
    return train_ds, val_ds


def make_dataloaders(cfg: LoaderConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds, val_ds = make_datasets(cfg)
    # Note: shuffle=True for train, False for val
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        drop_last=False,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    cfg = LoaderConfig()
    train_dl, val_dl = make_dataloaders(cfg)
    print(f"Train batches: {len(train_dl)} | Val batches: {len(val_dl)}")
    batch = next(iter(train_dl))
    print({k: tuple(v.shape) for k, v in batch.items()})
