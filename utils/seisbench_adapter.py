
"""SeisBench -> PLAN adapter.

This module converts a SeisBench waveform benchmark dataset (for example
`seisbench.data.Iquique`) into a PLAN-style graph dataset with:

- fixed station order
- per-event tensors of shape (N_stations, 3, T_event)
- optional sliding windows of shape (N_stations, 3, window_length)
- PyTorch Geometric `Data` objects containing `x`, `edge_index`, and
  `station_loc`

The adapter assumes the SeisBench dataset exposes:
- `dataset.metadata` as a pandas DataFrame
- `dataset[idx]` returning a dict with an `"X"` key, or at least the metadata
  row can be used together with `dataset[idx]["X"]`

The code is written to be conservative and robust to small API differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import torch_geometric.data as gdata
    from torch_geometric.loader import DataLoader
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required for this adapter."
    ) from exc


# ----------------------------
# Helpers
# ----------------------------

_COMPONENT_ALIASES = {"Z", "N", "E", "X", "Y", "R", "T"}


def _parse_datetime(value) -> pd.Timestamp:
    """Parse timestamps from SeisBench metadata robustly."""
    ts = pd.to_datetime(value, utc=True)
    if isinstance(ts, pd.Series):
        ts = ts.iloc[0]
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def _canonical_component_order(value: object, default: str = "ZNE") -> str:
    """Return a 3-char component order extracted from metadata."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    s = str(value).upper().replace("*", "")
    order: List[str] = []
    for ch in s:
        if ch in _COMPONENT_ALIASES and ch not in order:
            order.append(ch)
        if len(order) == 3:
            break
    if len(order) != 3:
        return default
    return "".join(order)


def _reorder_to_target(
    x: np.ndarray,
    source_order: str,
    target_order: str,
) -> np.ndarray:
    """
    Reorder components in a 3xT waveform array.

    Example:
        source_order="ZNE", target_order="ZEN" => indices [0, 2, 1]
    """
    if x.shape[0] != 3:
        raise ValueError(f"Expected 3 components, got shape {x.shape}.")
    source_order = _canonical_component_order(source_order)
    target_order = _canonical_component_order(target_order)

    idx = {comp: i for i, comp in enumerate(source_order)}
    if any(comp not in idx for comp in target_order):
        # Fallback: return unchanged if we cannot safely map.
        return x

    return np.stack([x[idx[comp]] for comp in target_order], axis=0)


def zscore_normalize(
    data: np.ndarray,
    axis: Optional[Tuple[int, ...]] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Z-score normalize a waveform array.

    Default: normalize each station/component independently over time.
    """
    mean = data.mean(axis=axis, keepdims=True)
    std = data.std(axis=axis, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (data - mean) / std


def fully_connected_edge_index(n_nodes: int, self_loops: bool = True) -> np.ndarray:
    """
    Build a dense PLAN-style edge index.

    For 16 stations, this yields 256 edges if self_loops=True, matching the
    example shape [2, 256].
    """
    rows, cols = np.meshgrid(np.arange(n_nodes), np.arange(n_nodes), indexing="ij")
    if not self_loops:
        mask = rows != cols
        rows = rows[mask]
        cols = cols[mask]
    return np.vstack([rows.reshape(-1), cols.reshape(-1)])


def build_station_table(
    metadata: pd.DataFrame,
    network_col: str = "station_network_code",
    station_col: str = "station_code",
    lat_col: str = "station_latitude_deg",
    lon_col: str = "station_longitude_deg",
    elev_col: str = "station_elevation_m",
) -> pd.DataFrame:
    """Extract unique station metadata and normalize to PLAN-like columns."""
    cols = [network_col, station_col, lat_col, lon_col, elev_col]
    missing = [c for c in cols if c not in metadata.columns]
    if missing:
        raise KeyError(f"Missing station metadata columns: {missing}")

    station_df = (
        metadata[cols]
        .drop_duplicates()
        .rename(
            columns={
                network_col: "#Network",
                station_col: "Station",
                lat_col: "Latitude",
                lon_col: "Longitude",
                elev_col: "Elevation",
            }
        )
        .reset_index(drop=True)
    )
    return station_df


def save_station_table(station_df: pd.DataFrame, path: str) -> None:
    """Write a PLAN-style station file using pipe separators."""
    station_df.to_csv(path, sep="|", index=False)


def station_loc_array(station_df: pd.DataFrame) -> np.ndarray:
    """Return station locations as [N, 3] = [lat, lon, elev]."""
    return station_df[["Latitude", "Longitude", "Elevation"]].to_numpy(dtype=float)


# ----------------------------
# Event tensor construction
# ----------------------------

@dataclass
class EventTensor:
    x: np.ndarray  # [N, 3, T]
    station_loc: np.ndarray  # [N, 3]
    edge_index: np.ndarray  # [2, E]
    event_key: object
    origin_time: pd.Timestamp
    sample_rate_hz: float
    station_order: List[Tuple[str, str]]  # [(network, station), ...]
    time_start_samples: int  # relative to origin, sample index of x[..., 0]


class SeisBenchPlanAdapter:
    """
    Convert a SeisBench benchmark dataset to PLAN-style event tensors.

    Parameters
    ----------
    dataset:
        SeisBench benchmark dataset (for example `sbd.Iquique(...)`).
    component_order:
        Target component order in the output tensor. PLAN uses ZEN by default
        in the repository example code.
    normalize:
        Apply z-score normalization per station/component over the full event
        tensor before windowing.
    self_loops:
        Whether the graph should include self edges.
    """

    def __init__(
        self,
        dataset,
        component_order: str = "ZEN",
        normalize: bool = True,
        self_loops: bool = True,
        event_group_cols: Sequence[str] = (
            "source_origin_time",
            "source_latitude_deg",
            "source_longitude_deg",
            "source_depth_km",
        ),
    ):
        if not hasattr(dataset, "metadata"):
            raise TypeError("Expected a SeisBench dataset with a `metadata` DataFrame.")
        self.dataset = dataset
        self.metadata = dataset.metadata.copy()
        self.component_order = _canonical_component_order(component_order, default="ZEN")
        self.normalize = normalize
        self.self_loops = self_loops
        self.event_group_cols = tuple(event_group_cols)

        self.station_df = build_station_table(self.metadata)
        self.station_lookup = {
            (row["#Network"], row["Station"]): i
            for i, row in self.station_df.iterrows()
        }
        self.station_loc = station_loc_array(self.station_df)
        self.edge_index = fully_connected_edge_index(
            len(self.station_df), self_loops=self.self_loops
        )

        # Precompute grouping keys while preserving metadata order.
        missing = [c for c in self.event_group_cols if c not in self.metadata.columns]
        if missing:
            raise KeyError(f"Missing event grouping columns: {missing}")

        # Use a stable stringified event key because some columns are timezone-aware timestamps.
        self._event_keys: List[object] = []
        self._group_to_indices: List[np.ndarray] = []
        grouped = self.metadata.groupby(list(self.event_group_cols), sort=False)
        for key, idx in grouped.groups.items():
            self._event_keys.append(key)
            self._group_to_indices.append(np.asarray(list(idx), dtype=int))

    def __len__(self) -> int:
        return len(self._group_to_indices)

    def _get_trace_waveform(self, trace_index: int) -> np.ndarray:
        """
        Access a SeisBench trace waveform.

        Supports both dict-returning datasets (`dataset[idx]["X"]`) and
        objects with a `get_sample`/`get_waveforms` API.
        """
        sample = self.dataset.get_waveforms(trace_index)

        if isinstance(sample, dict):
            if "X" not in sample:
                raise KeyError("Sample dict does not contain key 'X'.")
            x = sample["X"]
        else:
            # Fallback: some dataset objects may return a simple array-like sample.
            x = sample
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[0] != 3:
            raise ValueError(f"Expected waveform shape (3, T), got {x.shape}.")
        return x

    def _event_tensor_from_indices(self, trace_indices: np.ndarray, event_key) -> EventTensor:
        rows = self.metadata.iloc[trace_indices].copy()
        origin_time = _parse_datetime(rows.iloc[0]["source_origin_time"])

        # Build a network tensor spanning the full event extent across available stations.
        starts: List[int] = []
        lengths: List[int] = []
        prepared: List[Tuple[int, np.ndarray, int, int]] = []

        for trace_idx, row in zip(trace_indices, rows.itertuples(index=False)):
            network = getattr(row, "station_network_code")
            station = getattr(row, "station_code")
            station_key = (network, station)
            if station_key not in self.station_lookup:
                continue

            x = self._get_trace_waveform(int(trace_idx))
            src_order = getattr(row, "trace_component_order", "ZNE")
            x = _reorder_to_target(x, src_order, self.component_order)

            fs = float(getattr(row, "trace_sampling_rate_hz", getattr(self.dataset, "sampling_rate", 100.0)))
            if fs <= 0:
                raise ValueError(f"Invalid sampling rate: {fs}")

            trace_start = _parse_datetime(getattr(row, "trace_start_time"))
            rel_start_sec = (trace_start - origin_time).total_seconds()
            rel_start = int(np.round(rel_start_sec * fs))

            starts.append(rel_start)
            lengths.append(x.shape[1])
            prepared.append((self.station_lookup[station_key], x, rel_start, int(round(fs))))

        if not prepared:
            raise ValueError("No usable traces in this event group.")

        global_start = int(min(starts))
        global_end = int(max(s + x.shape[1] for _, x, s, _ in prepared))
        t_len = int(global_end - global_start)

        x_out = np.zeros((len(self.station_df), 3, t_len), dtype=np.float32)

        # If multiple traces exist for the same station in the same event, keep the first non-empty one.
        filled = np.zeros(len(self.station_df), dtype=bool)
        for station_idx, x, rel_start, _fs in prepared:
            if filled[station_idx]:
                continue
            offset = int(rel_start - global_start)
            end = offset + x.shape[1]
            if end > t_len:
                # Safety expansion is not needed because we computed t_len from max end,
                # but keep this guard for numeric edge cases.
                extra = end - t_len
                x_out = np.pad(x_out, ((0, 0), (0, 0), (0, extra)))
                t_len = x_out.shape[2]
            x_out[station_idx, :, offset:end] = x.astype(np.float32)
            filled[station_idx] = True

        if self.normalize:
            x_out = zscore_normalize(x_out, axis=2)

        # Ensure the tensor is PLAN-compatible with target component order.
        if self.component_order != "ZEN":
            # Leave as-is; the caller chose another order.
            pass

        sample_rate = float(rows.iloc[0].get("trace_sampling_rate_hz", getattr(self.dataset, "sampling_rate", 100.0)))

        return EventTensor(
            x=x_out,
            station_loc=self.station_loc.copy(),
            edge_index=self.edge_index.copy(),
            event_key=event_key,
            origin_time=origin_time,
            sample_rate_hz=sample_rate,
            station_order=list(self.station_lookup.keys()),
            time_start_samples=global_start,
        )

    def __getitem__(self, index: int) -> EventTensor:
        return self._event_tensor_from_indices(self._group_to_indices[index], self._event_keys[index])

    def export_station_file(self, path: str) -> None:
        save_station_table(self.station_df, path)


# ----------------------------
# Windowed PLAN-style dataset
# ----------------------------

class SeisBenchPlanWindowDataset(Dataset):
    """
    Sliding-window dataset over SeisBench event tensors.

    This mirrors PLAN's continuous-window logic:
        data[:, :, left:right]

    Parameters
    ----------
    adapter:
        A `SeisBenchPlanAdapter` instance.
    window_length:
        Number of samples per window. PLAN examples use 3072.
    start_index / end_index / interval:
        Window coordinates are in samples on the event tensor.
        By default, each event is windowed from 0 to its full length.
    """

    def __init__(
        self,
        adapter: SeisBenchPlanAdapter,
        window_length: int = 3072,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        interval: int = 500,
        batch_window_per_event: bool = True,
    ):
        self.adapter = adapter
        self.window_length = int(window_length)
        self.start_index = start_index
        self.end_index = end_index
        self.interval = int(interval)
        self.batch_window_per_event = batch_window_per_event

        self._event_cache: List[EventTensor] = [adapter[i] for i in range(len(adapter))]

        self._index_map: List[Tuple[int, int, int]] = []
        for event_idx, event in enumerate(self._event_cache):
            t_len = event.x.shape[2]
            start = 0 if self.start_index is None else max(0, int(self.start_index))
            end = t_len if self.end_index is None else min(t_len, int(self.end_index))
            if end <= start:
                continue
            lefts = np.arange(start, max(start, end - self.window_length + 1), self.interval, dtype=int)
            # If the tensor is shorter than the requested window, still produce a single padded window.
            if len(lefts) == 0:
                lefts = np.array([start], dtype=int)
            for left in lefts:
                right = int(left + self.window_length)
                self._index_map.append((event_idx, int(left), right))

    def __len__(self) -> int:
        return len(self._index_map)

    @staticmethod
    def _slice_and_pad(x: np.ndarray, left: int, right: int) -> np.ndarray:
        """Slice x[..., left:right] and right-pad with zeros if needed."""
        if left < 0:
            raise ValueError("left index must be non-negative after preprocessing.")
        if right <= left:
            raise ValueError("right index must be greater than left.")
        t_len = x.shape[2]
        if left >= t_len:
            # Entirely out of range: return zeros.
            return np.zeros((x.shape[0], x.shape[1], right - left), dtype=x.dtype)
        chunk = x[:, :, left:min(right, t_len)]
        if chunk.shape[2] < (right - left):
            pad = right - left - chunk.shape[2]
            chunk = np.pad(chunk, ((0, 0), (0, 0), (0, pad)))
        return chunk

    def __getitem__(self, index: int):
        event_idx, left, right = self._index_map[index]
        event = self._event_cache[event_idx]
        x = self._slice_and_pad(event.x, left, right)

        data = gdata.Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(event.edge_index, dtype=torch.long),
            station_loc=torch.tensor(event.station_loc, dtype=torch.float32),
        )
        # Keep a few useful fields for debugging / downstream mapping.
        data.event_key = event.event_key
        data.origin_time = str(event.origin_time)
        data.left_index = int(left)
        data.right_index = int(right)
        data.sample_rate_hz = float(event.sample_rate_hz)
        return data


def construct_plan_dataloader_from_seisbench(
    dataset,
    window_length: int = 3072,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    interval: int = 500,
    batchsize: int = 1,
    num_workers: int = 0,
    component_order: str = "ZEN",
    normalize: bool = True,
    self_loops: bool = True,
):
    """
    Convenience wrapper that returns a PyG DataLoader and the adapter.

    Example
    -------
    >>> import seisbench.data as sbd
    >>> ds = sbd.Iquique(sampling_rate=100)
    >>> train, dev, test = ds.train_dev_test()
    >>> loader, adapter = construct_plan_dataloader_from_seisbench(train)
    >>> batch = next(iter(loader))
    >>> batch.x.shape
    torch.Size([N, 3, 3072])
    """
    adapter = SeisBenchPlanAdapter(
        dataset,
        component_order=component_order,
        normalize=normalize,
        self_loops=self_loops,
    )
    window_ds = SeisBenchPlanWindowDataset(
        adapter=adapter,
        window_length=window_length,
        start_index=start_index,
        end_index=end_index,
        interval=interval,
    )
    loader = DataLoader(
        window_ds,
        shuffle=False,
        batch_size=batchsize,
        num_workers=num_workers,
    )
    return loader, adapter


# ----------------------------
# Optional direct save helpers
# ----------------------------

def export_plan_station_file(dataset, path: str) -> pd.DataFrame:
    """Build and export a PLAN station file from a SeisBench dataset."""
    adapter = SeisBenchPlanAdapter(dataset)
    adapter.export_station_file(path)
    return adapter.station_df


def export_event_tensors(
    dataset,
    out_dir: str,
    prefix: str = "event_",
    component_order: str = "ZEN",
    normalize: bool = True,
    self_loops: bool = True,
) -> List[str]:
    """
    Export one `.npz` per event containing:
        x, station_loc, edge_index, event_key, origin_time, sample_rate_hz
    """
    from pathlib import Path

    adapter = SeisBenchPlanAdapter(
        dataset,
        component_order=component_order,
        normalize=normalize,
        self_loops=self_loops,
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    for i in range(len(adapter)):
        event = adapter[i]
        fname = out_path / f"{prefix}{i:06d}.npz"
        np.savez_compressed(
            fname,
            x=event.x,
            station_loc=event.station_loc,
            edge_index=event.edge_index,
            event_key=np.array([str(event.event_key)], dtype=object),
            origin_time=np.array([str(event.origin_time)], dtype=object),
            sample_rate_hz=np.array([event.sample_rate_hz], dtype=float),
            time_start_samples=np.array([event.time_start_samples], dtype=int),
        )
        written.append(str(fname))
    return written
