#!/usr/bin/env python3
"""Inspect swarm H5 datasets written by scenario_json_collect_h5.py."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import h5py
except ImportError as exc:
    raise RuntimeError("h5py is required to inspect the H5 dataset.") from exc


def format_shape(shape: Iterable[int]) -> str:
    return "x".join(str(int(v)) for v in shape) if shape else "scalar"


def format_attr_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        if value.size <= 8:
            return np.array2string(value)
        return f"array(shape={value.shape}, dtype={value.dtype})"
    return str(value)


def print_attrs(obj, indent: str) -> None:
    if not obj.attrs:
        return
    for key, value in obj.attrs.items():
        print(f"{indent}@{key}: {format_attr_value(value)}")


def print_tree(group, prefix: str = "", max_depth: int | None = None, depth: int = 0, show_attrs: bool = False) -> None:
    if max_depth is not None and depth > max_depth:
        return

    if isinstance(group, h5py.Dataset):
        print(f"{prefix}{group.name} [{format_shape(group.shape)}] {group.dtype}")
        if show_attrs:
            print_attrs(group, prefix + "  ")
        return

    if group.name != "/":
        print(f"{prefix}{group.name}/")
        if show_attrs:
            print_attrs(group, prefix + "  ")

    child_prefix = prefix + "  "
    for name in sorted(group.keys()):
        child = group[name]
        if isinstance(child, h5py.Group):
            if max_depth is not None and depth >= max_depth:
                print(f"{child_prefix}{child.name}/ ...")
            else:
                print_tree(child, child_prefix, max_depth, depth + 1, show_attrs)
        else:
            print_tree(child, child_prefix, max_depth, depth + 1, show_attrs)


def maybe_read(group, path: str):
    return group[path][()] if path in group else None


def squeeze_bool_array(array) -> np.ndarray | None:
    if array is None:
        return None
    arr = np.asarray(array)
    if arr.size == 0:
        return arr
    if arr.ndim >= 1 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    return arr.astype(bool, copy=False)


def summarize_timestamps(episode) -> None:
    if "timestamp" not in episode:
        return
    timestamps = np.asarray(episode["timestamp"])
    if timestamps.size == 0:
        print("timestamps: empty")
        return
    dt = np.diff(timestamps)
    dt_text = "n/a" if dt.size == 0 else f"median={np.median(dt):.4f}, min={np.min(dt):.4f}, max={np.max(dt):.4f}"
    print(f"timestamps: steps={timestamps.size}, start={timestamps[0]:.4f}, end={timestamps[-1]:.4f}, {dt_text}")


def summarize_active_mask(episode) -> None:
    if "active_mask" not in episode:
        return
    active = squeeze_bool_array(episode["active_mask"][()])
    if active is None or active.size == 0:
        print("active_mask: empty")
        return
    counts = active.sum(axis=0)
    print(f"active_mask: per-drone active steps={counts.tolist()}")


def summarize_terminal_flags(episode) -> None:
    names = ["collision", "reach_goal", "terminated", "truncated", "done"]
    for name in names:
        if name not in episode:
            continue
        arr = squeeze_bool_array(episode[name][()])
        if arr is None or arr.size == 0:
            print(f"{name}: empty")
            continue
        any_hits = arr.any(axis=0)
        final_hits = arr[-1]
        print(f"{name}: any={any_hits.tolist()} final={final_hits.tolist()}")


def summarize_supervision(episode) -> None:
    if "supervision_label" in episode:
        print(f"supervision_label: {np.asarray(episode['supervision_label']).tolist()}")
    if "failure_reason" in episode:
        reasons = [value.decode('utf-8') if isinstance(value, bytes) else str(value) for value in episode["failure_reason"][()]]
        print(f"failure_reason: {reasons}")
    if "failure_severity" in episode:
        print(f"failure_severity: {np.asarray(episode['failure_severity']).tolist()}")
    if "failure_learnable" in episode:
        print(f"failure_learnable: {np.asarray(episode['failure_learnable']).astype(bool).tolist()}")


def summarize_numeric_dataset(episode, path: str) -> None:
    if path not in episode:
        print(f"{path}: missing")
        return
    ds = episode[path]
    if ds.size == 0:
        print(f"{path}: empty")
        return
    if not np.issubdtype(ds.dtype, np.number):
        print(f"{path}: dtype={ds.dtype}, stats skipped")
        return
    data = ds[()]
    print(
        f"{path}: shape={data.shape}, dtype={data.dtype}, "
        f"min={np.min(data):.6g}, max={np.max(data):.6g}, mean={np.mean(data):.6g}"
    )


def summarize_episode(file_handle, episode_name: str) -> None:
    episode_path = f"episodes/{episode_name}"
    if episode_path not in file_handle:
        raise KeyError(f"Episode not found: {episode_name}")
    episode = file_handle[episode_path]

    print(f"episode: {episode_name}")
    if episode.attrs:
        for key, value in episode.attrs.items():
            print(f"  @{key}: {format_attr_value(value)}")

    if "drone_id" in episode:
        drone_ids = np.asarray(episode["drone_id"])
        print(f"  drone_ids: {drone_ids.tolist()}")
    if "start" in episode and "goal" in episode:
        start = np.asarray(episode["start"])
        goal = np.asarray(episode["goal"])
        print(f"  start shape: {start.shape}, goal shape: {goal.shape}")

    summarize_timestamps(episode)
    summarize_active_mask(episode)
    summarize_supervision(episode)
    summarize_terminal_flags(episode)

    default_stats = [
        "reward",
        "observations/lidar_range",
        "observations/lidar",
        "expert/action_local",
        "expert/action_world",
        "info/root_state",
    ]
    for path in default_stats:
        summarize_numeric_dataset(episode, path)


def summarize_root(file_handle) -> None:
    print(f"file: {file_handle.filename}")
    print_attrs(file_handle, "  ")
    if "scenario/obstacle_centers" in file_handle:
        count = file_handle["scenario/obstacle_centers"].shape[0]
        print(f"scenario obstacles: {count}")
    if "sensor/lidar_ray_directions_local" in file_handle:
        shape = file_handle["sensor/lidar_ray_directions_local"].shape
        print(f"lidar ray directions shape: {shape}")
    if "episodes" in file_handle:
        episodes = sorted(file_handle["episodes"].keys())
        print(f"episodes: {episodes}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect swarm H5 datasets written by scenario_json_collect_h5.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("h5", type=str, help="Path to the H5 dataset")
    parser.add_argument("--episode", type=str, default="000000", help="Episode name to summarize")
    parser.add_argument("--tree", action=argparse.BooleanOptionalAction, default=True, help="Print the H5 tree")
    parser.add_argument("--tree-depth", type=int, default=3, help="Maximum group depth to print for the tree view")
    parser.add_argument("--show-attrs", action=argparse.BooleanOptionalAction, default=False, help="Show HDF5 attributes in the tree view")
    parser.add_argument("--summary", action=argparse.BooleanOptionalAction, default=True, help="Print swarm-specific summary statistics")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    h5_path = Path(args.h5).expanduser().resolve()
    with h5py.File(str(h5_path), "r") as file_handle:
        summarize_root(file_handle)
        if args.tree:
            print("\nTree")
            print_tree(file_handle, max_depth=args.tree_depth, show_attrs=args.show_attrs)
        if args.summary:
            print("\nSummary")
            summarize_episode(file_handle, args.episode)


if __name__ == "__main__":
    main()
