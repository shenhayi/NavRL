#!/usr/bin/env python3
"""
Collect a Primitive-Planner scenario replay into a swarm H5 dataset.

This mirrors the existing scenario_json_replay.py flow:
- load a planner JSON scenario
- spawn the same static obstacles and multirotor swarm
- replay the logged expert trajectories with LeePositionController

On top of replay, it writes a synchronized H5 file containing:
- per-drone expert references
- per-drone onboard D435-style RGB-D images
- per-drone Mid360-style LiDAR
- swarm states and replay outcomes

LiDAR collection supports two backends:
- RTX LiDAR via omni.isaac.sensor LidarRtx and a custom Mid360 profile
- a legacy hybrid path that uses Orbit RayCaster for static obstacles plus analytic
  peer-drone overlays

The H5 lidar datasets keep the same canonical range-image format regardless of backend.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent
ISAAC_TRAINING_DIR = TRAINING_DIR.parent
THIRD_PARTY_DIR = ISAAC_TRAINING_DIR / "third_party"
OMNIDRONES_DIR = THIRD_PARTY_DIR / "OmniDrones"
ORBIT_SOURCE_DIR = THIRD_PARTY_DIR / "orbit" / "source"
REPO_ROOT = ISAAC_TRAINING_DIR.parents[2]
ISAAC_SIM_DIR = REPO_ROOT / "isaac-sim"

for path in (SCRIPT_DIR, TRAINING_DIR, OMNIDRONES_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
for ext_dir in ORBIT_SOURCE_DIR.glob("extensions/*"):
    path_str = str(ext_dir)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


DEFAULT_ARGS = {
    "config": None,
    "json": None,
    "output": None,
    "headless": False,
    "anti_aliasing": 4,
    "drone_model": "Hummingbird",
    "sim_dt": 0.01,
    "ground_size": 120.0,
    "obstacle_material": "navrl",
    "save_stage": None,
    "device": None,
    "compression": "lzf",
    "store_global_cloud": False,
    "lidar_backend": "rtx",
    "rtx_lidar_config": None,
    "lidar_hres": 1.0,
    "lidar_vbeams": 40,
    "lidar_min_range": None,
    "lidar_max_range": None,
    "lidar_vfov_up_deg": None,
    "lidar_vfov_down_deg": None,
    "lidar_forward_tilt_deg": None,
    "lidar_attach_yaw_only": False,
    "lidar_offset": (0.0, 0.0, 0.0),
    "peer_box_size": (0.12, 0.12, 0.16),
    "camera_enabled": True,
    "camera_resolution": (640, 480),
    "camera_offset": (0.08, 0.0, 0.0),
    "camera_target": (1.08, 0.0, 0.0),
    "camera_focus_distance": 4.0,
    "camera_focal_length": 3.43,
    "camera_horizontal_aperture": 6.4,
    "camera_clip_range": (0.1, 30.0),
}
DEFAULT_COLLECT_CONFIG = TRAINING_DIR / "cfg" / "collect.yaml"


def load_runtime_config(config_path: Path) -> dict:
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError("Hydra/OmegaConf is required to load config files.") from exc

    resolved_path = config_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    with initialize_config_dir(config_dir=str(resolved_path.parent), version_base=None):
        cfg = compose(config_name=resolved_path.stem)
    return OmegaConf.to_container(cfg, resolve=True)


def get_nested(mapping: dict, *keys: str):
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
        if value is None:
            return None
    return value


def first_non_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def config_arg_defaults(config: dict) -> dict:
    collect_cfg = get_nested(config, "collect") or get_nested(config, "replay") or {}
    return {
        "json": first_non_none(collect_cfg.get("json"), config.get("json")),
        "output": collect_cfg.get("output"),
        "headless": first_non_none(collect_cfg.get("headless"), config.get("headless")),
        "anti_aliasing": first_non_none(collect_cfg.get("anti_aliasing"), get_nested(config, "viewer", "anti_aliasing")),
        "drone_model": first_non_none(collect_cfg.get("drone_model"), get_nested(config, "drone", "model_name")),
        "sim_dt": first_non_none(collect_cfg.get("sim_dt"), get_nested(config, "sim", "dt")),
        "ground_size": collect_cfg.get("ground_size"),
        "obstacle_material": collect_cfg.get("obstacle_material"),
        "save_stage": collect_cfg.get("save_stage"),
        "device": first_non_none(collect_cfg.get("device"), config.get("device"), get_nested(config, "sim", "device")),
        "compression": collect_cfg.get("compression"),
        "store_global_cloud": collect_cfg.get("store_global_cloud"),
        "lidar_backend": collect_cfg.get("lidar_backend"),
        "rtx_lidar_config": collect_cfg.get("rtx_lidar_config"),
        "lidar_hres": collect_cfg.get("lidar_hres"),
        "lidar_vbeams": collect_cfg.get("lidar_vbeams"),
        "lidar_min_range": collect_cfg.get("lidar_min_range"),
        "lidar_max_range": collect_cfg.get("lidar_max_range"),
        "lidar_vfov_up_deg": collect_cfg.get("lidar_vfov_up_deg"),
        "lidar_vfov_down_deg": collect_cfg.get("lidar_vfov_down_deg"),
        "lidar_forward_tilt_deg": collect_cfg.get("lidar_forward_tilt_deg"),
        "lidar_attach_yaw_only": collect_cfg.get("lidar_attach_yaw_only"),
        "lidar_offset": collect_cfg.get("lidar_offset"),
        "peer_box_size": collect_cfg.get("peer_box_size"),
        "camera_enabled": collect_cfg.get("camera_enabled"),
        "camera_resolution": collect_cfg.get("camera_resolution"),
        "camera_offset": collect_cfg.get("camera_offset"),
        "camera_target": collect_cfg.get("camera_target"),
        "camera_focus_distance": collect_cfg.get("camera_focus_distance"),
        "camera_focal_length": collect_cfg.get("camera_focal_length"),
        "camera_horizontal_aperture": collect_cfg.get("camera_horizontal_aperture"),
        "camera_clip_range": collect_cfg.get("camera_clip_range"),
    }


def build_parser(defaults: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay Primitive-Planner scenario_data.json and write a swarm H5 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=defaults["config"], help="Optional YAML/Hydra config file")
    parser.add_argument("--json", type=str, default=defaults["json"], help="Path to scenario_data.json")
    parser.add_argument("--output", type=str, default=defaults["output"], help="Output H5 path. Defaults to <json stem>.h5")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=defaults["headless"], help="Run headless")
    parser.add_argument("--anti-aliasing", type=int, default=defaults["anti_aliasing"], help="Viewport anti-aliasing level")
    parser.add_argument("--drone-model", type=str, default=defaults["drone_model"], help="OmniDrones multirotor model")
    parser.add_argument("--sim-dt", type=float, default=defaults["sim_dt"], help="Physics and rendering dt")
    parser.add_argument("--device", type=str, default=defaults["device"], help="Simulation device")
    parser.add_argument("--ground-size", type=float, default=defaults["ground_size"], help="Ground plane size in meters")
    parser.add_argument(
        "--obstacle-material",
        type=str,
        default=defaults["obstacle_material"],
        choices=("navrl", "transparent"),
        help="Visual style for obstacle cuboids",
    )
    parser.add_argument("--save-stage", type=str, default=defaults["save_stage"], help="Optional .usd/.usda path to save the generated stage")
    parser.add_argument(
        "--compression",
        type=str,
        default=defaults["compression"],
        choices=("none", "lzf", "gzip"),
        help="H5 dataset compression",
    )
    parser.add_argument("--store-global-cloud", action=argparse.BooleanOptionalAction, default=defaults["store_global_cloud"], help="Store scenario.global_cloud_world once in the H5")

    parser.add_argument("--lidar-backend", type=str, default=defaults["lidar_backend"], choices=("rtx", "raycast"), help="LiDAR backend used for collection")
    parser.add_argument("--rtx-lidar-config", type=str, default=defaults["rtx_lidar_config"], help="RTX LiDAR config name or .json path resolved from Isaac Sim lidar_configs")
    parser.add_argument("--lidar-hres", type=float, default=defaults["lidar_hres"], help="Horizontal angular resolution in degrees")
    parser.add_argument("--lidar-vbeams", type=int, default=defaults["lidar_vbeams"], help="Number of vertical LiDAR beams")
    parser.add_argument("--lidar-min-range", type=float, default=defaults["lidar_min_range"], help="Minimum LiDAR range in meters")
    parser.add_argument("--lidar-max-range", type=float, default=defaults["lidar_max_range"], help="Maximum LiDAR range in meters")
    parser.add_argument("--lidar-vfov-up-deg", type=float, default=defaults["lidar_vfov_up_deg"], help="Vertical FoV above horizon in degrees")
    parser.add_argument("--lidar-vfov-down-deg", type=float, default=defaults["lidar_vfov_down_deg"], help="Vertical FoV below horizon in degrees")
    parser.add_argument("--lidar-forward-tilt-deg", type=float, default=defaults["lidar_forward_tilt_deg"], help="Positive values pitch the LiDAR forward/down")
    parser.add_argument("--lidar-attach-yaw-only", action=argparse.BooleanOptionalAction, default=defaults["lidar_attach_yaw_only"], help="If set, LiDAR follows yaw only instead of full attitude")
    parser.add_argument("--lidar-offset", nargs=3, type=float, default=defaults["lidar_offset"], help="LiDAR translation offset from base_link")
    parser.add_argument("--peer-box-size", nargs=3, type=float, default=defaults["peer_box_size"], help="Peer-drone OBB size used for LiDAR overlay")

    parser.add_argument("--camera-enabled", action=argparse.BooleanOptionalAction, default=defaults["camera_enabled"], help="Enable onboard D435 capture via Replicator")
    parser.add_argument("--camera-resolution", nargs=2, type=int, default=defaults["camera_resolution"], help="D435-style camera resolution")
    parser.add_argument("--camera-offset", nargs=3, type=float, default=defaults["camera_offset"], help="Camera translation offset from base_link")
    parser.add_argument("--camera-target", nargs=3, type=float, default=defaults["camera_target"], help="Camera look-at target in the camera parent frame")
    parser.add_argument("--camera-focus-distance", type=float, default=defaults["camera_focus_distance"], help="Camera focus distance")
    parser.add_argument("--camera-focal-length", type=float, default=defaults["camera_focal_length"], help="Camera focal length in mm")
    parser.add_argument("--camera-horizontal-aperture", type=float, default=defaults["camera_horizontal_aperture"], help="Camera horizontal aperture in mm")
    parser.add_argument("--camera-clip-range", nargs=2, type=float, default=defaults["camera_clip_range"], help="Camera near/far clipping range")
    return parser


def parse_args() -> argparse.Namespace:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_COLLECT_CONFIG) if DEFAULT_COLLECT_CONFIG.is_file() else None,
    )
    bootstrap_args, _ = bootstrap_parser.parse_known_args()

    defaults = DEFAULT_ARGS.copy()
    if bootstrap_args.config is not None:
        config_path = Path(bootstrap_args.config)
        defaults.update({k: v for k, v in config_arg_defaults(load_runtime_config(config_path)).items() if v is not None})
        defaults["config"] = str(config_path.expanduser().resolve())

    parser = build_parser(defaults)
    args = parser.parse_args()
    if args.json is None:
        parser.error("Scenario JSON path is required. Pass --json or set collect.json/json in --config.")

    json_path = Path(args.json).expanduser().resolve()
    if args.output is None:
        output_path = json_path.with_suffix(".h5")
    else:
        output_path = Path(args.output).expanduser()
        suffix = output_path.suffix.lower()
        treat_as_directory = (output_path.exists() and output_path.is_dir()) or suffix not in (".h5", ".hdf5")
        if treat_as_directory:
            output_path = output_path / f"{json_path.stem}.h5"
        output_path = output_path.resolve()
    args.output = str(output_path)

    if len(args.lidar_offset) != 3:
        parser.error("--lidar-offset must have exactly 3 values.")
    if len(args.peer_box_size) != 3:
        parser.error("--peer-box-size must have exactly 3 values.")
    if len(args.camera_resolution) != 2:
        parser.error("--camera-resolution must have exactly 2 values.")
    if len(args.camera_offset) != 3:
        parser.error("--camera-offset must have exactly 3 values.")
    if len(args.camera_target) != 3:
        parser.error("--camera-target must have exactly 3 values.")
    if len(args.camera_clip_range) != 2:
        parser.error("--camera-clip-range must have exactly 2 values.")
    if args.lidar_hres <= 0.0:
        parser.error("--lidar-hres must be positive.")
    if args.lidar_vbeams <= 0:
        parser.error("--lidar-vbeams must be positive.")
    if args.sim_dt <= 0.0:
        parser.error("--sim-dt must be positive.")
    return args


def load_scenario(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required to write the dataset. Install it in the Python environment used to launch Isaac Sim."
        ) from exc
    return h5py


def interp_array(ts: np.ndarray, values: np.ndarray, t: float) -> np.ndarray:
    if len(ts) == 0:
        return np.zeros(values.shape[-1], dtype=float)
    if t <= ts[0]:
        return values[0]
    if t >= ts[-1]:
        return values[-1]
    idx = np.searchsorted(ts, t, side="right") - 1
    idx = max(0, min(idx, len(ts) - 2))
    dt = max(float(ts[idx + 1] - ts[idx]), 1e-9)
    alpha = (t - ts[idx]) / dt
    return values[idx] * (1.0 - alpha) + values[idx + 1] * alpha


def yaw_from_velocity(vel: np.ndarray, fallback_yaw: float) -> float:
    if np.linalg.norm(vel[:2]) < 1e-5:
        return fallback_yaw
    return math.atan2(float(vel[1]), float(vel[0]))


def scenario_duration(drones: list[dict]) -> float:
    duration = 0.0
    for drone in drones:
        traj = drone.get("trajectory")
        if traj and traj.get("timestamps"):
            duration = max(duration, float(traj["timestamps"][-1]))
    return max(duration, 0.1)


def resolve_sim_device(requested_device: str | None) -> str:
    if requested_device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device}' but CUDA is unavailable; falling back to cpu.")
        return "cpu"
    return requested_device


def resolve_drone_model_name(requested_name: str, registry: dict) -> str:
    if requested_name in registry:
        return requested_name

    normalized = requested_name.lower()
    for model_name in registry:
        if model_name.lower() == normalized:
            return model_name

    available = ", ".join(sorted(registry))
    raise KeyError(f"Unknown drone model '{requested_name}'. Available models: {available}")


def build_run_name(prefix: str = "scenario_collect") -> str:
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")
    return f"{prefix}_{timestamp}"


def _get_obstacle_value(obstacle: dict, *keys: str, default=None):
    for key in keys:
        if key in obstacle and obstacle[key] is not None:
            return obstacle[key]
    if default is not None:
        return default
    raise KeyError(f"Missing obstacle keys {keys}")


def _get_first_present(mapping: dict, *keys: str, default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def quat_wxyz_to_yaw(quat, fallback_yaw: float = 0.0) -> float:
    if quat is None or len(quat) != 4:
        return fallback_yaw
    w, x, y, z = (float(v) for v in quat)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def infer_timestamps(traj: dict, num_samples: int) -> np.ndarray:
    timestamps = _get_first_present(traj, "timestamps", "time")
    if timestamps is not None:
        return np.asarray(timestamps, dtype=float)
    if num_samples <= 0:
        return np.zeros((0,), dtype=float)
    dt = float(_get_first_present(traj, "dt", default=0.01))
    duration = _get_first_present(traj, "duration")
    if duration is not None and num_samples > 1:
        return np.linspace(0.0, float(duration), num=num_samples, dtype=float)
    return np.arange(num_samples, dtype=float) * dt


def infer_velocities(traj: dict, positions: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    velocities = _get_first_present(traj, "velocities", "velocities_world")
    if velocities is not None:
        vel = np.asarray(velocities, dtype=float)
        if len(vel) == len(positions):
            return vel

    vel = np.zeros_like(positions, dtype=float)
    if len(positions) <= 1:
        return vel

    diffs = np.diff(positions, axis=0)
    dt = np.maximum(np.diff(timestamps), 1e-6)
    vel[:-1] = diffs / dt[:, None]
    vel[-1] = 0.0
    return vel


def infer_accelerations(traj: dict, velocities: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    accelerations = _get_first_present(traj, "accelerations", "accelerations_world")
    if accelerations is not None:
        acc = np.asarray(accelerations, dtype=float)
        if len(acc) == len(velocities):
            return acc

    acc = np.zeros_like(velocities, dtype=float)
    if len(velocities) <= 1:
        return acc

    diffs = np.diff(velocities, axis=0)
    dt = np.maximum(np.diff(timestamps), 1e-6)
    acc[:-1] = diffs / dt[:, None]
    acc[-1] = 0.0
    return acc


def extract_track_data(scenario: dict) -> tuple[list[dict], list[tuple[float, float, float]], list[float]]:
    start_positions = []
    track_data = []
    init_yaws = []

    for index, drone_data in enumerate(scenario.get("drones", [])):
        traj = drone_data.get("trajectory") or {}
        positions_raw = _get_first_present(traj, "positions", "positions_world")
        if positions_raw is None:
            continue

        positions = np.asarray(positions_raw, dtype=float)
        if positions.ndim != 2 or positions.shape[0] == 0:
            continue

        timestamps = infer_timestamps(traj, len(positions))
        if len(timestamps) != len(positions):
            continue

        velocities = infer_velocities(traj, positions, timestamps)
        accelerations = infer_accelerations(traj, velocities, timestamps)
        initial_orientation = np.asarray(drone_data.get("initial_orientation_wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=float)
        goal = np.asarray(drone_data.get("goal", positions[-1]), dtype=float)
        fallback_yaw = quat_wxyz_to_yaw(initial_orientation, 0.0)
        init_yaw = yaw_from_velocity(velocities[0], fallback_yaw)

        start_positions.append(tuple(positions[0].tolist()))
        init_yaws.append(init_yaw)
        track_data.append(
            {
                "id": int(drone_data.get("id", index)),
                "timestamps": timestamps,
                "positions": positions,
                "velocities": velocities,
                "accelerations": accelerations,
                "goal": goal,
                "start": positions[0].copy(),
                "initial_orientation_wxyz": initial_orientation,
                "fallback_yaw": init_yaw,
            }
        )

    return track_data, start_positions, init_yaws


def extract_box_obstacles(scenario: dict) -> tuple[list[dict], list[str]]:
    obstacles = []
    raw_types = []
    supported_types = {"box", "cube", "cuboid", "obb", "static_box", "static_obstacle"}

    for raw_obstacle in scenario.get("obstacles", []):
        if not isinstance(raw_obstacle, dict):
            continue

        raw_type = raw_obstacle.get("type")
        if raw_type is not None:
            raw_types.append(str(raw_type))

        normalized_type = str(raw_type).strip().lower() if raw_type is not None else "box"
        if normalized_type not in supported_types and not any(
            key in raw_obstacle for key in ("size_x", "x_width", "width", "dx")
        ):
            continue

        try:
            x = float(_get_obstacle_value(raw_obstacle, "x", "center_x", "cx"))
            y = float(_get_obstacle_value(raw_obstacle, "y", "center_y", "cy"))
            z = float(_get_obstacle_value(raw_obstacle, "z", "center_z", "cz"))
            size_x = float(_get_obstacle_value(raw_obstacle, "size_x", "x_width", "width", "dx"))
            size_y = float(_get_obstacle_value(raw_obstacle, "size_y", "y_width", "length", "dy"))
            size_z = float(_get_obstacle_value(raw_obstacle, "size_z", "z_width", "height", "dz"))
            yaw = float(_get_obstacle_value(raw_obstacle, "yaw", "angle", "heading", default=0.0))
        except (KeyError, TypeError, ValueError):
            continue

        obstacles.append(
            {
                "x": x,
                "y": y,
                "z": z,
                "size_x": size_x,
                "size_y": size_y,
                "size_z": size_z,
                "yaw": yaw,
            }
        )

    return obstacles, sorted(set(raw_types))


def track_duration(track_data: list[dict]) -> float:
    duration = 0.0
    for track in track_data:
        timestamps = track.get("timestamps")
        if timestamps is not None and len(timestamps) > 0:
            duration = max(duration, float(timestamps[-1]))
    return max(duration, 0.1)


def make_obstacle_materials(sim_utils, material_style: str):
    if material_style == "navrl":
        body_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 1.0, 0.0),
            metallic=0.2,
            roughness=0.45,
            opacity=1.0,
        )
        cap_material = body_material
    else:
        body_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.90, 0.73, 0.22),
            emissive_color=(0.08, 0.06, 0.02),
            roughness=0.35,
            metallic=0.05,
            opacity=0.42,
        )
        cap_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.42, 0.10),
            emissive_color=(0.35, 0.10, 0.03),
            roughness=0.15,
            metallic=0.0,
            opacity=1.0,
        )
    return body_material, cap_material


def _float_from_scenario(sensor: dict, key: str, fallback: float) -> float:
    value = sensor.get(key)
    if value is None:
        return fallback
    return float(value)


def apply_scenario_sensor_defaults(args: argparse.Namespace, scenario: dict) -> None:
    sensor = scenario.get("sensor") or {}
    if args.lidar_min_range is None:
        args.lidar_min_range = _float_from_scenario(sensor, "min_range", 0.1)
    if args.lidar_max_range is None:
        args.lidar_max_range = _float_from_scenario(sensor, "max_range", 10.0)
    if args.lidar_vfov_up_deg is None:
        args.lidar_vfov_up_deg = _float_from_scenario(sensor, "vertical_fov_up_deg", 52.0)
    if args.lidar_vfov_down_deg is None:
        args.lidar_vfov_down_deg = _float_from_scenario(sensor, "vertical_fov_down_deg", 7.0)
    if args.lidar_forward_tilt_deg is None:
        args.lidar_forward_tilt_deg = _float_from_scenario(sensor, "forward_tilt_deg", 15.0)
    if getattr(args, "rtx_lidar_config", None) is None:
        sensor_model = str(sensor.get("model", "mid360")).lower()
        if "mid360" in sensor_model:
            args.rtx_lidar_config = "MID360_40ch20hz1024res"


def get_rtx_lidar_config_root() -> Path:
    return ISAAC_SIM_DIR / "exts" / "omni.isaac.sensor" / "data" / "lidar_configs"


def resolve_rtx_lidar_config_path(config_name: str) -> Path:
    candidate = Path(config_name).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    config_root = get_rtx_lidar_config_root()
    stem = Path(config_name).stem
    direct_path = config_root / f"{stem}.json"
    if direct_path.is_file():
        return direct_path.resolve()

    matches = sorted(config_root.rglob(f"{stem}.json"))
    if not matches:
        raise FileNotFoundError(f"RTX lidar config '{config_name}' not found under {config_root}")
    return matches[0].resolve()


def load_rtx_lidar_profile(config_name: str) -> dict:
    config_path = resolve_rtx_lidar_config_path(config_name)
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_rtx_ray_directions_local(profile_data: dict, num_cols: int) -> np.ndarray:
    profile = profile_data.get("profile") or {}
    if str(profile.get("scanType", "rotary")).lower() != "rotary":
        raise NotImplementedError("Only rotary RTX lidar configs are supported by this collector.")

    emitters = profile.get("emitters")
    if not isinstance(emitters, dict):
        raise RuntimeError("RTX lidar profile is missing profile.emitters for rotary ray direction reconstruction.")

    elevation_deg = np.asarray(emitters.get("elevationDeg") or [], dtype=np.float32)
    if elevation_deg.size == 0:
        raise RuntimeError("RTX lidar profile has no emitter elevationDeg values.")
    azimuth_offsets_deg = np.asarray(emitters.get("azimuthDeg") or np.zeros_like(elevation_deg), dtype=np.float32)
    if azimuth_offsets_deg.size != elevation_deg.size:
        raise RuntimeError("RTX lidar profile emitter azimuthDeg/elevationDeg length mismatch.")

    start_az_deg = float(profile.get("startAzimuthDeg", 0.0))
    end_az_deg = float(profile.get("endAzimuthDeg", 360.0))
    if end_az_deg <= start_az_deg:
        end_az_deg = start_az_deg + 360.0
    base_azimuth_deg = np.linspace(start_az_deg, end_az_deg, num=num_cols, endpoint=False, dtype=np.float32)

    azimuth_deg = base_azimuth_deg[:, None] + azimuth_offsets_deg[None, :]
    elevation_rad = np.deg2rad(elevation_deg[None, :])
    azimuth_rad = np.deg2rad(azimuth_deg)

    cos_elev = np.cos(elevation_rad)
    ray_dirs = np.stack(
        [
            cos_elev * np.cos(azimuth_rad),
            cos_elev * np.sin(azimuth_rad),
            np.sin(elevation_rad),
        ],
        axis=-1,
    )
    return np.ascontiguousarray(ray_dirs.astype(np.float32, copy=False))


def make_goal_direction(track_data: list[dict]) -> np.ndarray:
    goal_dirs = []
    for track in track_data:
        direction = np.asarray(track["goal"] - track["start"], dtype=np.float32)
        direction[2] = 0.0
        if np.linalg.norm(direction[:2]) < 1e-6:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        goal_dirs.append(direction)
    return np.stack(goal_dirs, axis=0)


def sample_track(track: dict, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    timestamps = track["timestamps"]
    pos = interp_array(timestamps, track["positions"], t)
    active = t <= float(timestamps[-1]) + 1e-6
    if active:
        vel = interp_array(timestamps, track["velocities"], t)
        acc = interp_array(timestamps, track["accelerations"], t)
    else:
        vel = np.zeros(3, dtype=np.float32)
        acc = np.zeros(3, dtype=np.float32)
    yaw = yaw_from_velocity(vel, track["fallback_yaw"])
    return pos, vel, acc, yaw, active


def quat_from_euler_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def compute_scene_bounds(obstacles: list[dict], track_data: list[dict]) -> tuple[float, float, float, float, float]:
    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []
    z_max = 0.0

    for obs in obstacles:
        x = float(obs["x"])
        y = float(obs["y"])
        z = float(obs["z"])
        size_x = float(obs["size_x"])
        size_y = float(obs["size_y"])
        size_z = float(obs["size_z"])
        yaw = float(obs.get("yaw", 0.0))
        cos_yaw = abs(math.cos(yaw))
        sin_yaw = abs(math.sin(yaw))
        half_x = 0.5 * (cos_yaw * size_x + sin_yaw * size_y)
        half_y = 0.5 * (sin_yaw * size_x + cos_yaw * size_y)
        x_mins.append(x - half_x)
        x_maxs.append(x + half_x)
        y_mins.append(y - half_y)
        y_maxs.append(y + half_y)
        z_max = max(z_max, z + 0.5 * size_z)

    for track in track_data:
        positions = track["positions"]
        if len(positions) == 0:
            continue
        x_mins.append(float(np.min(positions[:, 0])))
        x_maxs.append(float(np.max(positions[:, 0])))
        y_mins.append(float(np.min(positions[:, 1])))
        y_maxs.append(float(np.max(positions[:, 1])))
        z_max = max(z_max, float(np.max(positions[:, 2])))

    if not x_mins:
        return (-10.0, 10.0, -10.0, 10.0, 10.0)

    return min(x_mins), max(x_maxs), min(y_mins), max(y_maxs), z_max


def compute_viewer_camera_pose(obstacles: list[dict], track_data: list[dict]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    x_min, x_max, y_min, y_max, z_max = compute_scene_bounds(obstacles, track_data)
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    span = max(x_max - x_min, y_max - y_min, 1.0)
    eye_z = max(0.85 * span + 8.0, z_max + 6.0)
    return (center_x, center_y, eye_z), (center_x, center_y, 0.0)


def make_box_triangles(center: np.ndarray, size: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    hx, hy, hz = 0.5 * size
    corners = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rot = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    points = corners @ rot.T + center.astype(np.float32)
    triangles = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int32,
    )
    return points, triangles


def create_static_scene_mesh(stage, mesh_root_path: str, obstacles: list[dict], ground_size: float):
    from pxr import UsdGeom

    stage.DefinePrim(mesh_root_path, "Xform")
    mesh_path = f"{mesh_root_path}/mesh"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)

    points = []
    counts = []
    indices = []
    vertex_offset = 0

    half_ground = 0.5 * ground_size
    ground_vertices = np.array(
        [
            [-half_ground, -half_ground, 0.01],
            [half_ground, -half_ground, 0.01],
            [half_ground, half_ground, 0.01],
            [-half_ground, half_ground, 0.01],
        ],
        dtype=np.float32,
    )
    ground_triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    points.append(ground_vertices)
    counts.extend([3, 3])
    indices.extend((ground_triangles + vertex_offset).reshape(-1).tolist())
    vertex_offset += len(ground_vertices)

    for obs in obstacles:
        box_points, box_triangles = make_box_triangles(
            center=np.array([obs["x"], obs["y"], obs["z"]], dtype=np.float32),
            size=np.array([obs["size_x"], obs["size_y"], obs["size_z"]], dtype=np.float32),
            yaw=float(obs["yaw"]),
        )
        points.append(box_points)
        counts.extend([3] * len(box_triangles))
        indices.extend((box_triangles + vertex_offset).reshape(-1).tolist())
        vertex_offset += len(box_points)

    all_points = np.concatenate(points, axis=0) if points else np.zeros((0, 3), dtype=np.float32)
    mesh.GetPointsAttr().Set(all_points)
    mesh.GetFaceVertexCountsAttr().Set(counts)
    mesh.GetFaceVertexIndicesAttr().Set(indices)
    mesh.CreateSubdivisionSchemeAttr().Set("none")
    UsdGeom.Imageable(mesh).MakeInvisible()
    return mesh_root_path, mesh_path


def orientation_from_view(camera, target):
    from pxr import Gf

    camera_position = Gf.Vec3d(camera)
    target_position = Gf.Vec3d(target)
    up_axis = Gf.Vec3d(0, 0, 1)
    matrix_gf = Gf.Matrix4d(1).SetLookAt(camera_position, target_position, up_axis)
    matrix_gf = matrix_gf.GetInverse()
    quat = matrix_gf.ExtractRotationQuat()
    return (quat.real, *quat.imaginary)


def set_camera_usd_attributes(prim_path: str, focal_length: float, focus_distance: float, horizontal_aperture: float, clip_range: tuple[float, float]):
    from pxr import Sdf
    import omni.isaac.core.utils.prims as prim_utils

    prim = prim_utils.get_prim_at_path(prim_path)
    attributes = {
        "cameraProjectionType": (Sdf.ValueTypeNames.Token, "pinhole"),
        "focalLength": (Sdf.ValueTypeNames.Float, float(focal_length)),
        "focusDistance": (Sdf.ValueTypeNames.Float, float(focus_distance)),
        "horizontalAperture": (Sdf.ValueTypeNames.Float, float(horizontal_aperture)),
        "clippingRange": (Sdf.ValueTypeNames.Float2, tuple(float(v) for v in clip_range)),
    }
    for name, (value_type, value) in attributes.items():
        attr = prim.GetAttribute(name)
        if not attr.IsValid():
            attr = prim.CreateAttribute(name, value_type)
        attr.Set(value)


@dataclass
class CameraFrame:
    rgb: np.ndarray
    depth: np.ndarray


class D435CameraRig:
    def __init__(
        self,
        resolution: tuple[int, int],
        offset: tuple[float, float, float],
        target: tuple[float, float, float],
        focal_length: float,
        focus_distance: float,
        horizontal_aperture: float,
        clip_range: tuple[float, float],
    ) -> None:
        self.resolution = tuple(int(v) for v in resolution)
        self.offset = tuple(float(v) for v in offset)
        self.target = tuple(float(v) for v in target)
        self.focal_length = float(focal_length)
        self.focus_distance = float(focus_distance)
        self.horizontal_aperture = float(horizontal_aperture)
        self.clip_range = tuple(float(v) for v in clip_range)
        self.render_products = []
        self.annotators = []
        self.prim_paths: list[str] = []

    def spawn(self, prim_paths: list[str]) -> None:
        import omni.isaac.core.utils.prims as prim_utils

        self.prim_paths = prim_paths
        orientation = orientation_from_view(self.offset, self.target)
        for prim_path in prim_paths:
            if prim_utils.is_prim_path_valid(prim_path):
                raise RuntimeError(f"Duplicate camera prim at {prim_path}.")
            prim_utils.create_prim(
                prim_path,
                prim_type="Camera",
                translation=self.offset,
                orientation=orientation,
            )
            set_camera_usd_attributes(
                prim_path,
                focal_length=self.focal_length,
                focus_distance=self.focus_distance,
                horizontal_aperture=self.horizontal_aperture,
                clip_range=self.clip_range,
            )

    def initialize(self, rep, sim) -> None:
        for prim_path in self.prim_paths:
            render_product = rep.create.render_product(prim_path, self.resolution)
            rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera", device="cpu")
            rgb_annotator.attach([render_product])
            depth_annotator.attach([render_product])
            self.render_products.append(render_product)
            self.annotators.append({"rgb": rgb_annotator, "depth": depth_annotator})

        for _ in range(2):
            sim.render()

    @staticmethod
    def _convert_buffer(data: Any, dtype: np.dtype) -> np.ndarray:
        array = np.asarray(data)
        if array.dtype == np.object_ and hasattr(data, "shape"):
            array = np.frombuffer(data, dtype=dtype).reshape(*data.shape)
        return np.ascontiguousarray(array)

    def capture(self) -> CameraFrame:
        rgb_frames = []
        depth_frames = []
        for annotators in self.annotators:
            rgb = self._convert_buffer(annotators["rgb"].get_data(), np.uint8)
            depth = self._convert_buffer(annotators["depth"].get_data(), np.float32)

            if rgb.ndim == 3 and rgb.shape[-1] >= 3:
                rgb = rgb[..., :3]
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            depth = np.nan_to_num(depth, nan=self.clip_range[1], posinf=self.clip_range[1], neginf=self.clip_range[0])
            depth = depth.astype(np.float32, copy=False)

            rgb_frames.append(rgb.astype(np.uint8, copy=False))
            depth_frames.append(depth)

        return CameraFrame(
            rgb=np.stack(rgb_frames, axis=0),
            depth=np.stack(depth_frames, axis=0),
        )


@dataclass
class RtxLidarFrame:
    range_image: np.ndarray
    hit_type: np.ndarray
    peer_id: np.ndarray


class RtxLidarRig:
    def __init__(
        self,
        config_name: str,
        offset: tuple[float, float, float],
        forward_tilt_deg: float,
        range_limits: tuple[float, float],
    ) -> None:
        self.config_name = Path(config_name).stem
        self.offset = tuple(float(v) for v in offset)
        self.forward_tilt_deg = float(forward_tilt_deg)
        self.range_limits = tuple(float(v) for v in range_limits)
        self.profile = load_rtx_lidar_profile(config_name)
        self.sensors = []
        self.prim_paths: list[str] = []
        self.num_rows: int | None = None
        self.num_cols: int | None = None
        self.horizontal_resolution_deg: float | None = None

    def spawn(self, prim_paths: list[str], LidarRtx) -> None:
        self.prim_paths = prim_paths
        orientation = np.asarray(quat_from_euler_deg(0.0, self.forward_tilt_deg, 0.0), dtype=np.float32)
        translation = np.asarray(self.offset, dtype=np.float32)
        for idx, prim_path in enumerate(prim_paths):
            sensor = LidarRtx(
                prim_path=prim_path,
                name=f"mid360_{idx}",
                translation=translation,
                orientation=orientation,
                config_file_name=self.config_name,
            )
            sensor.add_linear_depth_data_to_frame()
            sensor.initialize()
            self.sensors.append(sensor)

    def initialize(self, sim) -> None:
        if not self.sensors:
            raise RuntimeError("RTX lidar rig has no sensors to initialize.")
        for _ in range(3):
            sim.render()
        first_sensor = self.sensors[0]
        self.num_rows = int(first_sensor.get_num_rows())
        self.num_cols = int(first_sensor.get_num_cols())
        self.horizontal_resolution_deg = float(first_sensor.get_horizontal_resolution())
        if self.num_rows <= 0 or self.num_cols <= 0:
            raise RuntimeError(
                f"RTX lidar produced invalid resolution rows={self.num_rows} cols={self.num_cols} for config {self.config_name}"
            )

    def ray_directions_local(self) -> np.ndarray:
        if self.num_cols is None:
            raise RuntimeError("RTX lidar rig must be initialized before computing ray directions.")
        return compute_rtx_ray_directions_local(self.profile, self.num_cols)

    def capture(self) -> RtxLidarFrame:
        if self.num_rows is None or self.num_cols is None:
            raise RuntimeError("RTX lidar rig must be initialized before capture.")

        range_images = []
        for sensor in self.sensors:
            data = sensor.get_current_frame()
            depth = np.asarray(data["linear_depth_data"], dtype=np.float32)
            if depth.ndim == 1:
                if depth.size != self.num_rows * self.num_cols:
                    raise RuntimeError(
                        f"Unexpected RTX lidar flat scan size {depth.size}; expected {self.num_rows * self.num_cols}."
                    )
                depth = depth.reshape(self.num_rows, self.num_cols)
            elif depth.ndim == 2:
                if depth.shape == (self.num_cols, self.num_rows):
                    depth = depth.T
                elif depth.shape != (self.num_rows, self.num_cols):
                    depth = depth.reshape(self.num_rows, self.num_cols)
            else:
                depth = depth.reshape(self.num_rows, self.num_cols)

            depth = np.nan_to_num(
                depth,
                nan=self.range_limits[1],
                posinf=self.range_limits[1],
                neginf=self.range_limits[0],
            )
            depth = np.clip(depth, self.range_limits[0], self.range_limits[1]).astype(np.float32, copy=False)
            range_images.append(depth.T)

        lidar_range = np.stack(range_images, axis=0)
        hit_type = (lidar_range < (self.range_limits[1] - 1e-4)).astype(np.uint8, copy=False)
        peer_id = np.full(lidar_range.shape, -1, dtype=np.int16)
        return RtxLidarFrame(range_image=lidar_range, hit_type=hit_type, peer_id=peer_id)


class SwarmH5Writer:
    def __init__(self, h5py, output_path: Path, compression: str):
        self.h5py = h5py
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(str(output_path), "w")
        self.compression = None if compression == "none" else compression
        self.datasets: dict[str, Any] = {}

    def write_metadata(
        self,
        args: argparse.Namespace,
        scenario_path: Path,
        scenario: dict,
        track_data: list[dict],
        obstacles: list[dict],
        ray_dirs_local: np.ndarray,
    ) -> None:
        root = self.file
        root.attrs["dataset_version"] = "swarm_collect_v1"
        root.attrs["source_json"] = str(scenario_path)
        root.attrs["sim_dt"] = float(args.sim_dt)
        root.attrs["num_drones"] = len(track_data)
        root.attrs["num_obstacles"] = len(obstacles)
        root.attrs["compression"] = "none" if self.compression is None else self.compression
        root.attrs["lidar_model"] = str((scenario.get("sensor") or {}).get("model", "mid360"))
        root.attrs["lidar_backend"] = str(getattr(args, "lidar_backend", "raycast"))
        if getattr(args, "rtx_lidar_config", None):
            root.attrs["lidar_config_name"] = str(args.rtx_lidar_config)
        root.attrs["lidar_hres"] = float(args.lidar_hres)
        root.attrs["lidar_vbeams"] = int(args.lidar_vbeams)
        root.attrs["lidar_min_range"] = float(args.lidar_min_range)
        root.attrs["lidar_max_range"] = float(args.lidar_max_range)
        root.attrs["lidar_forward_tilt_deg"] = float(args.lidar_forward_tilt_deg)
        root.attrs["camera_model"] = "D435"
        root.attrs["camera_resolution"] = np.asarray(args.camera_resolution, dtype=np.int32)

        scenario_group = root.require_group("scenario")
        obstacle_centers = np.asarray([[obs["x"], obs["y"], obs["z"]] for obs in obstacles], dtype=np.float32)
        obstacle_sizes = np.asarray([[obs["size_x"], obs["size_y"], obs["size_z"]] for obs in obstacles], dtype=np.float32)
        obstacle_yaws = np.asarray([obs["yaw"] for obs in obstacles], dtype=np.float32)
        scenario_group.create_dataset("obstacle_centers", data=obstacle_centers)
        scenario_group.create_dataset("obstacle_sizes", data=obstacle_sizes)
        scenario_group.create_dataset("obstacle_yaws", data=obstacle_yaws)
        if args.store_global_cloud and isinstance(get_nested(scenario, "scenario", "global_cloud_world"), list):
            scenario_group.create_dataset(
                "global_cloud_world",
                data=np.asarray(get_nested(scenario, "scenario", "global_cloud_world"), dtype=np.float32),
                compression=self.compression,
            )

        sensors_group = root.require_group("sensor")
        sensors_group.create_dataset("lidar_ray_directions_local", data=ray_dirs_local.astype(np.float32))
        sensors_group.create_dataset("peer_box_size", data=np.asarray(args.peer_box_size, dtype=np.float32))
        sensors_group.create_dataset("lidar_offset", data=np.asarray(args.lidar_offset, dtype=np.float32))
        sensors_group.create_dataset("camera_offset", data=np.asarray(args.camera_offset, dtype=np.float32))
        sensors_group.create_dataset("camera_target", data=np.asarray(args.camera_target, dtype=np.float32))

        episode = root.require_group("episodes/000000")
        episode.attrs["num_drones"] = len(track_data)
        episode.create_dataset("drone_id", data=np.asarray([track["id"] for track in track_data], dtype=np.int32))
        episode.create_dataset("start", data=np.asarray([track["start"] for track in track_data], dtype=np.float32))
        episode.create_dataset("goal", data=np.asarray([track["goal"] for track in track_data], dtype=np.float32))
        episode.create_dataset(
            "initial_orientation_wxyz",
            data=np.asarray([track["initial_orientation_wxyz"] for track in track_data], dtype=np.float32),
        )
        episode.create_dataset(
            "track_end_time",
            data=np.asarray([track["timestamps"][-1] for track in track_data], dtype=np.float32),
        )

    def _ensure_dataset(self, name: str, sample: np.ndarray):
        if name in self.datasets:
            return self.datasets[name]

        group = self.file
        parts = name.split("/")
        for part in parts[:-1]:
            group = group.require_group(part)
        dataset_name = parts[-1]
        shape = (0,) + sample.shape
        maxshape = (None,) + sample.shape
        chunks = (1,) + sample.shape
        dataset = group.create_dataset(
            dataset_name,
            shape=shape,
            maxshape=maxshape,
            chunks=chunks,
            dtype=sample.dtype,
            compression=self.compression,
        )
        self.datasets[name] = dataset
        return dataset

    def append(self, name: str, sample: np.ndarray | torch.Tensor | float | int | bool) -> None:
        if isinstance(sample, torch.Tensor):
            sample = sample.detach().cpu().numpy()
        sample_np = np.asarray(sample)
        dataset = self._ensure_dataset(name, sample_np)
        next_idx = dataset.shape[0]
        dataset.resize(next_idx + 1, axis=0)
        dataset[next_idx] = sample_np

    def close(self) -> None:
        self.file.flush()
        self.file.close()


def rotate_local_vectors(quat_wxyz: torch.Tensor, local_vectors: torch.Tensor) -> torch.Tensor:
    from omni_drones.utils.torch import quaternion_to_rotation_matrix

    rot = quaternion_to_rotation_matrix(quat_wxyz)
    return torch.einsum("nij,nrj->nri", rot, local_vectors)


def rotate_local_points(quat_wxyz: torch.Tensor, local_points: torch.Tensor) -> torch.Tensor:
    from omni_drones.utils.torch import quaternion_to_rotation_matrix

    rot = quaternion_to_rotation_matrix(quat_wxyz)
    return torch.einsum("nij,nrj->nri", rot, local_points)


def ray_obb_intersections_local(
    ray_origins_local: torch.Tensor,
    ray_dirs_local: torch.Tensor,
    half_extents: torch.Tensor,
    min_range: float,
    max_range: float,
) -> torch.Tensor:
    eps = 1e-8
    parallel = ray_dirs_local.abs() <= eps
    outside_parallel = parallel & ((ray_origins_local < -half_extents) | (ray_origins_local > half_extents))

    safe_inv = torch.where(parallel, torch.full_like(ray_dirs_local, 1e8), 1.0 / ray_dirs_local)
    t1 = (-half_extents - ray_origins_local) * safe_inv
    t2 = (half_extents - ray_origins_local) * safe_inv
    t_min = torch.minimum(t1, t2).amax(dim=-1)
    t_max = torch.maximum(t1, t2).amin(dim=-1)

    t_hit = torch.where(t_min >= 0.0, t_min, t_max)
    invalid = outside_parallel.any(dim=-1) | (t_max < 0.0) | (t_min > t_max)
    valid_range = (t_hit >= min_range) & (t_hit <= max_range)
    return torch.where(invalid | (~valid_range), torch.full_like(t_hit, float("inf")), t_hit)


def compute_peer_overlay(
    ray_starts_w: torch.Tensor,
    ray_dirs_w: torch.Tensor,
    root_state: torch.Tensor,
    peer_box_size: torch.Tensor,
    min_range: float,
    max_range: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from omni_drones.utils.torch import quaternion_to_rotation_matrix

    num_drones, num_rays, _ = ray_dirs_w.shape
    peer_dist = torch.full((num_drones, num_rays), float("inf"), device=ray_dirs_w.device)
    peer_id = torch.full((num_drones, num_rays), -1, dtype=torch.int16, device=ray_dirs_w.device)

    peer_centers = root_state[:, :3]
    peer_rot = quaternion_to_rotation_matrix(root_state[:, 3:7])
    half_extents = peer_box_size.view(1, 3) * 0.5

    for peer_idx in range(num_drones):
        ego_mask = torch.arange(num_drones, device=ray_dirs_w.device) != peer_idx
        if not torch.any(ego_mask):
            continue
        ego_indices = torch.nonzero(ego_mask, as_tuple=False).squeeze(-1)
        rel_origins = ray_starts_w[ego_indices] - peer_centers[peer_idx].view(1, 1, 3)
        world_to_peer = peer_rot[peer_idx].transpose(0, 1)
        origins_local = torch.einsum("mri,ij->mrj", rel_origins, world_to_peer)
        dirs_local = torch.einsum("mri,ij->mrj", ray_dirs_w[ego_indices], world_to_peer)
        dist = ray_obb_intersections_local(origins_local, dirs_local, half_extents, min_range, max_range)
        better = dist < peer_dist[ego_indices]
        peer_dist[ego_indices] = torch.where(better, dist, peer_dist[ego_indices])
        peer_id[ego_indices] = torch.where(
            better,
            torch.full_like(peer_id[ego_indices], peer_idx),
            peer_id[ego_indices],
        )

    return peer_dist, peer_id


def combine_lidar_hits(
    static_dist: torch.Tensor,
    peer_dist: torch.Tensor,
    peer_id: torch.Tensor,
    max_range: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    final_dist = static_dist.clone()
    final_peer_id = torch.full_like(peer_id, -1)
    hit_type = torch.zeros_like(peer_id, dtype=torch.uint8)

    static_hit = static_dist < max_range
    hit_type[static_hit] = 1

    peer_better = peer_dist < final_dist
    final_dist = torch.where(peer_better, peer_dist, final_dist)
    final_peer_id = torch.where(peer_better, peer_id, final_peer_id)
    hit_type = torch.where(peer_better, torch.full_like(hit_type, 2), hit_type)

    no_hit = final_dist >= max_range
    final_dist = torch.where(no_hit, torch.full_like(final_dist, max_range), final_dist)
    hit_type = torch.where(no_hit, torch.zeros_like(hit_type), hit_type)
    final_peer_id = torch.where(no_hit, torch.full_like(final_peer_id, -1), final_peer_id)
    return final_dist, hit_type, final_peer_id


def compute_navrl_state(root_state: torch.Tensor, goal_pos: torch.Tensor, goal_dir_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from utils import vec_to_new_frame

    rpos = goal_pos - root_state[:, :3]
    distance = rpos.norm(dim=-1, keepdim=True)
    distance_2d = rpos[:, :2].norm(dim=-1, keepdim=True)
    distance_z = rpos[:, 2:3]

    rpos_clipped = rpos / distance.clamp_min(1e-6)
    rpos_clipped_g = vec_to_new_frame(rpos_clipped, goal_dir_world).squeeze(1)
    vel_g = vec_to_new_frame(root_state[:, 7:10], goal_dir_world).squeeze(1)
    drone_state = torch.cat([rpos_clipped_g, distance_2d, distance_z, vel_g], dim=-1)
    return drone_state, rpos


def compute_rewards_and_flags(
    root_state: torch.Tensor,
    goal_pos: torch.Tensor,
    lidar_navrl: torch.Tensor,
    lidar_max_range: float,
    prev_vel_w: torch.Tensor,
    height_range: torch.Tensor,
    final_step: bool,
) -> dict[str, torch.Tensor]:
    rpos = goal_pos - root_state[:, :3]
    distance = rpos.norm(dim=-1, keepdim=True)
    vel_direction = rpos / distance.clamp_min(1e-6)
    reward_vel = (root_state[:, 7:10] * vel_direction).sum(dim=-1, keepdim=True)
    reward_safety_static = torch.log((lidar_max_range - lidar_navrl).clamp(min=1e-6, max=lidar_max_range)).mean(dim=(1, 2), keepdim=True)
    penalty_smooth = (root_state[:, 7:10] - prev_vel_w).norm(dim=-1, keepdim=True)

    penalty_height = torch.zeros_like(distance)
    above_height = root_state[:, 2:3] > (height_range[:, 1:2] + 0.2)
    below_height = root_state[:, 2:3] < (height_range[:, 0:1] - 0.2)
    penalty_height[above_height] = (root_state[:, 2:3] - height_range[:, 1:2] - 0.2)[above_height] ** 2
    penalty_height[below_height] = (height_range[:, 0:1] - 0.2 - root_state[:, 2:3])[below_height] ** 2

    reward = reward_vel + 1.0 + reward_safety_static - penalty_smooth * 0.1 - penalty_height * 8.0

    collision = torch.amax(lidar_navrl, dim=(1, 2), keepdim=True) > (lidar_max_range - 0.3)
    reach_goal = distance < 0.5
    below_bound = root_state[:, 2:3] < 0.2
    above_bound = root_state[:, 2:3] > 4.0
    terminated = below_bound | above_bound | collision
    truncated = torch.full_like(terminated, final_step)
    done = terminated | truncated

    return {
        "reward": reward,
        "collision": collision,
        "reach_goal": reach_goal,
        "terminated": terminated,
        "truncated": truncated,
        "done": done,
    }


def tensor_to_np(tensor: torch.Tensor, dtype: np.dtype | None = None) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def main() -> None:
    args = parse_args()
    scenario_path = Path(args.json).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    scenario = load_scenario(scenario_path)
    apply_scenario_sensor_defaults(args, scenario)
    if args.lidar_attach_yaw_only:
        raise NotImplementedError("--lidar-attach-yaw-only is not supported in this collector yet.")
    obstacles, obstacle_types = extract_box_obstacles(scenario)

    from omni.isaac.kit import SimulationApp

    app_experience = None
    if "EXP_PATH" in os.environ:
        if args.headless:
            app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
        else:
            app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"

    simulation_app = (
        SimulationApp({"headless": args.headless, "anti_aliasing": args.anti_aliasing}, experience=app_experience)
        if app_experience
        else SimulationApp({"headless": args.headless, "anti_aliasing": args.anti_aliasing})
    )
    print(f"SimulationApp experience: {app_experience or 'default'}")

    writer = None
    try:
        import omni
        import omni.isaac.orbit.sim as sim_utils
        import omni.isaac.core.utils.prims as prim_utils
        from omni.isaac.core.simulation_context import SimulationContext
        from omni.isaac.core.utils.extensions import enable_extension
        from omni.isaac.core.utils.semantics import add_update_semantics
        from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
        from omni_drones.controllers import LeePositionController
        from omni_drones.robots.drone import MultirotorBase
        from omni_drones.utils.torch import euler_to_quaternion
        from utils import vec_to_new_frame

        h5py = load_h5py()

        if not args.headless:
            enable_extension("omni.kit.viewport.rtx")
            enable_extension("omni.kit.viewport.pxr")
            enable_extension("omni.kit.viewport.bundle")
            from omni.isaac.core.utils.viewports import set_camera_view
        else:
            set_camera_view = None

        rep = None
        if args.camera_enabled:
            enable_extension("omni.replicator.isaac")
            import omni.replicator.core as rep
        if args.lidar_backend == "rtx":
            enable_extension("omni.isaac.sensor")

        if args.lidar_backend == "raycast":
            from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
        else:
            from omni.isaac.sensor import LidarRtx

        args.drone_model = resolve_drone_model_name(args.drone_model, MultirotorBase.REGISTRY)
        sim_device = resolve_sim_device(args.device)

        sim = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=args.sim_dt,
            rendering_dt=args.sim_dt,
            backend="torch",
            device=sim_device,
        )
        stage = sim.stage
        sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(args.ground_size, args.ground_size)).func(
            "/World/defaultGroundPlane",
            sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(args.ground_size, args.ground_size)),
            translation=(0, 0, 0.01),
        )
        sim_utils.DistantLightCfg(color=(0.8, 0.8, 0.8), intensity=3500.0).func(
            "/World/Light",
            sim_utils.DistantLightCfg(color=(0.8, 0.8, 0.8), intensity=3500.0),
            translation=(1.0, 0.0, 10.0),
        )
        sim_utils.DomeLightCfg(color=(0.25, 0.25, 0.30), intensity=1500.0).func(
            "/World/SkyLight",
            sim_utils.DomeLightCfg(color=(0.25, 0.25, 0.30), intensity=1500.0),
        )

        stage.DefinePrim("/World/Scenario", "Xform")
        stage.DefinePrim("/World/Scenario/Obstacles", "Xform")

        static_mesh_root = None
        if args.lidar_backend == "raycast":
            static_mesh_root, _ = create_static_scene_mesh(stage, "/World/Scenario/LidarMesh", obstacles, args.ground_size)

        static_obstacles = []
        static_obstacle_states = []
        obstacle_body_material, obstacle_cap_material = make_obstacle_materials(sim_utils, args.obstacle_material)

        for idx, obs in enumerate(obstacles):
            prim_path = f"/World/Scenario/Obstacles/Obstacle_{idx:03d}"
            cap_path = f"/World/Scenario/Obstacles/ObstacleCap_{idx:03d}"
            yaw = float(obs.get("yaw", 0.0))
            size_x = float(obs["size_x"])
            size_y = float(obs["size_y"])
            size_z = float(obs["size_z"])

            obstacle_cfg = RigidObjectCfg(
                prim_path=prim_path,
                spawn=sim_utils.CuboidCfg(
                    size=(size_x, size_y, size_z),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=True,
                        disable_gravity=True,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    visual_material=obstacle_body_material,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(float(obs["x"]), float(obs["y"]), float(obs["z"])),
                    rot=(math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)),
                ),
            )

            cap_thickness = min(0.08, max(0.04, size_z * 0.08))
            cap_z = float(obs["z"]) + 0.5 * size_z + 0.5 * cap_thickness
            cap_cfg = RigidObjectCfg(
                prim_path=cap_path,
                spawn=sim_utils.CuboidCfg(
                    size=(size_x, size_y, cap_thickness),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=True,
                        disable_gravity=True,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    visual_material=obstacle_cap_material,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(float(obs["x"]), float(obs["y"]), cap_z),
                    rot=(math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)),
                ),
            )

            static_obstacles.append(RigidObject(cfg=obstacle_cfg))
            static_obstacle_states.append(
                [
                    float(obs["x"]),
                    float(obs["y"]),
                    float(obs["z"]),
                    math.cos(yaw * 0.5),
                    0.0,
                    0.0,
                    math.sin(yaw * 0.5),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )

            static_obstacles.append(RigidObject(cfg=cap_cfg))
            static_obstacle_states.append(
                [
                    float(obs["x"]),
                    float(obs["y"]),
                    cap_z,
                    math.cos(yaw * 0.5),
                    0.0,
                    0.0,
                    math.sin(yaw * 0.5),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )

        drone_cls = MultirotorBase.REGISTRY[args.drone_model]
        drone = drone_cls()

        track_data, start_positions, init_yaws = extract_track_data(scenario)
        if not track_data:
            raise RuntimeError("No non-empty trajectories found in scenario JSON.")

        drone.spawn(translations=start_positions)
        sim.reset()
        drone.initialize()

        n = len(track_data)
        if static_obstacles:
            obstacle_state_tensor = torch.tensor(static_obstacle_states, dtype=torch.float32, device=sim.device)
            for obstacle_asset, obstacle_state in zip(static_obstacles, obstacle_state_tensor):
                obstacle_asset.write_root_state_to_sim(obstacle_state.unsqueeze(0))
                obstacle_asset.write_data_to_sim()
                obstacle_asset.update(args.sim_dt)

        init_pos_np = np.stack([track["positions"][0] for track in track_data], axis=0)
        init_pos = torch.tensor(init_pos_np, dtype=torch.float32, device=sim.device)
        init_rpy = torch.zeros((n, 3), dtype=torch.float32, device=sim.device)
        init_rpy[:, 2] = torch.tensor(init_yaws, dtype=torch.float32, device=sim.device)
        init_rot = euler_to_quaternion(init_rpy)
        init_vels = torch.zeros((n, 6), dtype=torch.float32, device=sim.device)

        drone.set_world_poses(init_pos, init_rot)
        drone.set_velocities(init_vels)
        sim._physics_sim_view.flush()

        for drone_idx in range(n):
            drone_prim_path = f"/World/envs/env_0/{drone.name}_{drone_idx}"
            obstacle_prim_path = f"/World/Scenario/Obstacles/Obstacle_{drone_idx:03d}"
            if prim_utils.is_prim_path_valid(drone_prim_path):
                add_update_semantics(prim_utils.get_prim_at_path(drone_prim_path), "drone", "class")
            if prim_utils.is_prim_path_valid(obstacle_prim_path):
                add_update_semantics(prim_utils.get_prim_at_path(obstacle_prim_path), "obstacle", "class")

        lidar_resolution = None
        lidar_ray_dirs_local = None
        rtx_lidar_rig = None
        if args.lidar_backend == "raycast":
            lidar_quat = quat_from_euler_deg(0.0, args.lidar_forward_tilt_deg, 0.0)
            vertical_ray_angles = torch.linspace(
                -float(args.lidar_vfov_down_deg),
                float(args.lidar_vfov_up_deg),
                int(args.lidar_vbeams),
            )
            lidar_cfg = RayCasterCfg(
                prim_path=f"/World/envs/env_0/{drone.name}_*/base_link",
                offset=RayCasterCfg.OffsetCfg(pos=tuple(float(v) for v in args.lidar_offset), rot=lidar_quat),
                attach_yaw_only=bool(args.lidar_attach_yaw_only),
                pattern_cfg=patterns.BpearlPatternCfg(
                    horizontal_res=float(args.lidar_hres),
                    vertical_ray_angles=vertical_ray_angles,
                ),
                debug_vis=False,
                mesh_prim_paths=[static_mesh_root],
                max_distance=float(args.lidar_max_range),
            )
            lidar = RayCaster(lidar_cfg)
            lidar._initialize_impl()
            lidar_hbeams = int(round(360.0 / float(args.lidar_hres)))
            lidar_resolution = (lidar_hbeams, int(args.lidar_vbeams))
            lidar_ray_dirs_local = tensor_to_np(lidar.ray_directions[0].reshape(*lidar_resolution, 3), np.float32)
        else:
            if not args.rtx_lidar_config:
                raise RuntimeError("RTX lidar backend requires --rtx-lidar-config or a matching scenario sensor.model default.")
            lidar = None
            lidar_paths = [f"/World/envs/env_0/{drone.name}_{i}/base_link/mid360_lidar" for i in range(n)]
            print(f"Initializing RTX lidar rig with config {Path(args.rtx_lidar_config).stem}...")
            rtx_lidar_rig = RtxLidarRig(
                config_name=args.rtx_lidar_config,
                offset=tuple(float(v) for v in args.lidar_offset),
                forward_tilt_deg=float(args.lidar_forward_tilt_deg),
                range_limits=(float(args.lidar_min_range), float(args.lidar_max_range)),
            )
            rtx_lidar_rig.spawn(lidar_paths, LidarRtx)
            print("RTX lidar rig spawned.")

        camera_rig = None
        if args.camera_enabled:
            print("Initializing D435 camera rig...")
            camera_paths = [f"/World/envs/env_0/{drone.name}_{i}/base_link/D435" for i in range(n)]
            camera_rig = D435CameraRig(
                resolution=tuple(args.camera_resolution),
                offset=tuple(float(v) for v in args.camera_offset),
                target=tuple(float(v) for v in args.camera_target),
                focal_length=float(args.camera_focal_length),
                focus_distance=float(args.camera_focus_distance),
                horizontal_aperture=float(args.camera_horizontal_aperture),
                clip_range=tuple(float(v) for v in args.camera_clip_range),
            )
            camera_rig.spawn(camera_paths)
            camera_rig.initialize(rep, sim)
            print("D435 camera rig ready.")

        controller = LeePositionController(g=9.81, uav_params=drone.params).to(sim.device)

        viewer_eye, viewer_target = compute_viewer_camera_pose(obstacles, track_data)
        if not args.headless:
            set_camera_view(eye=viewer_eye, target=viewer_target)

        if args.save_stage:
            save_path = str(Path(args.save_stage).expanduser().resolve())
            omni.usd.get_context().save_as_stage(save_path)
            print(f"Saved generated stage to {save_path}")

        total_duration = track_duration(track_data)
        total_steps = int(math.floor(total_duration / args.sim_dt)) + 1
        fixed_goal_dirs_np = make_goal_direction(track_data)
        fixed_goal_dirs = torch.tensor(fixed_goal_dirs_np, dtype=torch.float32, device=sim.device)
        goal_pos = torch.tensor(np.stack([track["goal"] for track in track_data], axis=0), dtype=torch.float32, device=sim.device)
        height_range = torch.stack(
            [
                torch.minimum(init_pos[:, 2], goal_pos[:, 2]),
                torch.maximum(init_pos[:, 2], goal_pos[:, 2]),
            ],
            dim=-1,
        )
        prev_vel_w = torch.zeros((n, 3), dtype=torch.float32, device=sim.device)
        peer_box_size = None
        if args.lidar_backend == "raycast":
            peer_box_size = torch.tensor(args.peer_box_size, dtype=torch.float32, device=sim.device)


        print(f"Loaded scenario: {scenario_path}")
        if args.config:
            print(f"Loaded defaults from config: {Path(args.config).expanduser().resolve()}")
        print(
            f"Loaded {len(obstacles)} replay obstacles."
            + (f" Raw obstacle types: {', '.join(obstacle_types)}" if obstacle_types else "")
        )
        print(f"Collecting {n} drones for {total_duration:.2f}s into {output_path}")
        print(
            "LiDAR:"
            f" backend={args.lidar_backend}"
            f" model={str((scenario.get('sensor') or {}).get('model', 'mid360'))}"
            + (f" config={Path(args.rtx_lidar_config).stem}" if args.lidar_backend == "rtx" and args.rtx_lidar_config else "")
            + f" hres={args.lidar_hres:.2f}deg"
            + f" vbeams={args.lidar_vbeams}"
            + f" range=({args.lidar_min_range:.2f}, {args.lidar_max_range:.2f})"
            + f" tilt={args.lidar_forward_tilt_deg:.2f}deg"
        )
        if args.camera_enabled:
            print(
                "Camera:"
                f" model=D435"
                f" resolution={args.camera_resolution[0]}x{args.camera_resolution[1]}"
                f" focal_length={args.camera_focal_length:.2f}mm"
            )
        else:
            print("Camera: disabled")

        sim.play()
        sim.render()
        if args.lidar_backend == "rtx":
            rtx_lidar_rig.initialize(sim)
            lidar_resolution = (rtx_lidar_rig.num_cols, rtx_lidar_rig.num_rows)
            args.lidar_hres = float(rtx_lidar_rig.horizontal_resolution_deg)
            args.lidar_vbeams = int(rtx_lidar_rig.num_rows)
            lidar_ray_dirs_local = rtx_lidar_rig.ray_directions_local()
            print(f"RTX LiDAR resolved {rtx_lidar_rig.num_cols}x{rtx_lidar_rig.num_rows} from config {Path(args.rtx_lidar_config).stem}")

        writer = SwarmH5Writer(h5py, output_path, args.compression)
        writer.write_metadata(
            args=args,
            scenario_path=scenario_path,
            scenario=scenario,
            track_data=track_data,
            obstacles=obstacles,
            ray_dirs_local=lidar_ray_dirs_local,
        )

        for step_idx in range(total_steps):
            if not simulation_app.is_running():
                break
            if sim.is_stopped():
                break
            if not sim.is_playing():
                sim.render()
                continue

            sim_t = min(step_idx * args.sim_dt, total_duration)
            ref_pos_np = []
            ref_vel_np = []
            ref_acc_np = []
            ref_yaw_np = []
            active_mask_np = []
            for track in track_data:
                pos, vel, acc, yaw, active = sample_track(track, sim_t)
                ref_pos_np.append(pos)
                ref_vel_np.append(vel)
                ref_acc_np.append(acc)
                ref_yaw_np.append(yaw)
                active_mask_np.append(active)

            ref_pos = torch.tensor(np.stack(ref_pos_np, axis=0), dtype=torch.float32, device=sim.device)
            ref_vel = torch.tensor(np.stack(ref_vel_np, axis=0), dtype=torch.float32, device=sim.device)
            ref_acc = torch.tensor(np.stack(ref_acc_np, axis=0), dtype=torch.float32, device=sim.device)
            ref_yaw = torch.tensor(np.asarray(ref_yaw_np, dtype=np.float32), dtype=torch.float32, device=sim.device)
            active_mask = torch.tensor(np.asarray(active_mask_np, dtype=bool), dtype=torch.bool, device=sim.device)

            root_state = drone.get_state()[..., :13].squeeze(0)
            if args.lidar_backend == "raycast":
                lidar.update(args.sim_dt)

                ray_origins_w = lidar.data.pos_w.unsqueeze(1) + rotate_local_points(lidar.data.quat_w, lidar.ray_starts)
                ray_dirs_w = rotate_local_vectors(lidar.data.quat_w, lidar.ray_directions)
                static_dist = (lidar.data.ray_hits_w - ray_origins_w).norm(dim=-1)
                static_valid = torch.isfinite(static_dist)
                static_dist = torch.where(
                    static_valid,
                    static_dist.clamp(min=float(args.lidar_min_range), max=float(args.lidar_max_range)),
                    torch.full_like(static_dist, float(args.lidar_max_range)),
                )

                peer_dist, peer_id = compute_peer_overlay(
                    ray_starts_w=ray_origins_w,
                    ray_dirs_w=ray_dirs_w,
                    root_state=root_state,
                    peer_box_size=peer_box_size,
                    min_range=float(args.lidar_min_range),
                    max_range=float(args.lidar_max_range),
                )
                final_dist, hit_type, final_peer_id = combine_lidar_hits(
                    static_dist=static_dist,
                    peer_dist=peer_dist,
                    peer_id=peer_id,
                    max_range=float(args.lidar_max_range),
                )
                lidar_range = final_dist.reshape(n, *lidar_resolution)
                lidar_navrl = (float(args.lidar_max_range) - final_dist).reshape(n, 1, *lidar_resolution)
                lidar_hit_type_np = tensor_to_np(hit_type.reshape(n, *lidar_resolution), np.uint8)
                lidar_peer_id_np = tensor_to_np(final_peer_id.reshape(n, *lidar_resolution), np.int16)
            else:
                lidar_frame = rtx_lidar_rig.capture()
                lidar_range = torch.as_tensor(lidar_frame.range_image, dtype=torch.float32, device=sim.device)
                lidar_navrl = (float(args.lidar_max_range) - lidar_range).unsqueeze(1)
                lidar_hit_type_np = lidar_frame.hit_type
                lidar_peer_id_np = lidar_frame.peer_id

            navrl_state, _ = compute_navrl_state(root_state, goal_pos, fixed_goal_dirs)
            goal_direction = fixed_goal_dirs.unsqueeze(1)
            action_local = vec_to_new_frame(ref_vel, fixed_goal_dirs).squeeze(1)
            step_flags = compute_rewards_and_flags(
                root_state=root_state,
                goal_pos=goal_pos,
                lidar_navrl=lidar_navrl.squeeze(1),
                lidar_max_range=float(args.lidar_max_range),
                prev_vel_w=prev_vel_w,
                height_range=height_range,
                final_step=(step_idx == total_steps - 1),
            )
            camera_frame = camera_rig.capture() if camera_rig is not None else None

            action = controller(
                root_state,
                target_pos=ref_pos,
                target_vel=ref_vel,
                target_acc=ref_acc,
                target_yaw=ref_yaw,
            )

            writer.append("episodes/000000/timestamp", np.float32(sim_t))
            writer.append("episodes/000000/active_mask", tensor_to_np(active_mask, np.bool_))
            writer.append("episodes/000000/info/root_state", tensor_to_np(root_state, np.float32))
            writer.append("episodes/000000/info/drone_state", tensor_to_np(root_state[:, :13], np.float32))
            writer.append("episodes/000000/observations/state", tensor_to_np(navrl_state, np.float32))
            writer.append("episodes/000000/observations/direction", tensor_to_np(goal_direction, np.float32))
            writer.append("episodes/000000/observations/lidar", tensor_to_np(lidar_navrl, np.float32))
            writer.append("episodes/000000/observations/lidar_range", tensor_to_np(lidar_range, np.float32))
            writer.append("episodes/000000/observations/lidar_hit_type", lidar_hit_type_np)
            writer.append("episodes/000000/observations/lidar_peer_id", lidar_peer_id_np)
            if camera_frame is not None:
                writer.append("episodes/000000/observations/d435_rgb", camera_frame.rgb)
                writer.append("episodes/000000/observations/d435_depth", camera_frame.depth.astype(np.float32, copy=False))

            writer.append("episodes/000000/expert/position_ref", tensor_to_np(ref_pos, np.float32))
            writer.append("episodes/000000/expert/velocity_ref", tensor_to_np(ref_vel, np.float32))
            writer.append("episodes/000000/expert/acceleration_ref", tensor_to_np(ref_acc, np.float32))
            writer.append("episodes/000000/expert/yaw_ref", tensor_to_np(ref_yaw, np.float32))
            writer.append("episodes/000000/expert/action_world", tensor_to_np(ref_vel, np.float32))
            writer.append("episodes/000000/expert/action_local", tensor_to_np(action_local, np.float32))

            writer.append("episodes/000000/reward", tensor_to_np(step_flags["reward"], np.float32))
            writer.append("episodes/000000/collision", tensor_to_np(step_flags["collision"], np.bool_))
            writer.append("episodes/000000/reach_goal", tensor_to_np(step_flags["reach_goal"], np.bool_))
            writer.append("episodes/000000/terminated", tensor_to_np(step_flags["terminated"], np.bool_))
            writer.append("episodes/000000/truncated", tensor_to_np(step_flags["truncated"], np.bool_))
            writer.append("episodes/000000/done", tensor_to_np(step_flags["done"], np.bool_))

            prev_vel_w = root_state[:, 7:10].clone()
            drone.apply_action(action)
            sim.step(render=True)

        print(f"Wrote swarm dataset to {output_path}")

    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            simulation_app.close()


if __name__ == "__main__":
    main()
