"""Fetch ARKitScenes 3dod scenes from Apple's public CDN (no account needed)."""

import shutil
import subprocess
import zipfile
from pathlib import Path

CDN = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"


def _complete(frames_dir):
    """A scene is whole when colour, depth and intrinsics all have the same non-zero frame count."""
    counts = {
        len(list((frames_dir / sub).glob(pattern)))
        for sub, pattern in (
            ("lowres_wide", "*.png"),
            ("lowres_depth", "*.png"),
            ("lowres_wide_intrinsics", "*.pincam"),
        )
    }
    return (
        len(counts) == 1
        and 0 not in counts
        and (frames_dir / "lowres_wide.traj").exists()
    )


def download_scene(video_id, split, dest):
    """Download + unzip one 3dod scene under dest/<split>/<video_id>/; returns the frames directory.
    A kill mid-unzip leaves no marker, so the next run re-downloads rather than reusing half a scene."""
    scene_dir = Path(dest) / split / str(video_id)
    frames_dir = scene_dir / f"{video_id}_frames"
    done = scene_dir / ".unzipped"
    if done.exists() or _complete(frames_dir):
        done.touch()
        return frames_dir
    shutil.rmtree(scene_dir, ignore_errors=True)
    scene_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path = scene_dir.parent / f"{video_id}.zip"
    tmp = zip_path.with_suffix(".zip.part")
    subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "-o",
            str(tmp),
            f"{CDN}/threedod/{split}/{video_id}.zip",
        ],
        check=True,
    )
    tmp.rename(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(scene_dir.parent)
    zip_path.unlink()
    done.touch()
    return frames_dir


def frame_stamps(frames_dir):
    """Sorted colour-frame timestamps ('6790.132' strings) present in a scene's frames directory."""
    return sorted(
        p.stem.split("_", 1)[1]
        for p in (Path(frames_dir) / "lowres_wide").glob("*.png")
    )


def rgb_path(frames_dir, stamp):
    vid = Path(frames_dir).name.split("_")[0]
    return Path(frames_dir) / "lowres_wide" / f"{vid}_{stamp}.png"


def depth_path(frames_dir, stamp):
    vid = Path(frames_dir).name.split("_")[0]
    return Path(frames_dir) / "lowres_depth" / f"{vid}_{stamp}.png"
