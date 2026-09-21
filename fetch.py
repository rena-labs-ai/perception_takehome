"""Download the assignment's scenes from Apple's ARKitScenes CDN and lay them out under data/<scene>/.

Colour and depth frames are copied as-is; the calibration you work from is the one in variant/ (see README).
"""

import hashlib
import json
import shutil
from pathlib import Path

from perception.arkit import depth_path, download_scene, rgb_path

ROOT = Path(__file__).resolve().parent
VARIANT, DATA = ROOT / "variant", ROOT / "data"


def _fingerprint(sync):
    return hashlib.sha256(json.dumps(sync["pairs"]).encode()).hexdigest()


def extract(scene, raw_root):
    out = DATA / scene["name"]
    sync_path = VARIANT / scene["name"] / "sync.json"
    sync = json.loads(sync_path.read_text())
    stamp = out / ".extracted_from"
    if out.exists():
        if stamp.exists() and stamp.read_text() == _fingerprint(sync):
            print(f"{scene['name']}: already extracted")
            return
        raise SystemExit(
            f"{out} was extracted from a different frame pairing. Move or delete that directory and "
            "re-run; anything you added under it would be lost, so we will not remove it for you."
        )
    frames_dir = download_scene(scene["video_id"], scene["split"], raw_root)
    tmp = DATA / f".{scene['name']}.part"
    shutil.rmtree(tmp, ignore_errors=True)
    for sub in ("rgb", "depth"):
        (tmp / sub).mkdir(parents=True)
    for i, (color_stamp, depth_stamp) in enumerate(sync["pairs"]):
        shutil.copyfile(rgb_path(frames_dir, color_stamp), tmp / "rgb" / f"{i:06d}.png")
        shutil.copyfile(
            depth_path(frames_dir, depth_stamp), tmp / "depth" / f"{i:06d}.png"
        )
    for name in ("calib.json", "boxes.json"):
        src = VARIANT / scene["name"] / name
        if src.exists():
            shutil.copyfile(src, tmp / name)
    (tmp / ".extracted_from").write_text(_fingerprint(sync))
    tmp.rename(out)
    print(f"{scene['name']}: {len(sync['pairs'])} frames -> {out}")


def main():
    scenes = json.loads((VARIANT / "scenes.json").read_text())["scenes"]
    shared = Path.home() / "datasets" / "arkitscenes" / "3dod"
    raw_root = shared if shared.is_dir() else DATA / ".arkit_raw"
    for scene in scenes:
        extract(scene, raw_root)
    if raw_root != shared:
        shutil.rmtree(raw_root, ignore_errors=True)


if __name__ == "__main__":
    main()
