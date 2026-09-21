"""Score Part 2 (answers.json) and Part 3 (instances.json) against what ships with the assignment.

Consistency and coverage are computed on every scene. Hit rate, confidence AUROC and the Part 3
instance metrics need boxes.json / query_targets.json, which ship for dev scenes only; we compute
them on the test scenes when we grade. Usage: uv run python score.py [answers.json] [instances.json]
"""

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from perception.geometry import transform, unproject
from perception.scene import Scene

ROOT = Path(__file__).resolve().parent
GOAL_MARGIN_M = 0.35
INSTANCE_MARGIN_M = 0.25


def lift_detection(scene, frame, box):
    """World point for a 2D detection: unproject the depth inside the box's central half, 20th percentile of
    range (the near surface, not the wall behind), through the scene's calibration."""
    x0, y0, x1, y1 = box
    cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 4, (y1 - y0) / 4
    depth = scene.depth_m(frame)
    h, w = depth.shape
    r0, r1 = int(max(0, cy - hh)), int(min(h, cy + hh + 1))
    c0, c1 = int(max(0, cx - hw)), int(min(w, cx + hw + 1))
    patch = depth[r0:r1, c0:c1]
    valid = patch > 0
    if valid.sum() < 10:
        return None
    z_near = np.percentile(patch[valid], 20)
    band = valid & (np.abs(patch - z_near) < 0.06)
    if not band.any():
        return None
    masked = np.zeros_like(depth)
    masked[r0:r1, c0:c1] = np.where(band, patch, 0)
    pts, _ = unproject(masked, scene.depth_cam)
    cam = transform(scene.T_color_depth, pts)
    return transform(scene.pose(frame), cam).mean(axis=0)


def in_box(point, box, margin):
    local = (np.asarray(point) - np.asarray(box["center"])) @ np.asarray(box["axes"]).T
    return bool(np.all(np.abs(local) <= np.asarray(box["size"]) / 2 + margin))


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    if labels.all() or not labels.any():
        return None
    pos, neg = scores[labels], scores[~labels]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (
        pos[:, None] == neg[None, :]
    ).sum()
    return float(wins / (len(pos) * len(neg)))


def score_part2(scene, queries, answers, targets=None):
    """queries: list from variant/queries.json; answers: {query_id: {frame: {goal_world, confidence}}}.

    hit_rate is precision over the frames you answered; goal_recall counts every frame whose target was
    visible, so declining the hard ones moves one number and not the other."""
    spreads, answered, total = [], 0, 0
    confidences, hits, abstain_on_absent, absent_frames = [], [], 0, 0
    visible_frames, found_on_visible = 0, 0
    for q in queries:
        goals = []
        for f in q["frames"]:
            total += 1
            a = answers.get(q["id"], {}).get(str(f))
            goal = a.get("goal_world") if a else None
            key = targets[q["id"]]["frames"][str(f)] if targets else None
            if key is not None:
                if key["visible_targets"]:
                    visible_frames += 1
                else:
                    absent_frames += 1
                    abstain_on_absent += goal is None
            if goal is None:
                continue
            answered += 1
            goals.append(np.asarray(goal, float))
            if key is not None:
                boxes = [b for b in scene.boxes if b["uid"] in key["visible_targets"]]
                hit = any(in_box(goal, b, GOAL_MARGIN_M) for b in boxes)
                hits.append(hit)
                found_on_visible += hit and bool(key["visible_targets"])
                confidences.append(float(a.get("confidence", 0.0)))
        if len(goals) >= 2:
            spreads.append(
                max(float(np.linalg.norm(a - b)) for a, b in combinations(goals, 2))
            )
    out = {
        "queries": len(queries),
        "answered_frac": answered / total if total else 0.0,
        "median_spread_m": float(np.median(spreads)) if spreads else None,
    }
    if targets is not None:
        out.update(
            {
                "hit_rate": float(np.mean(hits)) if hits else None,
                "goal_recall": found_on_visible / visible_frames
                if visible_frames
                else None,
                "confidence_auroc": auroc(confidences, hits) if hits else None,
                "abstain_rate_on_absent": abstain_on_absent / absent_frames
                if absent_frames
                else None,
            }
        )
    return out


def detection_truth(scene, detections):
    """(frame, det_index) -> GT uid the lifted detection lands in (same label), or None."""
    truth = {}
    for f, dets in detections.items():
        for k, d in enumerate(dets):
            p = lift_detection(scene, int(f), d["box"])
            uid = None
            if p is not None:
                for b in scene.boxes:
                    if b["label"] == d["label"] and in_box(p, b, INSTANCE_MARGIN_M):
                        uid = b["uid"]
                        break
            truth[(int(f), k)] = uid
    return truth


def score_part3(scene, detections, instances):
    """instances: [{id, label, center_world, observations: [[frame, det_index], ...]}]."""
    out = {
        "instances": len(instances),
        "instances_per_class": dict(Counter(i["label"] for i in instances)),
    }
    if scene.boxes is None:
        return out
    matched = defaultdict(list)
    false_instances = 0
    for inst in instances:
        hit = [
            b["uid"]
            for b in scene.boxes
            if b["label"] == inst["label"]
            and in_box(inst["center_world"], b, GOAL_MARGIN_M)
        ]
        if hit:
            matched[hit[0]].append(inst["id"])
        else:
            false_instances += 1
    truth = detection_truth(scene, detections)
    claims = defaultdict(set)
    purity = []
    for inst in instances:
        uids = [truth.get((int(f), int(k))) for f, k in inst.get("observations", [])]
        uids = [u for u in uids if u is not None]
        for u in set(uids):
            claims[u].add(inst["id"])
        if uids:
            purity.append(Counter(uids).most_common(1)[0][1] / len(uids))
    out.update(
        {
            "gt_boxes": len(scene.boxes),
            "recall": len(matched) / len(scene.boxes),
            "fragmentation": float(np.mean([len(v) for v in matched.values()]))
            if matched
            else None,
            "false_instances": false_instances,
            "id_switches": int(sum(max(0, len(v) - 1) for v in claims.values())),
            "observation_purity": float(np.mean(purity)) if purity else None,
        }
    )
    return out


def main():
    answers_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "answers.json"
    instances_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "instances.json"
    if not (ROOT / "data").is_dir():
        raise SystemExit("no data/ yet — run `uv run python fetch.py` first")
    queries = json.loads((ROOT / "variant" / "queries.json").read_text())["scenes"]
    detections = json.loads((ROOT / "variant" / "detections.json").read_text())[
        "scenes"
    ]
    answers = json.loads(answers_path.read_text()) if answers_path.exists() else {}
    instances = (
        json.loads(instances_path.read_text()) if instances_path.exists() else {}
    )
    report = {"part2": {}, "part3": {}}
    for name in queries:
        scene = Scene(ROOT / "data" / name)
        targets_path = ROOT / "variant" / name / "query_targets.json"
        targets = (
            json.loads(targets_path.read_text()) if targets_path.exists() else None
        )
        if name in answers:
            report["part2"][name] = score_part2(
                scene, queries[name], answers[name], targets
            )
        if name in instances:
            report["part3"][name] = score_part3(
                scene, detections[name], instances[name]
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
