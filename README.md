# Ndimensions Labs — Robotics Perception Take-Home

## Time budget

The assignment should take **16 hours maximum**. We mean this. We'd rather see clear reasoning about what you chose not to do than a polished submission that took you days. Note roughly how long you spent on each part.

**Return within 1 week of receiving this.** You'll then walk us through your work in a 45-minute session. Submit early if you're done early; do not spend more time than necessary.

**Use of AI tools is fine and expected.** Please say which models you used. The parts of the assignment that matter can't be answered without looking at the data.


## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.10. Everything runs on CPU; no GPU is needed anywhere.

```bash
# Fork this repo to your own GitHub account first (Fork button, top right) — you have read-only
# access here, so your work lives in your fork.
git clone https://github.com/<your-username>/perception_takehome.git && cd perception_takehome
uv sync
uv run python fetch.py          # ~0.5 GB from Apple's public CDN, no account needed
uv run python score.py          # prints an empty report — setup is done
```

`fetch.py` downloads four scenes of Apple's [ARKitScenes](https://github.com/apple/ARKitScenes) (iPad RGB + LiDAR depth, real homes) and lays them out under `data/`. Those archives also contain Apple's own camera calibration and trajectory. **Don't use them.** The exercise is about deciding what to trust from what you were handed — `fetch.py` copies only the image frames, and we'll ask how you found each issue. A finding without a method scores nothing.

## Usage

```python
from perception.scene import Scene

s = Scene("data/scene_a")
s.color, s.depth_cam  # intrinsics; depth_cam also carries scale_m and T_color_depth
s.pose(i)  # 4x4 camera-to-world for frame i
s.rgb(i), s.depth_m(i)  # HxWx3 uint8, HxW float32 metres (0 = no return)
s.boxes  # dev scenes only: surveyed object boxes in the world frame
```

`perception/geometry.py` has unproject / project / transform. `score.py` scores your `answers.json` (Part 2) and `instances.json` (Part 3) on the dev scenes; format below. We run the same scorer on the test scenes when we grade.

## What you're given

- `data/<scene>/` — four scenes at 10 Hz, 256×192 RGB and 16-bit depth, roughly 500–1300 frames each, plus `calib.json`. Two are **dev** scenes (`variant/scenes.json` says which): they also carry `boxes.json`, oriented world-frame boxes of the furniture from a survey of that room, and `variant/<scene>/query_targets.json`, the labelled targets for that scene's Part 2 queries. The two **test** scenes carry neither; we hold those back.
- `calib.json` — the rig calibration **as handed over by the integrator**: colour intrinsics, depth intrinsics with `scale_m` (raw units → metres) and `T_color_depth` (depth sensor → colour camera), and one timestamp + camera-to-world pose per frame. The world frame is the robot's map frame, y up.
- `variant/<scene>/sync.json` — the recorder's log of which depth frame it paired with each colour frame, as camera timestamps. `fetch.py` extracts the frames in this order, so `data/<scene>/rgb/000042.png` and `data/<scene>/depth/000042.png` are row 42 of this file. Every recording in the fleet has some pairing noise in it.
- `variant/queries.json` — Part 2 queries: text, and the frames each query is asked from.
- `variant/detections.json` — open-vocabulary 2D detections ([OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble), 17-class furniture vocabulary, threshold 0.1) for every frame of every scene: `{label, score, box: [x0, y0, x1, y1]}` in colour pixels. Rerunning the detector is not required and doesn't count against your time; the notebook-free path is to use these as they are.
- `robot_eval_results.json` — on-robot results referenced by Part 4b.

The story: a mobile robot with a single RGB-D head camera drove through four homes. Depth and colour were paired by the recorder, one timestamp per frame. The calibration came from the integrator; the trajectory came from the robot's localisation. The robot is asked for things by name and must resolve a named object to a place it can drive to.

## Part 1 — Is this rig trustworthy?

Some of what's in `calib.json`, and some of how frames were paired, is wrong. Some of it can be detected and fixed from the data alone, some can be detected but not fixed, and some cannot be verified at all with what you were handed.

Every one of these four recordings has *something* off — no two units in the fleet are calibrated alike, and every recorder drops the occasional frame. What separates them is size. At least one scene carries nothing worse than the residual you would accept and ship. Deciding which errors are worth acting on is the exercise; a list of everything that is non-zero is not an answer.

Investigate all four scenes and write up:

1. **What's wrong, and what merely differs.** Per scene, per issue: what, how big, which frames, how you found it, how confident you are, and whether it is worth acting on. Quantify; don't assert. Where a file tells you something is off, say what you measured to decide it mattered.
2. **What you did about it.** Fixed (with the corrected value and how you got it), worked around, or flagged. If you fixed something, show the before/after on a measurement that would have caught it.
3. **What you cannot verify** with this data, and what one additional measurement or fixture would let you.
4. **One check** you'd run on every recording at capture time to catch these before they reach a map. Pseudocode is fine.

Deliverable: [calibration_report.md](deliverables/calibration_report.md), plus whatever scripts and plots you used. Use the corrected calibration for Parts 2 and 3 where you have one.

## Part 2 — Query to a 3D goal, and do you believe it?

For each query in `variant/queries.json`, from each listed frame, return a **goal in the world frame** — where the named object is, as a 3D point — and a **confidence in [0, 1] that the goal matches the instruction**. Return `null` for the goal when you think the instruction cannot be satisfied from that frame.

Queries range from unambiguous ("the refrigerator") to qualified ("the chair nearest the sink") to genuinely ambiguous, and some name an object that is not visible in that frame. The shipped detections are a real detector's output: sometimes wrong, sometimes missing. Your confidence is scored on whether it separates goals that match the instruction from ones that don't; your goals are scored on whether they land on the right object and on whether the same query asked from different viewpoints lands in the same place.

Returning `null` is a real answer, not a free pass: `hit_rate` is measured over the frames you answered, and `goal_recall` over every frame where the object was actually there. Declining the hard frames moves the second number, so tell us where you drew the line and why.

`answers.json`:

```json
{"scene_a": {"a01": {"165": {"goal_world": [x, y, z], "confidence": 0.8},
                     "450": {"goal_world": null, "confidence": 0.1}}}}
```

Deliverable: your code, `answers.json`, and [grounding_notes.md](deliverables/grounding_notes.md): how you lift a detection to 3D, how you pick among candidates, what your confidence actually measures, and what you'd change with a labelling budget of 200 frames.

## Part 3 — Make it persist

The detections are per frame. Turn them into **one instance per physical object per scene**, with a stable id, a class and a world-frame position, and record which detections each instance explains. Rooms contain several objects of the same class, and most objects are seen from opposite sides at different times, so geometry alone will not carry identity.

`instances.json`:

```json
{"scene_a": [{"id": 1, "label": "chair", "center_world": [x, y, z],
              "observations": [[frame, detection_index], ...]}]}
```

`score.py` reports, on dev scenes: recall of surveyed boxes, fragmentation (instances per box), false instances, id switches and observation purity. We don't publish a target number — tell us what you got, why, and what the remaining failure modes are.

If you run short on time, a design with pseudocode and an honest account of what it would get wrong is acceptable for this part. Skipping it is not.

Deliverable: your code, `instances.json`, and [mapping_notes.md](deliverables/mapping_notes.md).

## Part 4 — Deploy and measure

Written. No code required.

**4a.** The detector behind `detections.json` runs at ~2 s/frame on a laptop CPU. On the robot it has to run on an embedded GPU (Jetson-class) inside a 150 ms budget while the robot moves at 0.5 m/s. What do you run, at what rate, and what do you give up? Then: in a new home it underperforms on the objects that matter there, and you can afford 300 labelled frames. What do you label, how do you choose them, and how do you know it worked before the robot goes back?

**4b.** `robot_eval_results.json` is the result of the same grounding stack on two of our robots across two homes. Give your three most likely explanations for what you see, ranked, with reasoning. For each, name the single measurement or experiment that confirms or eliminates it. Then say what you'd instrument next, in order.

**4c.** None of this has ground truth in a customer's home. How would you know the map is right and the goals are good after a week of operation? And if we add a second camera on the arm's wrist, where does its calibration enter this pipeline, and what breaks first if it's wrong?

Deliverable: [deployment.md](deliverables/deployment.md).

### Feedback (optional)

1. What would you have asked us before starting, if you could have?
2. What is the biggest thing this exercise fails to test about your ability to do this job?

---

## Final deliverables

Put everything in the Google Drive folder we shared with you, then reply to that email to let us know you're done.

1. **Code** — a link to your repo (or a zip), with whatever you changed or added.
2. **Writeups** — the four markdown files under `deliverables/` (or one `report.md` if you prefer), your feedback answers, and any plots or scripts your reasoning leans on.
3. **Results** — `answers.json`, `instances.json`, and the output of `uv run python score.py`.

## How we evaluate this

We care most about judgment: whether you measured before you fixed, whether your conclusions follow from evidence, whether you can tell a defect worth fixing from one worth living with, and whether you know what you don't know. A submission that finds the important issues, states what could not be verified, and honestly reports a Part 3 that half-works beats a tidy one that trusted the calibration.

We do not care about code polish, test coverage, or documentation beyond what's needed to follow your reasoning.
