# UTD-MHAD Dataset Description

## Overview

The **UTD-MHAD** dataset contains synchronized multimodal recordings of 27 human actions performed by 8 subjects, with up to 4 trials per subject per action. Files follow the naming convention `a*_s*_t*`, where:

- **`a`** — action index (1–27)
- **`s`** — subject index (1–8)
- **`t`** — trial index (1–4), i.e. the repetition number for that subject/action pair

## Modalities

| Modality | Format | Shape | Description |
|----------|--------|-------|-------------|
| **RGB video** | `a*_s*_t*_color.avi` | H × W × 3 × T | Color video |
| **Inertial** | `a*_s*_t*_inertial.mat` | T × 6 | 3-axis accelerometer (g) + 3-axis gyroscope (°/s) |
| **Depth** | `a*_s*_t*_depth.mat` | 240 × 320 × T | Per-pixel distance from sensor |
| **Skeleton** | `a*_s*_t*_skeleton.mat` | 20 × 3 × T | 3D XYZ coordinates of 20 body joints |

**This assignment uses only RGB + Inertial.** Depth and Skeleton are available in the dataset but excluded per the assignment instructions.

## Sensor Placement

The inertial sensor placement differs by action group — a critical structural property of the dataset:

- Actions **1–21**: sensor on the **right wrist** — captures arm and hand gestures
- Actions **22–27**: sensor on the **right thigh** — captures locomotion and lower-body movements

This means the inertial signal distribution (magnitude, frequency content, axis orientation) shifts fundamentally between the two groups.

## Action Classes

Full list available in `Sample_Code/Action_List.txt`.

## Working Subset (8 Classes)

All models in this assignment are trained and evaluated on the following 8-class subset, chosen to maximise diversity across motion type, sensor placement, and inter-class similarity:

| Action | Name | Sensor | Notes |
|--------|------|--------|-------|
| 1 | swipe left | wrist | Confusable pair with action 2 |
| 2 | swipe right | wrist | Confusable pair with action 1 |
| 4 | clap | wrist | Symmetric, distinctive signal |
| 13 | boxing | wrist | High-energy, sharp accelerations |
| 19 | knock | wrist | Repetitive short bursts |
| 22 | jog | thigh | Periodic locomotion |
| 23 | walk | thigh | Locomotion, lower energy than jog |
| 27 | squat | thigh | Lower-body, non-periodic |

Actions 1 and 2 are intentionally included as a confusable pair to stress-test model discriminability.

---

## Potential Data Issues & Deployment Implications

| # | Modality | Issue | Deployment Impact |
|---|----------|-------|-------------------|
| 1 | Inertial | **Sensor placement heterogeneity** — wrist (actions 1–21) vs. thigh (actions 22–27) produce fundamentally different signal distributions | Model may learn placement as a shortcut; placement must be fixed or auto-detected at deployment |
| 2 | Inertial | **Continuous signal, unknown boundaries** — dataset clips are pre-segmented; real deployment is a continuous stream | Requires an upstream onset detector or sliding window before classification can occur |
| 3 | RGB | **Static, controlled background** — single lab environment with fixed lighting | Model learns background cues that will not generalise to real scenes |
| 4 | RGB | **Fixed subject position and scale** — subject always centred at fixed distance from camera | Needs a person detector and tracker as a preprocessing step in production |
| 5 | Both | **Small dataset, few subjects** — 8 subjects, ~32 samples/class | Model may overfit to subject identity; new subjects at deployment are effectively out-of-distribution |
| 6 | Both | **Missing modality** — sensor or camera can fail due to hardware fault, occlusion, or power loss | System must degrade gracefully on a single modality (addressed in Part 4) |
