---
language:
- en
pretty_name: CUHK-X — Small Model Track
tags:
- multimodal
- human-activity
- action-recognition
- depth
- infrared
- thermal
- imu
- radar
- skeleton
task_categories:
- video-classification
---
# CUHK-X — Small Model Track

Multimodal **human action recognition (classification)**.
Given a multimodal clip, predict its action class (`action_id`, **0–39, 40 classes**).

## Repository layout

```
.
├── Training/
│   ├── class_mapping.csv     # action_id <-> action_name (40 classes)
│   └── data/
│       └── HAR.z01 … HAR.z08 + HAR.zip   # multi-volume zip
│           →  HAR/data/<modality>/<action>/<user>/<trial>/<files>
└── Testing/
    ├── data/
    │   └── small_model_track_test.zip    →  small_model_track_test/<id>/<modality>/<files>
    └── test_file/
        ├── test.csv              # path + empty `prediction` (to fill)
        └── sample_submission.csv # submission example
```

## Labels

- **Training labels live in the path**: in `HAR/data/<modality>/<action>/<user>/<trial>`,
  the `<action>` (e.g. `0_Wash_face`) is the class.
- Convert between `action_name` and `action_id` with `class_mapping.csv`.
- Test clips are anonymized (`SM_test_XXXX`); predict their `action_id`.

## Modalities (6; no RGB, no raw Depth)

| Modality        | Type                     | Example file                         |
| --------------- | ------------------------ | ------------------------------------ |
| `Depth_Color` | colorized depth (frames) | `Depth_<datetime>_<idx>_Color.png` |
| `IR`          | infrared (frames)        | `IR_<datetime>_<idx>.png`          |
| `Thermal`     | thermal (frames)         | `frame_000063.jpg`                 |
| `IMU`         | inertial sensor          | `*.csv`                            |
| `Radar`       | mmWave radar             | `radar_output_T<ts>.csv`           |
| `Skeleton`    | skeleton                 | pose data +`visualizations/`       |

Sampling rates differ across modalities; not every clip has every modality.

## Extracting the data

Training is a multi-volume zip (`HAR.z01`…`HAR.z08` + `HAR.zip`; keep all volumes in one folder):

```bash
cd Training/data
zip -s 0 HAR.zip --out HAR_full.zip   # merge volumes (zip 3.0+)
unzip HAR_full.zip                    # -> HAR/data/<modality>/<action>/<user>/<trial>/...
```

7-Zip / WinRAR / double-click also handle split zips. Test set:

```bash
cd Testing/data && unzip small_model_track_test.zip   # -> small_model_track_test/<id>/<modality>/...
```

## Submission

In `Testing/test_file/test.csv`, fill each row's `prediction` with the predicted `action_id` (0–39).
See `sample_submission.csv` for the format.

## Statistics

- 40 action classes
- 405 test clips

## Quick start

```python
import csv
id2name = {r["action_id"]: r["action_name"]
           for r in csv.DictReader(open("Training/class_mapping.csv", encoding="utf-8-sig"))}
# Training: the <action> folder in the path is the label, e.g.
#   HAR/data/IR/0_Wash_face/user10/4-2-1/  ->  action_name="0_Wash_face", action_id="0"
```