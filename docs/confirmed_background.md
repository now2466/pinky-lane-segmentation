# Confirmed-background supervision experiment

Legacy four-label masks keep `legacy_partial_class4`: their background CE is
`-log(p_background + p_speed_bump)` because unannotated bumps may exist.
This can leave false speed-bump predictions unpenalized on known flat objects.
It does not prove why the network specifically recognizes red tape.

For visually verified **training** images only, optional manifest field
`confirmed_background_polygons` contains integer `(x,y)` polygon vertices in
320x240 coordinates. Both `image_sha256` and `mask_sha256` must match the files.
The loader marks only original background pixels within these polygons as an
internal sentinel 254. Semantic PNGs remain unchanged (0–4, 255); neither the
model output nor exported model gains a sixth class. Foreground/255 never change.

Confirmed pixels receive strict background CE and count as negative pixels in
speed-bump Dice. Other legacy background retains marginal CE and remains excluded
from speed-bump Dice. This feature is forbidden for val/test and four-class runs.

2026-10-05 controlled experiment: 48 visually inspected legacy training frames
containing the flat red tape around a blue floor marker. Red-component selection
is limited to those approved frames and the lower image; no automatic extension.
Train/val/test counts stay 4900/823/401 and all original images/masks are preserved.
Two runs start at the same v11 best checkpoint: unmodified-data control versus
confirmed-background manifest. Both use 20 epochs, seed 42, batch 16, original
AdamW/config/lighting augmentation, and validation-only checkpoint selection.
Test is evaluated unchanged; video comparisons use raw predictions without
postprocessing. Color-defined tape diagnostics are not ground-truth IoU and the
video has already been used for qualitative feedback, so is not a blind holdout.

Do not promote the experimental model until tape false positives decrease without
an unacceptable loss of actual bump or lane performance. Training, evaluation,
and comparison artifacts are kept outside the public repository.
