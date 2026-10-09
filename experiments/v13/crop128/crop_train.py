"""Fine-tune the v13 U-Net on the bottom 128 rows (rows 112-239) of every frame.
Same pinky_lane training code and config as the original v13 run; only the input rows change.
    python crop_train.py --config config-unet-crop128.json --dataset data-v13 --warm-start runs/unet/best.pt --output runs/unet-crop128"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "unet-src"))
CROP = 112
import pinky_lane.dataset as ds
_read = ds.LaneDataset.read
def read_cropped(self, row):
    bgr, labels, policy = _read(self, row)          # all original checks run on the full frame
    return bgr[CROP:].copy(), labels[CROP:].copy(), policy
ds.LaneDataset.read = read_cropped
import pinky_lane.train as tr
tr.main()
