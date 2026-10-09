"""v2 floor split: white paint = locally bright + low saturation (lit carpet is not paint)."""
import cv2, numpy as np
IGN_TOP = 110

def paint_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    local = cv2.medianBlur(gray.astype(np.uint8), 41).astype(np.int16)
    m = (gray - local > 30) & (hsv[..., 1] < 70) & (gray > 120)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 60:
            keep[lab == i] = 1
    return keep.astype(bool)

def regions(bgr, lane_mask, min_area=150):
    lanes = (lane_mask >= 1) & (lane_mask <= 4)
    barrier = (lanes | paint_mask(bgr)).astype(np.uint8)
    barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    barrier = cv2.dilate(barrier, np.ones((3, 3), np.uint8))
    free = barrier == 0
    free[:IGN_TOP] = False
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=4)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    out = np.zeros(lane_mask.shape, np.int32)
    for k, i in enumerate(sorted(keep, key=lambda i: -stats[i, cv2.CC_STAT_AREA]), 1):
        out[lab == i] = k
    # absorb small leftover holes (carpet speckle) into the surrounding region
    hole = (out == 0) & ~lanes
    hole[:IGN_TOP] = False
    if out.max():
        for _ in range(6):
            grown = cv2.dilate(out.astype(np.uint16), np.ones((3, 3), np.uint8)).astype(np.int32)
            fill = hole & (out == 0) & (grown > 0)
            out[fill] = grown[fill]
    return out
