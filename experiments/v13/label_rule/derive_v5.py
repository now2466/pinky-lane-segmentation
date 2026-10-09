"""v5: drivable = floor connected to the robot (bottom centre) without crossing a white line.
White lines = lane_left/lane_right/crosswalk/bump labels + unlabelled white paint.
Where both lane lines are in a row, carpet outside them is always 0 (no leak around a line end).
Walls rising from the top band are 0. Rows above ignore_top are 255."""
import numpy as np, cv2
from derive_v3 import wall_mask, IGN_TOP
from regions2 import paint_mask

def derive(src, bgr):
    H, W = src.shape
    lane = (src >= 1) & (src <= 4)
    paint = paint_mask(bgr) & ~lane
    n0, l0, s0, _ = cv2.connectedComponentsWithStats(paint.astype(np.uint8), connectivity=8)
    for i in range(1, n0):
        if s0[i, cv2.CC_STAT_AREA] < 80: paint[l0 == i] = False
    wall = wall_mask(bgr, src)
    barrier = (lane | paint | wall).astype(np.uint8)
    barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    barrier = cv2.dilate(barrier, np.ones((3, 3), np.uint8)) > 0
    free = ~barrier; free[:IGN_TOP] = False
    n, lab, st, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=4)
    out = np.full(src.shape, 255, np.uint8)
    floor = ~lane; floor[:IGN_TOP] = False
    seed = lab[H - 15:, W // 4: 3 * W // 4]
    cnt = np.bincount(seed[seed > 0], minlength=n)
    robot = set()
    if cnt.sum():
        robot = set(np.flatnonzero(cnt >= max(20, 0.15 * cnt.max())).tolist())
    road = np.isin(lab, list(robot)) if robot else np.zeros_like(free)
    out[floor] = 0
    out[road] = 5
    # rows with both lines: outside them is never road
    for y in range(IGN_TOP, H):
        L = np.flatnonzero(src[y] == 1); R = np.flatnonzero(src[y] == 2)
        if L.size and R.size and L.max() < R.min():
            out[y, :L.min()][out[y, :L.min()] == 5] = 0
            out[y, R.max() + 1:][out[y, R.max() + 1:] == 5] = 0
    # thin barrier halo between road and line: give it to the road side if it touches road
    halo = barrier & ~lane & ~paint & ~wall & floor
    grown = cv2.dilate((out == 5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    out[halo & grown & (out == 0)] = 5
    out[lane] = src[lane]
    out[paint & ~lane] = 0
    out[:IGN_TOP] = 255
    return out
