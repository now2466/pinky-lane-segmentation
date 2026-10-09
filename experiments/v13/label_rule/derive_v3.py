"""v3: drivable straight from the lane classes, row by row.
lane_left (1) / lane_right (2) are the road boundaries:
  both in a row:  between = drivable (5), left of left / right of right = not (0)
  only left:      right of it = 5, left of it = 0
  only right:     left of it = 5, right of it = 0
Rows below the lowest row with a line (the lines left the frame near the robot) reuse
line edges extrapolated from the lowest FIT rows. Rows above ignore_top stay 255.
Crosswalk / speed bump keep their class. Walls (bright flat area rising from the top band,
in columns that have a line below them) become 0."""
import numpy as np, cv2
IGN_TOP, FIT, MAX_RESID = 110, 30, 4.0

def wall_mask(bgr, src):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (7, 7)); std = np.sqrt(np.maximum(cv2.blur(gray * gray, (7, 7)) - mean * mean, 0))
    cand = (gray >= 160) & (std <= 10) & (src == 0)
    lane = (src >= 1) & (src <= 4)
    has = lane.any(0)
    top = np.where(has, lane.argmax(0), 0)
    above = (np.arange(src.shape[0])[:, None] < top[None, :]) & has[None, :]
    cand &= above
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    big = np.zeros_like(cand)
    touch = set(np.unique(lab[IGN_TOP][cand[IGN_TOP]]).tolist()) - {0}
    for i in touch:
        if st[i, cv2.CC_STAT_AREA] >= 200: big |= lab == i
    return big

def row_label(W, L, R):
    xs = np.arange(W); lab = np.full(W, 255, np.uint8)
    if L is not None and R is not None:
        lmin, lmax = L; rmin, rmax = R
        if lmax < rmin: lab[(xs > lmax) & (xs < rmin)] = 5
        lab[xs < lmin] = 0; lab[xs > rmax] = 0
    elif L is not None:
        lab[xs > L[1]] = 5; lab[xs < L[0]] = 0
    elif R is not None:
        lab[xs < R[0]] = 5; lab[xs > R[1]] = 0
    return lab

def _fit(rows, vals):
    if len(rows) < 8: return None
    p = np.polyfit(rows, vals, 1)
    if np.sqrt(np.mean((np.polyval(p, rows) - vals) ** 2)) > MAX_RESID: return None
    return p

def derive(src, bgr):
    H, W = src.shape
    out = np.full(src.shape, 255, np.uint8)
    edges = {}
    for y in range(IGN_TOP, H):
        r = src[y]; L = np.flatnonzero(r == 1); R = np.flatnonzero(r == 2)
        edges[y] = ((L.min(), L.max()) if L.size else None, (R.min(), R.max()) if R.size else None)
    rows_with = [y for y in edges if edges[y][0] is not None or edges[y][1] is not None]
    if rows_with:
        low = max(rows_with)
        fits = {}
        for side in (0, 1):
            rs = [y for y in range(low - FIT, low + 1) if y in edges and edges[y][side] is not None]
            fits[side] = None
            if rs and max(rs) >= low - 3:      # this line reaches (nearly) the lowest line row
                a = _fit(np.array(rs), np.array([edges[y][side][0] for y in rs], float))
                b = _fit(np.array(rs), np.array([edges[y][side][1] for y in rs], float))
                if a is not None and b is not None: fits[side] = (a, b)
        last = {0: None, 1: None}
        for side in (0, 1):
            for y in range(low, IGN_TOP - 1, -1):
                if edges[y][side] is not None: last[side] = (y, edges[y][side]); break
        for y in range(low + 1, H):
            ext = []
            for side in (0, 1):            # 0 = left line, 1 = right line
                f = fits[side]
                if f is not None:
                    lo, hi = np.polyval(f[0], y), np.polyval(f[1], y)
                elif last[side] is not None and low - last[side][0] <= 3:
                    lo, hi = last[side][1]                         # no reliable fit: hold the last edge
                else:
                    ext.append(None); continue
                if side == 0 and hi < 0:   ext.append((-1, -1))  # left line left through the left side
                elif side == 1 and lo >= W: ext.append((W, W))    # right line left through the right side
                else: ext.append((int(np.clip(round(lo), 0, W - 1)), int(np.clip(round(hi), 0, W - 1))))
            # a line that ended near a frame side, with no fit: it left the frame there
            for side in (0, 1):
                if ext[side] is None and last[side] is not None and low - last[side][0] <= 3:
                    lo, hi = last[side][1]
                    if side == 0 and lo <= 2: ext[side] = (-1, -1)
                    if side == 1 and hi >= W - 3: ext[side] = (W, W)
            edges[y] = (ext[0], ext[1])
    for y in range(IGN_TOP, H):
        L, R = edges[y]
        if L is None and R is None:
            continue
        lab = row_label(W, L, R)
        free = src[y] == 0
        out[y, free] = lab[free]
    lane = (src >= 1) & (src <= 4)
    out[lane] = src[lane]
    out[wall_mask(bgr, src)] = 0
    out[:IGN_TOP] = 255
    return out
