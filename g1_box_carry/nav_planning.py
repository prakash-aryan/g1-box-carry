"""2-D path planning on an occupancy grid (numpy/scipy).

Occupancy from axis-aligned boxes in a height band, a clearance map, A* with a cost term that penalises
proximity to obstacles, and path smoothing. The module has no simulator dependencies; boxes are given in
metres, Z-up.
"""
import heapq, math

import numpy as np
from scipy.ndimage import distance_transform_edt


class Grid:
    def __init__(self, boxes, window, floor, res=0.05, z_band=(0.05, 0.9), max_area_frac=0.4):
        """Rasterise axis-aligned boxes into an occupancy grid.

        boxes: (lo, hi) arrays of shape (N, 3), e.g. per-face boxes from scene_utils.face_boxes.
        window: (x0, y0, x1, y1) in metres.

        Boxes overlapping floor + z_band are marked occupied. Boxes covering more than max_area_frac of the
        window (merged floor, ceiling or wall shells) are ignored.
        """
        self.x0, self.y0, x1, y1 = window
        self.res = res
        self.nx = max(int((x1 - self.x0) / res), 1)
        self.ny = max(int((y1 - self.y0) / res), 1)
        area = max((x1 - self.x0) * (y1 - self.y0), 1e-6)
        lo, hi = boxes
        keep = ((hi[:, 2] >= floor + z_band[0]) & (lo[:, 2] <= floor + z_band[1])
                & ((hi[:, 0] - lo[:, 0]) * (hi[:, 1] - lo[:, 1]) <= max_area_frac * area))
        lo, hi = lo[keep], hi[keep]
        i0 = np.floor((lo[:, 0] - self.x0) / res).astype(np.int64)
        j0 = np.floor((lo[:, 1] - self.y0) / res).astype(np.int64)
        i1 = np.maximum(np.ceil((hi[:, 0] - self.x0) / res).astype(np.int64), i0 + 1)
        j1 = np.maximum(np.ceil((hi[:, 1] - self.y0) / res).astype(np.int64), j0 + 1)
        i0, j0 = i0.clip(0, self.nx), j0.clip(0, self.ny)
        i1, j1 = i1.clip(0, self.nx), j1.clip(0, self.ny)
        w, h = i1 - i0, j1 - j0
        occ = np.zeros((self.nx, self.ny), dtype=bool)
        K = 4  # boxes of up to KxK cells are rasterised vectorised, larger ones in a loop
        small = (w <= K) & (h <= K) & (w > 0) & (h > 0)
        for di in range(K):
            for dj in range(K):
                sel = small & (di < w) & (dj < h)
                occ[i0[sel] + di, j0[sel] + dj] = True
        for a, b, c, d in zip(i0[~small], i1[~small], j0[~small], j1[~small]):
            occ[a:b, c:d] = True
        occ[[0, -1], :] = True
        occ[:, [0, -1]] = True
        self.occ = occ
        self.update_clearance()

    def update_clearance(self):
        self.clearance = distance_transform_edt(~self.occ) * self.res  # metres to nearest obstacle

    def add_rectangle(self, centre, yaw_deg, half_len, half_wid):
        """Mark an oriented rectangle as occupied (e.g. objects added to the scene)."""
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        ii, jj = np.nonzero(np.ones_like(self.occ))
        X = self.x0 + (ii + 0.5) * self.res - centre[0]
        Y = self.y0 + (jj + 0.5) * self.res - centre[1]
        inside = (np.abs(X * c + Y * s) <= half_len) & (np.abs(-X * s + Y * c) <= half_wid)
        self.occ[ii[inside], jj[inside]] = True
        self.update_clearance()

    def idx(self, x, y):
        return int(math.floor((x - self.x0) / self.res)), int(math.floor((y - self.y0) / self.res))

    def xy(self, i, j):
        return self.x0 + (i + 0.5) * self.res, self.y0 + (j + 0.5) * self.res

    def inside(self, i, j):
        return 0 <= i < self.nx and 0 <= j < self.ny

    def nearest_free(self, x, y, min_clear):
        """Grid cell with clearance >= min_clear nearest to (x, y), or None."""
        ok = self.clearance >= min_clear
        if not ok.any():
            return None
        ii, jj = np.nonzero(ok)
        cx, cy = self.x0 + (ii + 0.5) * self.res, self.y0 + (jj + 0.5) * self.res
        k = int(np.argmin(np.hypot(cx - x, cy - y)))
        return int(ii[k]), int(jj[k])

    def astar(self, start, goal, min_clear=0.45, prefer_clear=1.0):
        """8-connected A* between cells, only through cells with clearance >= min_clear. Extra cost near
        obstacles keeps paths centred in free space. Returns a list of (x, y) in metres, or None if
        unreachable."""
        free = self.clearance >= min_clear
        if not (free[start] and free[goal]):
            return None
        penalty = prefer_clear / np.maximum(self.clearance, 0.05)
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        g = {start: 0.0}
        came = {}
        h = lambda c: math.hypot(c[0] - goal[0], c[1] - goal[1])
        pq = [(h(start), start)]
        while pq:
            _, cur = heapq.heappop(pq)
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return [self.xy(*c) for c in reversed(path)]
            gc = g[cur]
            for di, dj, w in steps:
                n = (cur[0] + di, cur[1] + dj)
                if not self.inside(*n) or not free[n]:
                    continue
                cost = gc + w * (1.0 + penalty[n])
                if cost < g.get(n, math.inf):
                    g[n] = cost
                    came[n] = cur
                    heapq.heappush(pq, (cost + h(n), n))
        return None


def smooth(path, spacing=0.1, window=9):
    """Resample a polyline to even spacing and moving-average it (endpoints kept)."""
    p = np.asarray(path, dtype=float)
    if len(p) < 3:
        return p
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(s[-1] / spacing), 2)
    t = np.linspace(0.0, s[-1], n)
    q = np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])
    if len(q) > window:
        k = np.ones(window) / window
        pad = window // 2
        for d in range(2):
            ext = np.concatenate([np.full(pad, q[0, d]), q[:, d], np.full(pad, q[-1, d])])
            q[:, d] = np.convolve(ext, k, mode="valid")
    return q
