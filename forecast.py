"""
Layer K — the 20 s crowd forecast behind the 3D twin, and the motion state it
starts from.

  MotionFilter    constant-velocity Kalman filter per person on the ground
                  (metres). Pixel noise is mapped through the perspective, so a
                  far-away person whose box jitters by a pixel reads as standing
                  still instead of walking at 0.5 m/s.
  SceneMemory     what the twin learns while it runs: a flow field of where
                  people walk (and which way), and where and how often people
                  enter the view.
  CrowdForecaster the forecast itself — an anticipatory social force model:
                    • walkers keep their tracked speed and heading, and turn
                      towards the scene's learned flow over HEADING_MEMORY_S;
                    • standing people stay put (their velocity is within noise);
                    • collision avoidance follows the time-to-collision power law
                      (Karamouzas, Skinner & Guy, PRL 2014) — people only react
                      to others they are on course to meet, so a dense standing
                      crowd does not blow apart;
                    • walking speed falls with local density (Weidmann's
                      fundamental diagram, 1993);
                    • people leave through the edges of the camera's view and
                      slide along walls; newcomers arrive where and as often as
                      people have been seen entering.

Nothing here needs PyTorch or the rest of the pipeline; units are metres and
seconds throughout.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from scipy.spatial import cKDTree


@dataclass
class ForecastParams:
    steps:              int   = 25       # forecast steps …
    step_s:             float = 0.8      # … of this many seconds → 20 s
    substeps:           int   = 4        # integration substeps per step

    # Who is walking: de-biased speed (see debiased_speed) above WALK_MIN_SPEED
    walk_min_speed:     float = 0.3      # m/s
    walk_ramp:          float = 0.25     # m/s over which "standing" blends into "walking"
    # … and the speed is this many std devs above zero. The de-biased speed
    # above already removes the noise bias, so this only has to reject the
    # noise tail of a still crowd. Set high (2.5..3.5) it also gated genuine
    # walkers measured noisily — distant people in MOT17-09, where the forecast
    # then lost 2 m at 8 s to constant velocity (8.4 → 6.7, ≈ CV's 6.4). The
    # slightly higher speed floor keeps a dense still crowd from drifting.
    walk_z_lo:          float = 1.5
    walk_z_hi:          float = 2.5
    max_speed:          float = 2.0      # m/s
    standing_dodge:     float = 0.1      # share of the avoidance force a standing person acts on

    # Where walkers head: their own heading, turning towards the learned flow
    heading_memory_s:   float = 6.0

    # Anticipatory collision avoidance (Karamouzas et al. 2014)
    tau:                float = 0.5      # relaxation time towards the desired velocity (s)
    radius:             float = 0.25     # body radius (m)
    ttc_k:              float = 1.5      # interaction strength (m²/s²)
    ttc_tau0:           float = 3.0      # interaction horizon (s)
    ttc_cutoff:         float = 4.0      # ignore people farther than this (m)
    max_force:          float = 5.0      # m/s²
    ttc_sidestep:       float = 1.0      # sideways share of the avoidance force
    ttc_keep_right:     float = 0.3      # preference for passing on the right

    # Fundamental diagram (Weidmann 1993): speed ∝ 1 − exp(−γ (1/ρ − 1/ρmax))
    rho_max:            float = 5.4      # persons/m² at which walking stops
    weidmann_gamma:     float = 1.913
    density_radius:     float = 1.2      # m, neighbourhood for local density

    # Arrivals
    weidmann_max_gain:  float = 1.0      # density only slows people (a thinning crowd does not
                                         # speed them past their measured speed: tuned, MOT17 + local)
    flow_max:           float = 0.5      # most the learned lanes may turn a walker's heading (0–1)
    max_arrivals:       int   = 500


# ─── Motion state: Kalman filter on the ground ────────────────────────────────

class MotionFilter:
    """
    One constant-velocity Kalman filter per track id; state (x, y, vx, vy)
    in metres and m/s. Process noise is white acceleration with spectral
    density Q_ACC; a new track starts at rest with velocity std V0_STD, so a
    track needs consistent movement — beyond its measurement noise — before
    it reads as walking.
    """
    Q_ACC  = 0.1          # m²/s³: pedestrians change velocity smoothly (tuned on real tracks)
    V0_STD = 0.8          # m/s: prior on a new track's velocity

    def __init__(self, q_acc: float = Q_ACC, v0_std: float = V0_STD):
        self.q, self.v0_std = q_acc, v0_std
        self.x: Dict[int, np.ndarray] = {}
        self.P: Dict[int, np.ndarray] = {}
        self.t: Dict[int, float]      = {}

    def __len__(self):
        return len(self.x)

    def step(self, t: float, ids: Sequence[int], z: np.ndarray, R: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fold in one frame: measured positions z (N, 2) of tracks `ids` at time
        t, with measurement covariances R (N, 2, 2). Returns filtered
        positions (N, 2), velocities (N, 2) and velocity std (N,) (m/s).
        """
        n = len(ids)
        if n == 0:
            e = np.zeros((0, 2))
            return e, e, np.zeros(0)
        z = np.asarray(z, float).reshape(n, 2)
        R = np.asarray(R, float).reshape(n, 2, 2)
        X, P, dt = np.zeros((n, 4)), np.zeros((n, 4, 4)), np.zeros(n)
        fresh = np.zeros(n, bool)
        for i, k in enumerate(ids):
            if k in self.x:
                X[i], P[i], dt[i] = self.x[k], self.P[k], max(0.0, t - self.t[k])
            else:
                fresh[i] = True
                X[i, :2] = z[i]
                P[i, :2, :2] = R[i]
                P[i, 2, 2] = P[i, 3, 3] = self.v0_std ** 2
        # Predict (only tracks seen before)
        old = ~fresh
        if old.any():
            d = dt[old]
            F = np.tile(np.eye(4), (old.sum(), 1, 1))
            F[:, 0, 2] = F[:, 1, 3] = d
            q = self.q
            Q = np.zeros((old.sum(), 4, 4))
            Q[:, 0, 0] = Q[:, 1, 1] = q * d ** 3 / 3
            Q[:, 0, 2] = Q[:, 2, 0] = Q[:, 1, 3] = Q[:, 3, 1] = q * d ** 2 / 2
            Q[:, 2, 2] = Q[:, 3, 3] = q * d
            X[old] = np.einsum("nij,nj->ni", F, X[old])
            P[old] = F @ P[old] @ F.transpose(0, 2, 1) + Q
            # Update with the measurement
            S = P[old][:, :2, :2] + R[old]
            K = P[old][:, :, :2] @ np.linalg.inv(S)                     # (m, 4, 2)
            X[old] = X[old] + np.einsum("nij,nj->ni", K, z[old] - X[old][:, :2])
            P[old] = P[old] - K @ P[old][:, :2, :]
        for i, k in enumerate(ids):
            self.x[k], self.P[k], self.t[k] = X[i], P[i], t
        vel_std = np.sqrt(np.maximum(0.5 * (P[:, 2, 2] + P[:, 3, 3]), 0.0))
        return X[:, :2].copy(), X[:, 2:].copy(), vel_std

    def drop(self, ids):
        for k in ids:
            self.x.pop(k, None)
            self.P.pop(k, None)
            self.t.pop(k, None)

    def reset(self):
        self.x.clear()
        self.P.clear()
        self.t.clear()


def debiased_speed(speed, vel_std) -> np.ndarray:
    """
    Speed with the noise bias removed. A noisy velocity estimate is longer
    than the true one on average — E|v̂|² = |v|² + 2σ² — so a standing person
    measured with σ = 0.3 m/s "moves" at ~0.4 m/s. √(|v̂|² − 2σ²) is ~0 for
    them and ≈ |v| for anyone clearly walking.
    """
    speed, s = np.asarray(speed, float), np.asarray(vel_std, float)
    return np.sqrt(np.maximum(speed ** 2 - 2 * s ** 2, 0.0))


def walking_weight(speed, vel_std, p: ForecastParams = ForecastParams()) -> np.ndarray:
    """
    0 = standing, 1 = walking. Two tests, both ramps: the de-biased speed is
    above WALK_MIN_SPEED, and the speed is significant — WALK_Z_LO..HI
    standard deviations — so the noise tail of a standing crowd (a few in a
    hundred read 3σ fast) does not set off walking.
    """
    speed, s = np.asarray(speed, float), np.asarray(vel_std, float)
    fast = np.clip((debiased_speed(speed, s) - p.walk_min_speed) / p.walk_ramp, 0.0, 1.0)
    z = np.divide(speed, s, out=np.full_like(speed, np.inf), where=s > 1e-9)
    sure = np.clip((z - p.walk_z_lo) / (p.walk_z_hi - p.walk_z_lo), 0.0, 1.0)
    return fast * sure


# ─── Floor geometry ───────────────────────────────────────────────────────────

class Floor:
    """
    The walkable floor as a convex polygon whose edges are "wall" (the floor
    ends: people slide along it), or "view" / "range" (the camera's view ends:
    people walk out of the twin there, and in from there).
    """
    def __init__(self, edges: Sequence[dict]):
        self.a     = np.array([e["a"] for e in edges], float).reshape(-1, 2)
        self.b     = np.array([e["b"] for e in edges], float).reshape(-1, 2)
        self.kinds = [e.get("kind", "view") for e in edges]
        self.wall  = np.array([k == "wall" for k in self.kinds], bool)
        pts = np.vstack([self.a, self.b]) if len(self.a) else np.zeros((0, 2))
        centre = pts.mean(axis=0) if len(pts) else np.zeros(2)
        n = np.stack([self.b[:, 1] - self.a[:, 1], self.a[:, 0] - self.b[:, 0]], axis=1) \
            if len(self.a) else np.zeros((0, 2))
        norm = np.linalg.norm(n, axis=1)
        keep = norm > 1e-9
        self.a, self.b, self.wall = self.a[keep], self.b[keep], self.wall[keep]
        self.kinds = [k for k, ok in zip(self.kinds, keep) if ok]
        n = n[keep] / norm[keep, None]
        flip = np.einsum("ij,ij->i", centre - self.a, n) < 0
        n[flip] *= -1
        self.n   = n                                    # inward unit normals
        self.off = np.einsum("ij,ij->i", n, self.a)
        self.vertices = self.a.copy()

    @classmethod
    def from_polygon(cls, poly, kind: str = "view") -> "Floor":
        poly = np.asarray(poly, float).reshape(-1, 2)
        return cls([{"a": poly[k - 1], "b": poly[k], "kind": kind} for k in range(len(poly))])

    @property
    def valid(self) -> bool:
        return len(self.n) >= 3

    def depth(self, p: np.ndarray) -> np.ndarray:
        """Signed distance inside each edge, (N, E); negative = outside that edge."""
        return np.asarray(p, float).reshape(-1, 2) @ self.n.T - self.off

    def view_distance(self, p: np.ndarray) -> np.ndarray:
        """Distance to the nearest view/range edge (inf if none)."""
        d = self.depth(p)
        if self.wall.all():
            return np.full(len(d), np.inf)
        return d[:, ~self.wall].min(axis=1)

    def extent(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def cross(self, prev: np.ndarray, cur: np.ndarray, alive: np.ndarray):
        """
        People who stepped out of the floor between prev and cur: (gone — left
        through a view edge, wall — index of the wall they ran into or −1,
        edge — the edge they crossed or −1).
        Decided by the edge *segment* their step crossed, so a doorway (a
        view segment in line with walls) lets people out only through the gap.
        """
        n = len(cur)
        gone, wall, edge = np.zeros(n, bool), np.full(n, -1, int), np.full(n, -1, int)
        d = self.depth(cur)
        idx = np.nonzero(alive & (d < -1e-9).any(axis=1))[0]
        if not len(idx):
            return gone, wall, edge
        p0, r = prev[idx], cur[idx] - prev[idx]
        e = self.b - self.a
        cr = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
        denom = cr(r[:, None, :], e[None])                                # (m, E)
        w = self.a[None] - p0[:, None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = cr(w, e[None]) / denom                                     # along the step
            u = cr(w, r[:, None, :]) / denom                               # along the edge
        hit = (np.abs(denom) > 1e-12) & (s >= -1e-9) & (s <= 1 + 1e-9) & (u >= -1e-6) & (u <= 1 + 1e-6)
        s = np.where(hit, s, np.inf)
        k = np.argmin(s, axis=1)
        none = ~np.isfinite(s[np.arange(len(idx)), k])
        k[none] = np.argmin(d[idx[none]], axis=1)                         # started outside: worst edge
        is_wall = self.wall[k]
        gone[idx[~is_wall]] = True
        wall[idx[is_wall]] = k[is_wall]
        edge[idx] = k
        return gone, wall, edge

    def slide(self, P: np.ndarray, v: np.ndarray, wall: np.ndarray, *heads: np.ndarray):
        """Put people who hit a wall back on it and drop their velocity into it (in place)."""
        hit = wall >= 0
        if not hit.any():
            return
        for _ in range(2):                                      # a corner can need two walls
            for k in np.unique(wall[hit]):
                sel = hit & (wall == k)
                nk = self.n[k]
                dk = P[sel] @ nk - self.off[k]
                P[sel] -= np.minimum(dk, 0)[:, None] * nk
                vn = v[sel] @ nk
                v[sel] -= np.minimum(vn, 0)[:, None] * nk
                for h in heads:
                    hn = h[sel] @ nk
                    h[sel] -= np.minimum(hn, 0)[:, None] * nk
            dd = self.depth(P)
            still = hit & (dd[:, self.wall] < -1e-9).any(axis=1) if self.wall.any() else hit & False
            if not still.any():
                break
            wk = np.nonzero(self.wall)[0]
            wall = wall.copy()
            wall[still] = wk[np.argmin(dd[still][:, self.wall], axis=1)]
            hit = still


# ─── What the twin learns about the scene ─────────────────────────────────────

class FlowField:
    """A frozen, smoothed copy of SceneMemory's flow grid, safe to read from another thread."""
    def __init__(self, origin, cell, direction, reliability):
        self.origin, self.cell = np.asarray(origin, float), float(cell)
        self.direction, self.reliability = direction, reliability     # (H, W, 2), (H, W)

    def at(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unit flow direction (N, 2) and reliability (N,) in [0, 1] at points."""
        p = np.asarray(p, float).reshape(-1, 2)
        h, w = self.reliability.shape
        ij = np.floor((p - self.origin) / self.cell).astype(int)
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
        ix, iy = np.clip(ij[:, 0], 0, w - 1), np.clip(ij[:, 1], 0, h - 1)
        d = self.direction[iy, ix]
        r = np.where(inside, self.reliability[iy, ix], 0.0)
        return d, r


class SceneMemory:
    """
    Learned online from the tracked crowd:
      • a flow field on a CELL_M grid — decaying sums of walkers' velocities,
        so the twin knows the scene's lanes and their direction;
      • entries — people first seen within ENTRY_MARGIN_M of a view edge who
        then *walked* at least ENTRY_INWARD_M into the scene within
        ENTRY_CONFIRM_S. Someone arriving is walking; in a dense crowd, a
        head re-detected near the edge after flickering is not, so it does
        not count. The entry rate is taken over the last ENTRY_WINDOW_S;
      • obstacles — floor cells nobody ever stands on while the floor all
        around them is busy (pillars, railings, barriers), from a slowly
        decaying occupancy map of everyone seen (see obstacles()).
    """
    CELL_M          = 1.0
    MAX_CELLS       = 160          # per side; bigger floors get coarser cells
    HALF_LIFE_S     = 180.0
    RELIABLE_N      = 4.0          # observations for a cell to be fully trusted
    ENTRY_MARGIN_M  = 2.0
    ENTRY_INWARD_M  = 1.0
    ENTRY_CONFIRM_S = 3.0
    ENTRY_WINDOW_S  = 120.0
    WARMUP_S        = 2.0          # tracks present at the start are not entries
    MIN_SPAN_S      = 5.0          # history needed before an entry rate is used
    PRIOR_SPAN_S    = 30.0         # the rate estimate starts from "no arrivals", worth this long
    OCC_HALF_LIFE_S = 1800.0       # occupancy memory for obstacles
    OBST_MIN_AGE_S  = 60.0         # watch this long …
    OBST_MIN_OBS    = 1500.0       # … and see this many person-positions before naming obstacles
    OBST_EMPTY      = 0.02         # a cell is "never used": below this share of its ring's use
    OBST_RING_BUSY  = 0.6          # … and this share of its ring (2–3 cells away) is busy

    def __init__(self, floor: Floor):
        self.floor = floor
        lo, hi = floor.extent() if floor.valid else (np.zeros(2), np.ones(2) * 10)
        span = np.maximum(hi - lo, 1.0)
        self.cell = max(self.CELL_M, float(span.max()) / self.MAX_CELLS)
        self.origin = lo - self.cell
        self.shape = tuple((np.ceil(span / self.cell).astype(int) + 3)[::-1])     # (H, W)
        self.sum_v   = np.zeros(self.shape + (2,))
        self.sum_spd = np.zeros(self.shape)
        self.count   = np.zeros(self.shape)
        self.occ     = np.zeros(self.shape)             # everyone, walking or not
        self.occ_total = 0.0
        self.t_last: Optional[float] = None
        self.t_start: Optional[float] = None
        self.known: set = set()
        self.pending: Dict[int, Tuple[float, np.ndarray, float]] = {}   # id → (t, pos, edge dist)
        self.entries: deque = deque(maxlen=2000)                        # (t, pos, vel)

    def set_floor(self, floor: Floor):
        """New floor edges (e.g. the learned far wall) on the same ground coordinates."""
        self.floor = floor

    def observe(self, t: float, ids: Sequence[int], pos: np.ndarray, vel: np.ndarray,
                walking: np.ndarray):
        pos = np.asarray(pos, float).reshape(-1, 2)
        vel = np.asarray(vel, float).reshape(-1, 2)
        if self.t_start is None:
            self.t_start = t
        if self.t_last is not None and t > self.t_last:
            decay = 0.5 ** ((t - self.t_last) / self.HALF_LIFE_S)
            self.sum_v *= decay
            self.sum_spd *= decay
            self.count *= decay
            od = 0.5 ** ((t - self.t_last) / self.OCC_HALF_LIFE_S)
            self.occ *= od
            self.occ_total *= od
        self.t_last = t

        # Where anyone stands (for obstacles)
        if len(pos):
            ij = np.floor((pos - self.origin) / self.cell).astype(int)
            ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < self.shape[0])
            np.add.at(self.occ, (ij[ok, 1], ij[ok, 0]), 1.0)
            self.occ_total += float(ok.sum())

        # Flow field from walkers
        w = np.asarray(walking, float) > 0.5
        if w.any():
            ij = np.floor((pos[w] - self.origin) / self.cell).astype(int)
            ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < self.shape[0])
            ix, iy, v = ij[ok, 0], ij[ok, 1], vel[w][ok]
            np.add.at(self.sum_v, (iy, ix), v)
            np.add.at(self.sum_spd, (iy, ix), np.linalg.norm(v, axis=1))
            np.add.at(self.count, (iy, ix), 1.0)

        # Entries through the view edges
        if not self.floor.valid:
            return
        dist = self.floor.view_distance(pos) if len(pos) else np.zeros(0)
        warm = t - self.t_start < self.WARMUP_S
        for k, i in zip(ids, range(len(pos))):
            if k in self.known:
                if k in self.pending:
                    t0, p0, d0 = self.pending[k]
                    if dist[i] - d0 >= self.ENTRY_INWARD_M and w[i]:
                        self.entries.append((t0, p0, vel[i].copy()))
                        del self.pending[k]
                    elif t - t0 > self.ENTRY_CONFIRM_S:
                        del self.pending[k]
                continue
            self.known.add(k)
            if not warm and dist[i] < self.ENTRY_MARGIN_M:
                self.pending[k] = (t, pos[i].copy(), float(dist[i]))
        if len(self.known) > 50000:                     # long runs: forget ids long gone
            self.known = set(ids) | set(self.pending)

    def flow_field(self) -> FlowField:
        sv = np.stack([gaussian_filter(self.sum_v[..., k], 1.0) for k in range(2)], axis=-1)
        ss = gaussian_filter(self.sum_spd, 1.0)
        sc = gaussian_filter(self.count, 1.0)
        mag = np.linalg.norm(sv, axis=-1)
        direction = np.divide(sv, mag[..., None], out=np.zeros_like(sv), where=mag[..., None] > 1e-9)
        coherence = np.divide(mag, ss, out=np.zeros_like(mag), where=ss > 1e-9)
        reliability = coherence * (1 - np.exp(-sc / self.RELIABLE_N))
        return FlowField(self.origin, self.cell, direction, np.clip(reliability, 0, 1))

    def obstacles(self) -> np.ndarray:
        """
        Centres (K, 2) of floor cells that are obstacles: after OBST_MIN_AGE_S
        and OBST_MIN_OBS person-positions, a cell well inside the floor that
        nobody has used (< OBST_EMPTY of the use around it) although at least
        OBST_RING_BUSY of the ring 2–3 cells around it is busy. Up to ~4 m
        across; bigger never-used areas are left alone (they may just be
        quiet, not blocked).
        """
        if (self.t_start is None or self.t_last - self.t_start < self.OBST_MIN_AGE_S
                or self.occ_total < self.OBST_MIN_OBS or not self.floor.valid):
            return np.zeros((0, 2))
        ring = np.ones((7, 7))
        ring[2:5, 2:5] = 0
        n_ring = ring.sum()
        used = self.occ[self.occ > 0]
        typical = float(np.median(used)) if len(used) else 0.0
        if typical <= 0:
            return np.zeros((0, 2))
        ring_use  = convolve(self.occ, ring, mode="constant") / n_ring
        ring_busy = convolve((self.occ > 0.1 * typical).astype(float), ring, mode="constant") / n_ring
        cand = (self.occ < self.OBST_EMPTY * ring_use) & (ring_busy >= self.OBST_RING_BUSY) & \
               (ring_use >= 0.3 * typical)
        iy, ix = np.nonzero(cand)
        centres = self.origin + (np.column_stack([ix, iy]) + 0.5) * self.cell
        if not len(centres):
            return centres
        inside = self.floor.depth(centres).min(axis=1) >= self.cell      # well inside the floor
        return centres[inside]

    def arrivals(self, now: float) -> Tuple[float, List[Tuple[np.ndarray, np.ndarray]]]:
        """Entry rate (people/s) over the recent window, and recent entries (newest first)."""
        if self.t_start is None:
            return 0.0, []
        span = min(self.ENTRY_WINDOW_S, now - self.t_start - self.WARMUP_S)
        if span < self.MIN_SPAN_S:                      # too little history for a rate
            return 0.0, []
        recent = [(p, v) for (t0, p, v) in reversed(self.entries) if t0 >= now - span]
        # Shrunk towards zero while the history is short (a Gamma prior):
        # a few entries in 10 s are weak evidence of a steady inflow
        return len(recent) / (span + self.PRIOR_SPAN_S), recent


# ─── The twin checks its own forecasts ────────────────────────────────────────

class ForecastSkill:
    """
    Scores the twin's own zone-count forecasts against what then happened,
    per forecast step, and learns how far to trust each step's predicted
    *change* over "no change": w_k = Σ Δpred·Δtrue / Σ Δpred² (ridge-shrunk
    towards PRIOR_W, forgetting with HALF_LIFE_S). Calibrated counts are
    now + w_k·(forecast − now), so the twin is never much worse than
    persistence on a scene where its agents mislead it, and keeps the full
    forecast where they don't. Also reports the running errors (skill).
    """
    PRIOR_W     = 1.0         # until it has evidence, trust the forecast as it is
    PRIOR_N     = 30.0          # strength of the prior, in (people of change)²
    HALF_LIFE_S = 600.0
    MAX_W       = 1.0         # trust the forecast fully at most — never exaggerate it

    def __init__(self, steps: int, step_s: float):
        self.steps, self.step_s = steps, step_s
        k = steps + 1
        self.sxy, self.sxx = np.zeros(k), np.zeros(k)
        self.err_model, self.err_raw, self.err_keep = np.zeros(k), np.zeros(k), np.zeros(k)
        self.n = np.zeros(k)
        self.pending: List[dict] = []
        self.t_last: Optional[float] = None

    def weights(self) -> np.ndarray:
        w = (self.sxy + self.PRIOR_W * self.PRIOR_N) / (self.sxx + self.PRIOR_N)
        w[0] = 1.0
        return np.clip(w, 0.0, self.MAX_W)

    def calibrate(self, now_counts, pred_counts) -> np.ndarray:
        """(steps + 1, Z) forecast counts → calibrated counts."""
        now = np.asarray(now_counts, float)[None, :]
        return now + self.weights()[:, None] * (np.asarray(pred_counts, float) - now)

    def add(self, t0: float, now_counts, pred_counts):
        """Remember a forecast made at t0 (zone counts now, and per step)."""
        self.pending.append({"t0": t0, "now": np.asarray(now_counts, float),
                             "pred": np.asarray(pred_counts, float),
                             "cal": self.calibrate(now_counts, pred_counts),
                             "done": np.zeros(self.steps + 1, bool)})
        self.pending = self.pending[-(self.steps + 5):]

    def observe(self, t: float, counts, tol: Optional[float] = None):
        """What the zones hold at time t: score pending forecasts whose step falls here."""
        counts = np.asarray(counts, float)
        tol = 0.5 * self.step_s if tol is None else tol
        if self.t_last is not None and t > self.t_last:
            decay = 0.5 ** ((t - self.t_last) / self.HALF_LIFE_S)
            for a in (self.sxy, self.sxx, self.err_model, self.err_raw, self.err_keep, self.n):
                a *= decay
        self.t_last = t
        keep = []
        for f in self.pending:
            k = int(round((t - f["t0"]) / self.step_s))
            if 1 <= k <= self.steps and not f["done"][k] and abs(t - f["t0"] - k * self.step_s) <= tol:
                f["done"][k] = True
                dp, da = f["pred"][k] - f["now"], counts - f["now"]
                self.sxy[k] += float(dp @ da)
                self.sxx[k] += float(dp @ dp)
                self.err_model[k] += float(np.abs(f["cal"][k] - counts).mean())
                self.err_raw[k]   += float(np.abs(f["pred"][k] - counts).mean())
                self.err_keep[k]  += float(np.abs(f["now"] - counts).mean())
                self.n[k] += 1
            if t - f["t0"] <= self.steps * self.step_s + tol:
                keep.append(f)
        self.pending = keep

    def report(self) -> dict:
        """Recent mean zone-count error per step: calibrated, raw agents, no change."""
        n = np.maximum(self.n, 1e-9)
        has = self.n > 0.5
        r = lambda a: [round(float(x), 2) if h else None for x, h in zip(a / n, has)]
        return {"step_s": self.step_s, "weight": np.round(self.weights(), 2).tolist(),
                "mae_twin": r(self.err_model), "mae_agents": r(self.err_raw),
                "mae_no_change": r(self.err_keep), "scored": np.round(self.n, 1).tolist()}


# ─── The forecast ─────────────────────────────────────────────────────────────

@dataclass
class ForecastResult:
    paths:      np.ndarray        # (M, steps + 1, 2) metres; frozen before entering / after leaving
    enter_step: np.ndarray        # (M,) step at which each person is in the scene (0 = now)
    exit_step:  np.ndarray        # (M,) step at which they left through a view edge, −1 = stays
    n_tracked:  int               # the first n_tracked rows are today's people, the rest arrivals

    def active(self) -> np.ndarray:
        """(M, steps + 1) bool: in the scene at each step."""
        k = np.arange(self.paths.shape[1])[None, :]
        ex = np.where(self.exit_step < 0, np.iinfo(np.int32).max, self.exit_step)[:, None]
        return (k >= self.enter_step[:, None]) & (k < ex)


@dataclass
class PanicResult:
    paths:      np.ndarray        # (N, steps + 1, 2) metres; frozen after leaving
    exit_step:  np.ndarray        # (N,) step at which each person left the view, −1 = still in
    density:    np.ndarray        # (N, steps + 1) local density around each person, persons/m²
    step_s:     float
    danger_density: float

    def active(self) -> np.ndarray:
        k = np.arange(self.paths.shape[1])[None, :]
        ex = np.where(self.exit_step < 0, np.iinfo(np.int32).max, self.exit_step)[:, None]
        return k < ex

    def summary(self) -> dict:
        """How fast the area empties, and where / when it gets dangerously dense."""
        act = self.active()
        n = max(len(self.paths), 1)
        remaining = act.sum(axis=0) / n
        t = np.arange(act.shape[1]) * self.step_s
        at = lambda share: (float(t[np.argmax(remaining <= 1 - share)])
                            if (remaining <= 1 - share).any() else None)
        dens = np.where(act, self.density, 0.0)
        danger = dens >= self.danger_density
        peak = np.unravel_index(np.argmax(dens), dens.shape) if dens.size else (0, 0)
        return {
            "people": int(len(self.paths)),
            "remaining": np.round(remaining, 3).tolist(),
            "half_out_s": at(0.5), "ninety_out_s": at(0.9),
            "max_density": round(float(dens.max()) if dens.size else 0.0, 2),
            "max_density_at": np.round(self.paths[peak[0], peak[1]], 2).tolist() if dens.size else None,
            "max_density_s": round(float(peak[1] * self.step_s), 1),
            "danger_first_s": (round(float(t[np.argmax(danger.any(axis=0))]), 1)
                               if danger.any() else None),
            "people_in_danger": int(danger.any(axis=1).sum()),
        }


class CrowdForecaster:
    """Anticipatory social force forecast (see the module docstring)."""

    def __init__(self, params: Optional[ForecastParams] = None):
        self.p = params or ForecastParams()

    # ── Pieces ───────────────────────────────────────────────────────────
    def _weidmann(self, rho: np.ndarray) -> np.ndarray:
        p = self.p
        rho = np.clip(rho, 0.05, p.rho_max - 1e-3)
        return 1 - np.exp(-p.weidmann_gamma * (1 / rho - 1 / p.rho_max))

    def _density(self, pos: np.ndarray, alive: np.ndarray) -> np.ndarray:
        rho = np.zeros(len(pos))
        idx = np.nonzero(alive)[0]
        if len(idx) > 1:
            tree = cKDTree(pos[idx])
            n = tree.query_ball_point(pos[idx], self.p.density_radius, return_length=True) - 1
            rho[idx] = n / (math.pi * self.p.density_radius ** 2)
        return rho

    def _ttc(self, ia, x, v, R, n) -> np.ndarray:
        """
        Power-law forces on pedestrians `ia` from relative positions x and
        velocities v of whatever they may hit (radius sum R), summed per
        pedestrian: (n, 2) accelerations.
        """
        p, f = self.p, np.zeros((n, 2))
        a  = np.einsum("ij,ij->i", v, v)
        b  = np.einsum("ij,ij->i", x, v)
        c  = np.einsum("ij,ij->i", x, x) - R ** 2
        disc = b * b - a * c
        ok = (a > 1e-9) & (b < 0) & (c > 0) & (disc > 0)
        if not ok.any():
            return f
        ia, x, v, a, b, disc = ia[ok], x[ok], v[ok], a[ok], b[ok], np.sqrt(disc[ok])
        tau = (-b - disc) / a
        near = tau < 3 * p.ttc_tau0
        ia, x, v, a, b, disc, tau = ia[near], x[near], v[near], a[near], b[near], disc[near], tau[near]
        mag = -p.ttc_k * np.exp(-tau / p.ttc_tau0) / (a * tau ** 2) * (2 / tau + 1 / p.ttc_tau0)
        force = mag[:, None] * (v - (b[:, None] * v - a[:, None] * x) / disc[:, None])
        # Step aside, not only brake: a head-on approach gives a force straight
        # back along v, which stalls a walker in front of a pillar. People pick
        # a side — away from where the other is, with a keep-right preference
        # that decides dead-ahead cases (and doesn't cancel out between the
        # cells of a wide obstacle)
        vh = v / np.sqrt(a)[:, None]
        side = x - np.einsum("ij,ij->i", x, vh)[:, None] * vh
        sn = np.linalg.norm(side, axis=1)
        right = np.column_stack([vh[:, 1], -vh[:, 0]])
        side = side / np.maximum(sn, 1e-9)[:, None] * (sn > 1e-3)[:, None] + p.ttc_keep_right * right
        side /= np.maximum(np.linalg.norm(side, axis=1), 1e-9)[:, None]
        force += p.ttc_sidestep * np.linalg.norm(force, axis=1)[:, None] * side
        for k in range(2):
            f[:, k] = np.bincount(ia, force[:, k], minlength=n)
        return f

    def _cap(self, f: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(f, axis=1)
        big = norm > self.p.max_force
        f[big] *= (self.p.max_force / norm[big])[:, None]
        return f

    def _avoidance(self, pos, vel, alive, moving) -> np.ndarray:
        """Time-to-collision power-law forces (Karamouzas et al. 2014), as accelerations."""
        p, f = self.p, np.zeros_like(pos)
        idx = np.nonzero(alive)[0]
        if len(idx) < 2:
            return f
        pairs = cKDTree(pos[idx]).query_pairs(p.ttc_cutoff, output_type="ndarray")
        if not len(pairs):
            return f
        i, j = idx[pairs[:, 0]], idx[pairs[:, 1]]
        keep = moving[i] | moving[j]                    # two people standing can't collide
        i, j = i[keep], j[keep]
        if not len(i):
            return f
        ia = np.concatenate([i, j])
        ib = np.concatenate([j, i])
        f = self._ttc(ia, pos[ia] - pos[ib], vel[ia] - vel[ib], 2 * p.radius, len(pos))
        return self._cap(f)

    def _obstacle_avoidance(self, pos, vel, alive, obst, obst_r) -> np.ndarray:
        """The same law against static obstacles (pillars, railings)."""
        f = np.zeros_like(pos)
        idx = np.nonzero(alive & (np.einsum("ij,ij->i", vel, vel) > 1e-6))[0]
        if not len(idx) or not len(obst):
            return f
        # Obstacles are few: dense distances beat per-person tree queries
        d = np.linalg.norm(pos[idx][:, None, :] - obst[None, :, :], axis=2)
        ii, jo = np.nonzero(d < self.p.ttc_cutoff)
        if not len(ii):
            return f
        ia = idx[ii]
        return self._cap(self._ttc(ia, pos[ia] - obst[jo], vel[ia], obst_r + self.p.radius, len(pos)))

    DETOURS = np.radians([0, -15, 15, -30, 30, -45, 45, -60, 60, -80, 80])   # right turns first

    def _detour(self, P, e, speed, alive, obst, obst_r) -> np.ndarray:
        """
        Heading change (radians) per walker whose next ~2.5 s of path runs into
        an obstacle: the smallest turn, right before left, that clears it —
        the way people re-plan around a pillar instead of pushing into it.
        """
        ang = np.zeros(len(P))
        if not len(obst):
            return ang
        look = np.maximum(1.5, speed * 2.5)
        idx = np.nonzero(alive & (speed > 0.1))[0]
        if not len(idx):
            return ang
        near = cKDTree(obst).query_ball_point(P[idx], look[idx] + obst_r)
        idx = np.array([i for i, js in zip(idx, near) if js], int)
        if not len(idx):
            return ang
        clear_r = obst_r + self.p.radius
        c, s_ = np.cos(self.DETOURS), np.sin(self.DETOURS)
        e_i = e[idx]                                                    # (m, 2)
        cand = np.stack([c[None] * e_i[:, :1] - s_[None] * e_i[:, 1:2],
                         s_[None] * e_i[:, :1] + c[None] * e_i[:, 1:2]], axis=-1)   # (m, k, 2)
        w = obst[None, :, :] - P[idx][:, None, :]                       # (m, o, 2)
        t = np.clip(np.einsum("mkd,mod->mko", cand, w), 0, look[idx][:, None, None])
        closest = P[idx][:, None, None, :] + t[..., None] * cand[:, :, None, :]
        dist = np.linalg.norm(obst[None, None] - closest, axis=-1).min(axis=2)       # (m, k)
        ok = dist >= clear_r
        first = np.where(ok.any(axis=1), ok.argmax(axis=1), 0)
        ang[idx] = self.DETOURS[first]
        return ang

    def _push_out(self, P, v, alive, obst, obst_r):
        """Nobody ends up inside an obstacle: move them to its rim and drop the inward speed."""
        if not len(obst):
            return
        d, j = cKDTree(obst).query(P)
        inside = alive & (d < obst_r)
        if inside.any():
            n = P[inside] - obst[j[inside]]
            dn = np.maximum(np.linalg.norm(n, axis=1), 1e-9)
            n = n / dn[:, None]
            P[inside] = obst[j[inside]] + n * obst_r
            vn = np.einsum("ij,ij->i", v[inside], n)
            v[inside] -= np.minimum(vn, 0)[:, None] * n

    # ── Forecast ─────────────────────────────────────────────────────────
    def run(self, pos, vel, vel_std=None, floor: Optional[Floor] = None,
            flow: Optional[FlowField] = None,
            arrivals: Tuple[float, list] = (0.0, []),
            obstacles: Optional[np.ndarray] = None, obstacle_r: float = 0.75) -> ForecastResult:
        """
        Forecast STEPS × STEP_S seconds ahead from positions (N, 2), velocities
        (N, 2) and velocity noise (N,), inside `floor`, following `flow`, with
        newcomers from `arrivals` = (rate per s, [(entry position, entry
        velocity), ...]), around `obstacles` (centres, radius obstacle_r).
        """
        p = self.p
        obst = np.asarray(obstacles if obstacles is not None else np.zeros((0, 2)), float).reshape(-1, 2)
        pos = np.asarray(pos, float).reshape(-1, 2)
        vel = np.asarray(vel, float).reshape(-1, 2)
        n = len(pos)
        vel_std = np.zeros(n) if vel_std is None else np.asarray(vel_std, float).reshape(n)
        horizon = p.steps * p.step_s

        # Newcomers, evenly spread over the horizon at the learned rate
        rate, samples = arrivals
        spawn_step, spawn_pos, spawn_vel = [], [], []
        if rate > 0 and samples:
            m = min(int(rate * horizon + 0.5), p.max_arrivals)
            for k in range(m):
                t_k = (k + 0.5) / rate
                sp, sv = samples[k % len(samples)]
                spawn_step.append(max(1, int(math.ceil(t_k / p.step_s))))
                spawn_pos.append(sp)
                spawn_vel.append(sv)
        m = len(spawn_step)
        P = np.vstack([pos] + ([np.array(spawn_pos)] if m else []))
        V = np.vstack([vel] + ([np.array(spawn_vel)] if m else []))
        S = np.concatenate([vel_std, np.zeros(m)])
        enter = np.concatenate([np.zeros(n, int), np.array(spawn_step, int)])
        M = len(P)

        speed = np.linalg.norm(V, axis=1)
        walk  = walking_weight(speed, S, p)
        walk[n:] = 1.0                                   # newcomers were seen walking in
        # The filtered velocity is the unbiased estimate of where walkers go;
        # the de-biased speed only decides who walks (walking_weight)
        v_des = np.minimum(speed, p.max_speed) * walk
        e0 = np.divide(V, speed[:, None], out=np.zeros_like(V), where=speed[:, None] > 1e-9)
        moving = walk > 0
        dodge = p.standing_dodge + (1 - p.standing_dodge) * walk     # standing people barely step aside

        alive = enter == 0
        exit_step = np.full(M, -1, int)
        # Speeds are scaled by how much denser it gets around each person than it is now
        g0 = self._weidmann(self._density(P, alive))
        paths = np.zeros((M, p.steps + 1, 2))
        paths[:, 0] = P
        v = V * walk[:, None]                            # standing people start still
        dt = p.step_s / p.substeps
        walls = bool(floor is not None and floor.valid and floor.wall.any())
        for step in range(1, p.steps + 1):
            t = (step - 1) * p.step_s
            # Arrivals appear at their entry point at step `enter` and move from there
            born = (enter == step - 1) & (enter > 0)
            alive |= born
            rho = self._density(P, alive)
            detour = None
            if born.any():
                g0[born] = self._weidmann(rho[born])
                v[born] = V[born]
            speed_scale = np.clip(self._weidmann(rho) / np.maximum(g0, 1e-3), 0.0, p.weidmann_max_gain)
            for sub in range(p.substeps):
                ts = t + sub * dt
                # Heading: own heading, turning towards the scene's flow
                e = e0
                if flow is not None:
                    fd, fr = flow.at(P)
                    beta = (1 - math.exp(-ts / p.heading_memory_s)) * fr * p.flow_max
                    beta = np.where(np.einsum("ij,ij->i", fd, e0) > -0.2, beta, 0.0)  # never U-turn
                    e = (1 - beta)[:, None] * e0 + beta[:, None] * fd
                    en = np.linalg.norm(e, axis=1)
                    e = np.divide(e, en[:, None], out=e0.copy(), where=en[:, None] > 1e-9)
                if len(obst) and sub == 0:              # re-plan around obstacles each step
                    detour = self._detour(P, e, v_des * speed_scale, alive, obst, obstacle_r)
                if detour is not None and detour.any():
                    c, s_ = np.cos(detour), np.sin(detour)
                    e = np.column_stack([c * e[:, 0] - s_ * e[:, 1], s_ * e[:, 0] + c * e[:, 1]])
                desired = (v_des * speed_scale)[:, None] * e
                acc = (desired - v) / p.tau + dodge[:, None] * self._avoidance(P, v, alive, moving)
                if len(obst):
                    acc += self._obstacle_avoidance(P, v, alive, obst, obstacle_r)
                v = np.where(alive[:, None], v + acc * dt, 0.0)
                sp = np.linalg.norm(v, axis=1)
                fast = sp > p.max_speed
                v[fast] *= (p.max_speed / sp[fast])[:, None]
                prev = P
                P = np.where(alive[:, None], P + v * dt, P)
                self._push_out(P, v, alive, obst, obstacle_r)
                if floor is not None and floor.valid:
                    # Out through the camera's view: gone. Into a wall: slide along it
                    gone, wall, _ = floor.cross(prev, P, alive)
                    if gone.any():
                        exit_step[gone] = step
                        alive &= ~gone
                    if walls:
                        floor.slide(P, v, wall, e0)
            paths[:, step] = P
        return ForecastResult(paths=paths, enter_step=enter, exit_step=exit_step, n_tracked=n)

    # ── What-if: escape panic ────────────────────────────────────────────
    PANIC_SPEED    = 3.0      # m/s people try to run at when panicking (Helbing et al. use up to 5)
    CONTACT_R      = 0.2      # body half-width in a crush (bodies compress below the 0.25 m comfort radius)
    HERD_RADIUS    = 2.0      # m: whose direction a panicking person follows
    DANGER_DENSITY = 5.0      # persons/m²: crush risk (≈ Fruin F; Still's high-risk threshold)
    CONTACT_PASSES = 8
    EXIT_FLOW      = 1.3      # people per metre of exit per second: the measured maximum
                              # (Weidmann 1993; SFPE Handbook). Frictionless bodies would
                              # otherwise pour through a door like water

    @staticmethod
    def _nearest_exit(P: np.ndarray, floor: Optional[Floor]) -> Optional[np.ndarray]:
        """Nearest point on any edge of the camera's view (the way out of the area), per person."""
        if floor is None or not floor.valid or floor.wall.all():
            return None
        a, b = floor.a[~floor.wall], floor.b[~floor.wall]
        ab = b - a
        t = np.clip(np.einsum("nej,ej->ne", P[:, None, :] - a[None], ab) /
                    np.maximum(np.einsum("ej,ej->e", ab, ab), 1e-9)[None], 0, 1)
        q = a[None] + t[..., None] * ab[None]                       # (N, E, 2)
        k = np.argmin(np.linalg.norm(q - P[:, None, :], axis=2), axis=1)
        return q[np.arange(len(P)), k]

    def panic(self, pos, vel, vel_std=None, floor: Optional[Floor] = None,
              obstacles: Optional[np.ndarray] = None, obstacle_r: float = 0.75,
              level: float = 0.8, steps: int = 50, step_s: float = 0.8,
              substeps: int = 8) -> PanicResult:
        """
        What if this crowd panicked now? Escape-panic model (Helbing, Farkas &
        Vicsek, Nature 2000): with panic level p, everyone's desired speed
        rises towards PANIC_SPEED, their direction mixes the way out (the
        nearest edge of the view) with their neighbours' average direction
        (herding, weight p), anticipation of collisions fades (× 1 − p), and
        bodies press against each other (overlaps are resolved as contacts).
        Walls, obstacles and exits work as in run(). Returns every path and
        each person's local density over time — where it passes
        DANGER_DENSITY is where a crush could happen.
        """
        p = self.p
        level = float(np.clip(level, 0.0, 1.0))
        P = np.asarray(pos, float).reshape(-1, 2).copy()
        V0 = np.asarray(vel, float).reshape(-1, 2)
        n = len(P)
        obst = np.asarray(obstacles if obstacles is not None else np.zeros((0, 2)), float).reshape(-1, 2)
        paths = np.zeros((n, steps + 1, 2))
        dens = np.zeros((n, steps + 1))
        paths[:, 0] = P
        exit_step = np.full(n, -1, int)
        alive = np.ones(n, bool)
        if n == 0:
            return PanicResult(paths, exit_step, dens, step_s, self.DANGER_DENSITY)
        speed0 = np.linalg.norm(V0, axis=1)
        v_want = (1 - level) * np.maximum(speed0, 1.3) + level * self.PANIC_SPEED
        v = V0.copy()
        heading = np.divide(V0, speed0[:, None], out=np.zeros_like(V0), where=speed0[:, None] > 1e-6)
        dt = step_s / substeps
        r_dens = 1.0
        if floor is not None and floor.valid:                  # exit capacity per edge, people/s
            cap_rate = np.where(floor.wall, 0.0,
                                self.EXIT_FLOW * np.linalg.norm(floor.b - floor.a, axis=1))
            tokens = cap_rate * 0.5
        for step in range(steps + 1):
            # Local density (persons/m² within 1 m) — the crush measure
            idx = np.nonzero(alive)[0]
            if len(idx) > 1:
                cnt = cKDTree(P[idx]).query_ball_point(P[idx], r_dens, return_length=True)
                dens[idx, step] = cnt / (math.pi * r_dens ** 2)
            if step == steps:
                break
            # Where each person runs: the way out, mixed with the herd's direction
            goal_pt = self._nearest_exit(P, floor)
            goal = heading.copy() if goal_pt is None else goal_pt - P
            gn = np.linalg.norm(goal, axis=1)
            goal = np.divide(goal, gn[:, None], out=heading.copy(), where=gn[:, None] > 1e-6)
            herd = np.zeros_like(P)
            if len(idx) > 1:
                pairs = cKDTree(P[idx]).query_pairs(self.HERD_RADIUS, output_type="ndarray")
                if len(pairs):
                    sp = np.linalg.norm(v, axis=1)
                    u = np.divide(v, sp[:, None], out=np.zeros_like(v), where=sp[:, None] > 1e-6)
                    i, j = idx[pairs[:, 0]], idx[pairs[:, 1]]
                    for k in range(2):
                        herd[:, k] = np.bincount(i, u[j, k], minlength=n) + np.bincount(j, u[i, k], minlength=n)
            hn = np.linalg.norm(herd, axis=1)
            herd = np.divide(herd, hn[:, None], out=goal.copy(), where=hn[:, None] > 1e-6)
            e = (1 - level) * goal + level * herd
            en = np.linalg.norm(e, axis=1)
            e = np.divide(e, en[:, None], out=goal.copy(), where=en[:, None] > 1e-6)
            heading = e
            moving = np.ones(n, bool)
            for _ in range(substeps):
                prev = P.copy()
                acc = (v_want[:, None] * e - v) / p.tau
                if level < 1:
                    acc += (1 - level) * self._avoidance(P, v, alive, moving)
                if len(obst):
                    acc += self._obstacle_avoidance(P, v, alive, obst, obstacle_r)
                v = np.where(alive[:, None], v + acc * dt, 0.0)
                sp = np.linalg.norm(v, axis=1)
                fast = sp > 1.2 * v_want
                v[fast] *= (1.2 * v_want[fast] / sp[fast])[:, None]
                P = np.where(alive[:, None], P + v * dt, P)
                # Bodies in contact: push overlapping people apart (a few relaxation passes)
                for _ in range(self.CONTACT_PASSES):
                    ai = np.nonzero(alive)[0]
                    if len(ai) < 2:
                        break
                    prs = cKDTree(P[ai]).query_pairs(2 * self.CONTACT_R, output_type="ndarray")
                    if not len(prs):
                        break
                    i, j = ai[prs[:, 0]], ai[prs[:, 1]]
                    d = P[i] - P[j]
                    dn = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
                    push = ((2 * self.CONTACT_R - dn) / 2 / dn)[:, None] * d
                    np.add.at(P, i, push)
                    np.add.at(P, j, -push)
                self._push_out(P, v, alive, obst, obstacle_r)
                if floor is not None and floor.valid:
                    gone, wall, edge = floor.cross(prev, P, alive)
                    # An edge lets out at most EXIT_FLOW people per metre per second;
                    # the rest queue at it (as if it were a wall, for now)
                    tokens = np.minimum(tokens + cap_rate * dt, cap_rate * 1.0)
                    for k in np.unique(edge[gone]):
                        who = np.nonzero(gone & (edge == k))[0]
                        let = int(tokens[k])
                        if len(who) > let:
                            held = who[let:]
                            gone[held] = False
                            wall[held] = k
                        tokens[k] -= min(let, len(who))
                    if gone.any():
                        exit_step[gone] = step + 1
                        alive &= ~gone
                    floor.slide(P, v, wall)
                # Position-based dynamics: people move only as far as the bodies,
                # walls and obstacles around them let them — so a blocked crowd
                # really stops instead of keeping its running speed
                v = np.where(alive[:, None], (P - prev) / dt, 0.0)
            paths[:, step + 1] = P
        return PanicResult(paths, exit_step, dens, step_s, self.DANGER_DENSITY)
