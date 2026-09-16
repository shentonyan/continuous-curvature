#!/usr/bin/env python3
"""
curvcont.py -- curvature-continuity analysis and repair for vector outlines
               (a Kessler-style tool)

It does three things:
  1. Extract    pull a glyph outline out of a .ttf / .otf, convert quadratics to cubics
                exactly, and build a contour/segment structure
  2. Diagnose   compute signed curvature kappa(t) per segment; classify every node as
                G0 (corner) / G1 (tangent-continuous) / G2 (curvature-continuous) and
                quantify the kappa jump at each one
  3. Repair     a regularised nonlinear least-squares solve over handle lengths and node
                normal offsets, lifting smooth nodes to G2 while barely moving the glyph

The maths
---------
A cubic Bezier P0 P1 P2 P3 has a closed form for signed curvature at both ends:

    kappa(0) = (2/3) * (P1-P0) x (P2-P1) / |P1-P0|^3
    kappa(1) = (2/3) * (P2-P1) x (P3-P2) / |P3-P2|^3

At a node that is already G1 the two sides share one tangent direction u, and this
collapses to

    kappa = (2/3) * h / l^2      h = perpendicular distance from the control point to the
                                     tangent line,  l = handle length

so  G2  <=>  h_in / l_in^2 = h_out / l_out^2. That is the exact version of what type
designers call "balancing the handles".

The parameterisation keeps tangent directions fixed (so G1 holds by construction) and
frees only:
  - the two handle lengths l of each curve segment
  - a normal offset delta at each smooth node (nodes touching a line segment are frozen,
    since moving them would break G1)
Residuals = the kappa jump at each smooth node (normalised by em)
          + a curvature-profile anchor (see below)
          + regularisation on how far handles and nodes moved.

Curve/line nodes require kappa = 0 on the curve side; those are counted separately in the
report. Handle length alone can usually push the residual down to ~1e-3 (x em), at the
cost of visibly stretched handles. If that is not acceptable, the right fix is to insert
an Apple-squircle-style curvature ramp at the seam -- see squircle.py.

Usage
-----
    python3 curvcont.py --font /path/to/Font.ttf --char S --out out/
    python3 curvcont.py --font ... --char a --corner-deg 20 --reg 0.05
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np


# ==========================================================================
# 0. Basic geometry
# ==========================================================================
def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def unit(v):
    n = np.hypot(v[0], v[1])
    return v / n if n > 1e-12 else np.array([1.0, 0.0])


def bez_eval(P, t):
    """Cubic Bezier point plus first and second derivative. P: (4,2), t: (N,) -> (N,2) x 3"""
    P = np.asarray(P, float)
    t = np.asarray(t, float).reshape(-1, 1)
    m = 1.0 - t
    B = m**3 * P[0] + 3 * m**2 * t * P[1] + 3 * m * t**2 * P[2] + t**3 * P[3]
    d1 = 3 * (m**2 * (P[1] - P[0]) + 2 * m * t * (P[2] - P[1]) + t**2 * (P[3] - P[2]))
    d2 = 6 * (m * (P[2] - 2 * P[1] + P[0]) + t * (P[3] - 2 * P[2] + P[1]))
    return B, d1, d2


def bez_kappa(P, t):
    """Signed curvature kappa(t)"""
    _, d1, d2 = bez_eval(P, t)
    num = cross2(d1, d2)
    den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5
    return np.where(den > 1e-14, num / np.maximum(den, 1e-14), 0.0)


def kappa_end0(P):
    """Closed form for kappa(0)"""
    a, b = P[1] - P[0], P[2] - P[1]
    L = math.hypot(*a)
    return 0.0 if L < 1e-12 else (2.0 / 3.0) * float(a[0] * b[1] - a[1] * b[0]) / L**3


def kappa_end1(P):
    """Closed form for kappa(1)"""
    a, b = P[2] - P[1], P[3] - P[2]
    L = math.hypot(*b)
    return 0.0 if L < 1e-12 else (2.0 / 3.0) * float(a[0] * b[1] - a[1] * b[0]) / L**3


# ==========================================================================
# 1. Extraction: font -> contours / segments
# ==========================================================================
@dataclass
class Seg:
    """One segment: a line (is_line) or a cubic Bezier. Endpoints come from the contour."""
    is_line: bool
    h0: float = 0.0        # handle length at the start node
    h1: float = 0.0        # handle length at the end node


@dataclass
class Contour:
    """Closed contour: N nodes and N segments. Segment j runs node j -> j+1 (mod N)."""
    nodes: np.ndarray                  # (N,2) node positions
    segs: list                         # N Seg objects
    tan_out: np.ndarray = None         # (N,2) unit tangent leaving each node
    tan_in: np.ndarray = None          # (N,2) unit tangent arriving at each node

    def ctrl(self, j) -> np.ndarray:
        """The four control points of segment j"""
        N = len(self.segs)
        a, b = j, (j + 1) % N
        P0, P3 = self.nodes[a], self.nodes[b]
        s = self.segs[j]
        if s.is_line:
            return np.array([P0, P0 + (P3 - P0) / 3.0, P0 + 2 * (P3 - P0) / 3.0, P3])
        return np.array([P0, P0 + s.h0 * self.tan_out[a], P3 - s.h1 * self.tan_in[b], P3])


def extract_glyph(font_path: str, char: str, max_err: float = 0.05):
    """Pull one character's outline out of a font. Returns (contours, upem, glyph_name)."""
    from fontTools.ttLib import TTFont
    from fontTools.pens.recordingPen import RecordingPen
    from fontTools.pens.qu2cuPen import Qu2CuPen

    font = TTFont(font_path, fontNumber=0)
    upem = font["head"].unitsPerEm
    cmap = font.getBestCmap()
    cp = ord(char)
    if cp not in cmap:
        raise SystemExit(f"the font has no character {char!r} (U+{cp:04X})")
    gname = cmap[cp]

    rec = RecordingPen()
    pen = Qu2CuPen(rec, max_err=max_err * upem / 1000.0, all_cubic=True)
    font.getGlyphSet()[gname].draw(pen)

    contours, cur_pts, cur_segs = [], [], []

    def flush():
        if len(cur_pts) >= 2:
            contours.append((list(cur_pts), list(cur_segs)))
        cur_pts.clear(); cur_segs.clear()

    for op, args in rec.value:
        if op == "moveTo":
            flush(); cur_pts.append(np.array(args[0], float))
        elif op == "lineTo":
            cur_pts.append(np.array(args[0], float)); cur_segs.append(("line", None))
        elif op == "curveTo":
            c1, c2, p3 = [np.array(a, float) for a in args]
            cur_pts.append(p3); cur_segs.append(("curve", (c1, c2)))
        elif op in ("closePath", "endPath"):
            if len(cur_pts) >= 2 and np.allclose(cur_pts[-1], cur_pts[0], atol=1e-9):
                cur_pts.pop()                       # last point duplicates the first
            if len(cur_segs) < len(cur_pts):        # implicit closing line
                cur_segs.append(("line", None))
            flush()
    flush()

    out = []
    for pts, segs in contours:
        nodes = np.array(pts, float)
        N = len(nodes)
        segs = segs[:N]
        ct = Contour(nodes=nodes, segs=[], tan_out=np.zeros((N, 2)), tan_in=np.zeros((N, 2)))
        for j, (kind, data) in enumerate(segs):
            a, b = j, (j + 1) % N
            P0, P3 = nodes[a], nodes[b]
            if kind == "line":
                ct.segs.append(Seg(is_line=True))
                d = unit(P3 - P0)
                ct.tan_out[a], ct.tan_in[b] = d, d
            else:
                c1, c2 = data
                h0, h1 = float(np.hypot(*(c1 - P0))), float(np.hypot(*(P3 - c2)))
                ct.segs.append(Seg(is_line=False, h0=h0, h1=h1))
                ct.tan_out[a] = unit(c1 - P0) if h0 > 1e-9 else unit(P3 - P0)
                ct.tan_in[b] = unit(P3 - c2) if h1 > 1e-9 else unit(P3 - P0)
        out.append(ct)
    return out, upem, gname


# ==========================================================================
# 2. Diagnosis
# ==========================================================================
@dataclass
class NodeReport:
    contour: int
    node: int
    pos: tuple
    angle_deg: float     # tangent angle between the two sides (the G1 defect); 0 = tangent
    k_in: float
    k_out: float
    kind: str            # corner / smooth
    line_adjacent: bool

    @property
    def jump(self):
        return abs(self.k_in - self.k_out)


def diagnose(contours, upem, corner_deg=15.0):
    reps = []
    for ci, ct in enumerate(contours):
        N = len(ct.segs)
        for i in range(N):
            j_in = (i - 1) % N
            Pin, Pout = ct.ctrl(j_in), ct.ctrl(i)
            k_in, k_out = kappa_end1(Pin), kappa_end0(Pout)
            t_in = unit(Pin[3] - Pin[2]); t_out = unit(Pout[1] - Pout[0])
            ang = math.degrees(math.atan2(abs(float(cross2(t_in, t_out))),
                                          float(np.dot(t_in, t_out))))
            la = ct.segs[j_in].is_line or ct.segs[i].is_line
            reps.append(NodeReport(ci, i, tuple(np.round(ct.nodes[i], 2)), ang,
                                   k_in, k_out,
                                   "corner" if ang > corner_deg else "smooth", la))
    return reps


# ==========================================================================
# 3. The G2 solve
# ==========================================================================
def solve_g2(contours, upem, corner_deg=15.0, reg=0.04, node_reg=40.0,
             anchor=0.25, max_nfev=400, verbose=True):
    """Return a G2-corrected copy of the contours plus solver info."""
    from scipy.optimize import least_squares
    import copy

    cts = copy.deepcopy(contours)
    S = float(upem)

    # ---- assemble the unknowns ----
    xs, meta = [], []
    for ci, ct in enumerate(cts):
        for j, sg in enumerate(ct.segs):
            if not sg.is_line:
                meta.append(("h0", ci, j)); xs.append(sg.h0 / S)
                meta.append(("h1", ci, j)); xs.append(sg.h1 / S)

    for ci, ct in enumerate(cts):
        N = len(ct.segs)
        for i in range(N):
            j_in = (i - 1) % N
            la = ct.segs[j_in].is_line or ct.segs[i].is_line
            t_in, t_out = ct.tan_in[i], ct.tan_out[i]
            ang = math.degrees(math.atan2(abs(float(cross2(t_in, t_out))),
                                          float(np.dot(t_in, t_out))))
            if ang <= corner_deg and not la:   # smooth and not touching a line -> may slide
                meta.append(("dn", ci, i)); xs.append(0.0)

    x0 = np.array(xs, float)
    is_dn = np.array([m[0] == "dn" for m in meta], bool)   # node offsets vs handle lengths
    base_nodes = [ct.nodes.copy() for ct in cts]

    # Force G1 at smooth nodes: give both sides the bisector of the two tangents.
    for ci, ct in enumerate(cts):
        N = len(ct.segs)
        for i in range(N):
            j_in = (i - 1) % N
            if ct.segs[j_in].is_line or ct.segs[i].is_line:
                continue
            t_in, t_out = ct.tan_in[i], ct.tan_out[i]
            ang = math.degrees(math.atan2(abs(float(cross2(t_in, t_out))),
                                          float(np.dot(t_in, t_out))))
            if ang <= corner_deg:
                d = unit(t_in + t_out)
                ct.tan_in[i] = d; ct.tan_out[i] = d

    smooth_nodes = []
    for ci, ct in enumerate(cts):
        N = len(ct.segs)
        for i in range(N):
            j_in = (i - 1) % N
            t_in, t_out = ct.tan_in[i], ct.tan_out[i]
            ang = math.degrees(math.atan2(abs(float(cross2(t_in, t_out))),
                                          float(np.dot(t_in, t_out))))
            if ang <= corner_deg:
                smooth_nodes.append((ci, i, j_in))

    def apply(x):
        for ci, ct in enumerate(cts):
            ct.nodes[:] = base_nodes[ci]
        for idx, (tag, ci, k) in enumerate(meta):
            if tag == "h0":   cts[ci].segs[k].h0 = max(x[idx], 1e-4) * S
            elif tag == "h1": cts[ci].segs[k].h1 = max(x[idx], 1e-4) * S
            else:
                ct = cts[ci]
                t = ct.tan_out[k]
                n = np.array([-t[1], t[0]])
                ct.nodes[k] = base_nodes[ci][k] + x[idx] * S * n

    # Curvature anchors: the mid-value of kappa on the two sides of each smooth node.
    # Without this the solver happily buys "no jumps" with wildly oscillating kappa.
    apply(x0)
    anchors = [0.5 * (kappa_end1(cts[ci].ctrl(j_in)) + kappa_end0(cts[ci].ctrl(i)))
               for ci, i, j_in in smooth_nodes]

    def residuals(x):
        apply(x)
        res = []
        for idx, (ci, i, j_in) in enumerate(smooth_nodes):
            ct = cts[ci]
            ka, kb = kappa_end1(ct.ctrl(j_in)), kappa_end0(ct.ctrl(i))
            res.append((ka - kb) * S)                                    # G2
            res.append((0.5 * (ka + kb) - anchors[idx]) * S * anchor)    # profile anchor
        w = np.where(is_dn, reg * 60.0 * node_reg, reg * 60.0)
        res.extend(list((x - x0) * w))
        return np.array(res, float)

    nS = len(smooth_nodes)
    r0 = residuals(x0)
    sol = least_squares(residuals, x0, method="trf", max_nfev=max_nfev,
                        xtol=1e-12, ftol=1e-12, gtol=1e-12)
    apply(sol.x)
    info = {
        "unknowns": len(x0),
        "smooth_nodes": nS,
        "cost_before": float(0.5 * np.sum(r0[0:2 * nS:2] ** 2)),
        "cost_after": float(0.5 * np.sum(residuals(sol.x)[0:2 * nS:2] ** 2)),
        "nfev": int(sol.nfev),
    }
    if verbose:
        print(f"[solve] {info['unknowns']} unknowns, {info['smooth_nodes']} smooth nodes, "
              f"{info['nfev']} evaluations")
    return cts, info


# ==========================================================================
# 4. Sampling / deviation / export
# ==========================================================================
def sample_contours(contours, per_seg=48):
    """One (pts (M,2), kappa (M,), seg_index (M,)) tuple per contour."""
    out = []
    for ct in contours:
        P, K, SI = [], [], []
        for j in range(len(ct.segs)):
            C = ct.ctrl(j)
            t = np.linspace(0, 1, per_seg, endpoint=False)
            B, _, _ = bez_eval(C, t)
            P.append(B)
            K.append(np.zeros(per_seg) if ct.segs[j].is_line else bez_kappa(C, t))
            SI.append(np.full(per_seg, j))
        out.append((np.vstack(P), np.concatenate(K), np.concatenate(SI)))
    return out


def max_deviation(before, after, per_seg=60):
    """Maximum one-sided Hausdorff distance between the two outlines, in font units."""
    worst = 0.0
    for (pa, _, _), (pb, _, _) in zip(sample_contours(before, per_seg),
                                      sample_contours(after, per_seg)):
        d = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)).min(1)
        worst = max(worst, float(d.max()))
    return worst


def to_svg_path(contours):
    def f(v):
        return f"{v:.2f}".rstrip("0").rstrip(".")
    parts = []
    for ct in contours:
        N = len(ct.segs)
        parts.append(f"M {f(ct.nodes[0][0])} {f(ct.nodes[0][1])}")
        for j in range(N):
            C = ct.ctrl(j)
            if ct.segs[j].is_line:
                parts.append(f"L {f(C[3][0])} {f(C[3][1])}")
            else:
                parts.append(f"C {f(C[1][0])} {f(C[1][1])} {f(C[2][0])} {f(C[2][1])} "
                             f"{f(C[3][0])} {f(C[3][1])}")
        parts.append("Z")
    return " ".join(parts)


def write_svg(path, contours, upem, fill="#111"):
    d = to_svg_path(contours)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 {-upem} {upem} {upem*1.4}">\n'
           f'  <g transform="scale(1,-1)">\n'
           f'    <path d="{d}" fill="{fill}" fill-rule="nonzero"/>\n'
           f'  </g>\n</svg>\n')
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(svg)


# ==========================================================================
# 5. Rendering (outline + magnifier + red/green curvature combs)
# ==========================================================================
PALETTE = {
    "outline": "#3BB3D9",
    "before":  "#E24A4A",
    "after":   "#2ECC9B",
    "ink":     "#1B1E24",
    "muted":   "#9AA2AD",
    "paper":   "#FFFFFF",
}


def _comb(pts, kap, scale, orient):
    """Comb tips: offset each sample along the outward normal by kappa * scale."""
    d = np.gradient(pts, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= np.maximum(np.hypot(n[:, 0], n[:, 1])[:, None], 1e-12)
    return pts + orient * n * (kap * scale)[:, None]


def _orient(pts):
    """Contour winding, so the comb always grows away from the filled side."""
    a = float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
    return 1.0 if a > 0 else -1.0


def _draw_comb(ax, samples, color, scale, every=2, lw=.5, alpha=.45, closed_lw=1.5,
               outline_color=None, zorder=2):
    for pts, kap, _ in samples:
        o = _orient(pts)
        cl = np.vstack([pts, pts[:1]])
        if outline_color:
            ax.plot(cl[:, 0], cl[:, 1], color=outline_color, lw=closed_lw, zorder=zorder + 2)
        tips = _comb(pts, kap, scale, o)
        segs = np.empty((len(pts[::every]) * 3, 2)); segs[:] = np.nan
        segs[0::3] = pts[::every]; segs[1::3] = tips[::every]
        ax.plot(segs[:, 0], segs[:, 1], color=color, lw=lw, alpha=alpha, zorder=zorder)
        ax.plot(np.append(tips[:, 0], tips[0, 0]), np.append(tips[:, 1], tips[0, 1]),
                color=color, lw=1.0, alpha=.95, zorder=zorder + 1)


def render(before, after, upem, reports, out_png, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    sb, sa = sample_contours(before, 72), sample_contours(after, 72)
    kmax = max(float(np.abs(np.concatenate([k for _, k, _ in sb])).max()), 1e-9)
    scale = 0.11 * upem / kmax

    sm = [r for r in reports if r.kind == "smooth"]
    cc_ = [r for r in sm if not r.line_adjacent] or sm
    worst = max(cc_, key=lambda r: r.jump) if cc_ else None
    corners = [r for r in reports if r.kind == "corner"]

    fig = plt.figure(figsize=(15.5, 6.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.45], hspace=.30, wspace=.16)
    axL, axR = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
    axZ, axK = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])

    for ax, samp, col, lab in [(axL, sb, PALETTE["before"], "Before"),
                               (axR, sa, PALETTE["after"], "After")]:
        _draw_comb(ax, samp, col, scale, outline_color=PALETTE["outline"])
        for r in corners:
            ax.scatter([r.pos[0]], [r.pos[1]], s=16, marker="s",
                       color=PALETTE["ink"], zorder=8)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{lab} - curvature comb", fontsize=12, color=PALETTE["ink"])

    if worst is not None:
        R = 0.16 * upem
        cx, cy = worst.pos
        for ax in (axL, axR):
            ax.add_patch(Circle((cx, cy), R, fill=False, lw=1.2,
                                color=PALETTE["muted"], zorder=9))
        _draw_comb(axZ, sb, PALETTE["before"], scale, every=1, lw=.6, alpha=.42,
                   outline_color="#B9C4CC", closed_lw=1.3)
        _draw_comb(axZ, sa, PALETTE["after"], scale, every=1, lw=.6, alpha=.42,
                   outline_color=PALETTE["outline"], closed_lw=1.6, zorder=6)
        axZ.scatter([cx], [cy], s=30, color=PALETTE["ink"], zorder=12)
        axZ.set_xlim(cx - R, cx + R); axZ.set_ylim(cy - R * .75, cy + R * .75)
        axZ.set_aspect("equal"); axZ.axis("off")
        axZ.set_title(f"Worst node - contour {worst.contour}, node {worst.node} "
                      f"(red = before, green = after)",
                      fontsize=11, color=PALETTE["ink"])

    for samp, col, lab in [(sb, PALETTE["before"], "Before"),
                           (sa, PALETTE["after"], "After")]:
        pts, kap, _ = samp[0]
        cl = np.vstack([pts, pts[:1]])
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(cl, axis=0).T))])
        axK.plot(s / upem, np.append(kap, kap[0]) * upem, color=col, lw=1.4, label=lab)
    axK.axhline(0, color=PALETTE["muted"], lw=.8, ls=":")
    axK.set_xlabel("arc length / em", fontsize=10)
    axK.set_ylabel("$\\kappa$ $\\cdot$ em", fontsize=10)
    axK.legend(fontsize=9); axK.tick_params(labelsize=9)
    axK.set_title("$\\kappa(s)$ along the first contour", fontsize=11,
                  color=PALETTE["ink"])
    for sp in ("top", "right"):
        axK.spines[sp].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, color=PALETTE["ink"], y=.98)
    fig.savefig(out_png, dpi=150, facecolor=PALETTE["paper"], bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# 6. CLI
# ==========================================================================
def run(font_path, char, out_dir, corner_deg=15.0, reg=0.04, node_reg=40.0,
        anchor=0.25, quiet=False):
    os.makedirs(out_dir, exist_ok=True)
    contours, upem, gname = extract_glyph(font_path, char)
    before = diagnose(contours, upem, corner_deg)

    fixed, info = solve_g2(contours, upem, corner_deg, reg, node_reg, anchor,
                           verbose=not quiet)
    after = diagnose(fixed, upem, corner_deg)

    dev = max_deviation(contours, fixed)
    sm_b = [r for r in before if r.kind == "smooth"]
    sm_a = [r for r in after if r.kind == "smooth"]
    curve_b = [r for r in sm_b if not r.line_adjacent]
    curve_a = [r for r in sm_a if not r.line_adjacent]

    def mx(rs):
        return max((r.jump * upem for r in rs), default=0.0)

    report = {
        "font": os.path.basename(font_path), "char": char, "glyph": gname, "upem": upem,
        "contours": len(contours),
        "nodes": len(before),
        "corners_G0": sum(r.kind == "corner" for r in before),
        "smooth_nodes": len(sm_b),
        "smooth_curve_curve": len(curve_b),
        "smooth_line_adjacent": len(sm_b) - len(curve_b),
        "max_kappa_jump_before_x_em": round(mx(sm_b), 4),
        "max_kappa_jump_after_x_em": round(mx(sm_a), 4),
        "max_jump_curve_curve_before_x_em": round(mx(curve_b), 4),
        "max_jump_curve_curve_after_x_em": round(mx(curve_a), 4),
        "max_outline_deviation_units": round(dev, 3),
        "max_outline_deviation_pct_em": round(100 * dev / upem, 4),
        "solver": info,
    }

    write_svg(os.path.join(out_dir, f"{char}_before.svg"), contours, upem, "#1B1E24")
    write_svg(os.path.join(out_dir, f"{char}_after.svg"), fixed, upem, "#1B1E24")
    with open(os.path.join(out_dir, f"{char}_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    render(contours, fixed, upem, before,
           os.path.join(out_dir, f"{char}_curvature.png"),
           title=f'{os.path.basename(font_path)}  "{char}"  -  curvature continuity')

    if not quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("\nkappa jump per smooth node (x em):")
        for b, a in zip(sm_b, sm_a):
            tag = "line-adjacent" if b.line_adjacent else "curve-curve  "
            print(f"  contour {b.contour} node {b.node:>3}  {tag}  "
                  f"{b.jump*upem:8.4f}  ->  {a.jump*upem:8.4f}")
    return report


def main():
    ap = argparse.ArgumentParser(
        description="Curvature-continuity analysis and repair for vector outlines")
    ap.add_argument("--font", required=True, help="path to a .ttf / .otf")
    ap.add_argument("--char", required=True, help="the character to analyse, e.g. S")
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--corner-deg", type=float, default=15.0,
                    help="tangent angles above this are treated as real corners and left "
                         "alone (default 15)")
    ap.add_argument("--reg", type=float, default=0.04,
                    help="regularisation strength: higher is more conservative, the glyph "
                         "moves less (default 0.04)")
    ap.add_argument("--node-reg", type=float, default=40.0,
                    help="extra penalty on node normal offsets relative to handle lengths: "
                         "higher means solve it with handles alone (default 40)")
    ap.add_argument("--anchor", type=float, default=0.25,
                    help="curvature-profile anchor: higher keeps the original kappa profile "
                         "and only removes the jumps (default 0.25)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    run(a.font, a.char, a.out, a.corner_deg, a.reg, a.node_reg, a.anchor, a.quiet)


if __name__ == "__main__":
    main()
