"""
squircle.py -- reproducing Apple's continuous corner (the "squircle") from closed form

Three corner models
-------------------
1. circular      Plain border-radius: straight edge + quarter arc. kappa jumps 0 -> 1/r
                 at the seam (G1 only).
2. superellipse  Lame curve |x/A|^n + |y/A|^n = 1. kappa is continuous everywhere, but the
                 corner keeps no straight edge.
3. continuous    What Apple / Figma actually use: cubic Bezier -> arc -> cubic Bezier,
                 parameterised by a smoothing factor s in [0, 1]
                 (reverse-engineered by Daniel Furse, Figma 2018).

Closed form (corner radius r, smoothing s, beta = 45 deg * s)
-------------------------------------------------------------
    p = (1 + s) * r                    length the corner consumes along each edge
    arc = the inscribed circle: centre (r, r), radius r, swept from polar angle
          beta to 90 deg - beta
    endpoint  A0 = ( r(1 - cos beta),  r(1 - sin beta) )    <- verified numerically below
    l = r(cos beta - sin beta)         arc span in x and in y
    d = r(1 - cos beta)                lateral offset of the Bezier endpoint
    c = r * tan(beta/2) * cos beta     (c, d) is the tangent handle: length r*tan(beta/2),
                                       direction beta
    b = [ r(s + sin beta) - r*tan(beta/2)*cos beta ] / 3 ,   a = 2b

    identity   (a + b + c) + l + d = p     (the corner is exactly symmetric in x and y)

Curvature at the Bezier endpoint (closed form, verified below):
    kappa_end * r = 4*cos^3(beta/2) * [ s + sin beta - tan(beta/2)*cos beta ]
                    / ( 9*sin(beta/2) )
    This is never 1, so the construction is G1, not truly G2 -- kappa still jumps at the
    Bezier/arc seam, just far less than the plain corner's 0 -> 1/r.

Apple measures p = 1.528665 r, i.e. s = 0.528665. The "60%" preset in Figma's UI is an
approximation of it.

Run:  python3 squircle.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

APPLE_P_OVER_R = 1.528665                    # reverse-engineered from UIBezierPath
APPLE_SMOOTHING = APPLE_P_OVER_R - 1.0       # ~= 0.528665
FIGMA_IOS_SMOOTHING = 0.6


# ==========================================================================
# 1. Parametric construction
# ==========================================================================
@dataclass(frozen=True)
class CornerParams:
    r: float
    s: float
    p: float            # (1+s)*r
    arc_len: float      # l
    a: float
    b: float
    c: float
    d: float
    arc_measure: float  # arc angle in degrees = 90(1-s)
    beta: float         # radians


def corner_params(r: float, s: float, budget: float | None = None) -> CornerParams:
    """Solve every quantity of one corner from (r, s).

    budget: the most this corner may consume along one edge (usually min(w, h) / 2).
            When exceeded, the radius is clamped first, then the smoothing -- matching
            how Apple and Figma degrade.
    """
    s = float(min(max(s, 0.0), 1.0))
    if budget is not None and budget > 0 and (1.0 + s) * r > budget:
        r = min(r, budget)
        s = max(budget / r - 1.0, 0.0)

    beta = math.radians(45.0 * s)
    p = (1.0 + s) * r
    arc_len = r * (math.cos(beta) - math.sin(beta))
    d = r * (1.0 - math.cos(beta))
    c = r * math.tan(beta / 2.0) * math.cos(beta)
    b = (r * (s + math.sin(beta)) - r * math.tan(beta / 2.0) * math.cos(beta)) / 3.0
    return CornerParams(r, s, p, arc_len, 2.0 * b, b, c, d, 90.0 * (1.0 - s), beta)


def curvature_ratio(s: float) -> float:
    """Bezier end curvature divided by the arc's 1/r. Only 1.0 would be true G2."""
    if s <= 0:
        return 1.0
    beta = math.radians(45.0 * s)
    return (4.0 * math.cos(beta / 2) ** 3
            * (s + math.sin(beta) - math.tan(beta / 2) * math.cos(beta))
            / (9.0 * math.sin(beta / 2)))


def svg_path(w: float, h: float, r: float, s: float = APPLE_SMOOTHING) -> str:
    """The whole continuous-curvature rounded rectangle as an SVG path (the `d` attribute)."""
    q = corner_params(r, s, budget=min(w, h) / 2.0)
    a, b, c, d, p, L, R = q.a, q.b, q.c, q.d, q.p, q.arc_len, q.r
    e = a + b + c

    def f(x: float) -> str:
        t = f"{x:.4f}".rstrip("0").rstrip(".")
        return "0" if t in ("", "-0") else t

    return " ".join([
        f"M {f(w-p)} 0",
        f"c {f(a)} 0 {f(a+b)} 0 {f(e)} {f(d)}",
        f"a {f(R)} {f(R)} 0 0 1 {f(L)} {f(L)}",
        f"c {f(d)} {f(c)} {f(d)} {f(b+c)} {f(d)} {f(e)}",
        f"L {f(w)} {f(h-p)}",
        f"c 0 {f(a)} 0 {f(a+b)} {f(-d)} {f(e)}",
        f"a {f(R)} {f(R)} 0 0 1 {f(-L)} {f(L)}",
        f"c {f(-c)} {f(d)} {f(-(b+c))} {f(d)} {f(-e)} {f(d)}",
        f"L {f(p)} {f(h)}",
        f"c {f(-a)} 0 {f(-(a+b))} 0 {f(-e)} {f(-d)}",
        f"a {f(R)} {f(R)} 0 0 1 {f(-L)} {f(-L)}",
        f"c {f(-d)} {f(-c)} {f(-d)} {f(-(b+c))} {f(-d)} {f(-e)}",
        f"L 0 {f(p)}",
        f"c 0 {f(-a)} 0 {f(-(a+b))} {f(d)} {f(-e)}",
        f"a {f(R)} {f(R)} 0 0 1 {f(L)} {f(-L)}",
        f"c {f(c)} {f(-d)} {f(b+c)} {f(-d)} {f(e)} {f(-d)}",
        "Z",
    ])


# ==========================================================================
# 2. Sampling and curvature
# ==========================================================================
def _bezier(P, t):
    P = np.asarray(P, float)
    t = np.asarray(t, float)[:, None]
    mt = 1 - t
    B = mt**3 * P[0] + 3 * mt**2 * t * P[1] + 3 * mt * t**2 * P[2] + t**3 * P[3]
    d1 = 3 * (mt**2 * (P[1] - P[0]) + 2 * mt * t * (P[2] - P[1]) + t**2 * (P[3] - P[2]))
    d2 = 6 * (mt * (P[2] - 2 * P[1] + P[0]) + t * (P[3] - 2 * P[2] + P[1]))
    return B, d1, d2


def _kappa(d1, d2):
    """kappa = |x'y'' - y'x''| / (x'^2 + y'^2)^{3/2}  -- invariant under reparameterisation"""
    num = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5
    return np.abs(num) / np.maximum(den, 1e-15)


def continuous_corner(r: float, s: float = APPLE_SMOOTHING, n: int = 400):
    """One continuous corner: from (0, p) round to (p, 0), corner point at the origin.

    Returns (points, kappa, params).
    """
    q = corner_params(r, s)
    a, b, c, d, p, L = q.a, q.b, q.c, q.d, q.p, q.arc_len
    e = a + b + c

    P1 = [(0.0, p), (0.0, p - a), (0.0, p - (a + b)), (d, p - e)]
    P3 = [(y, x) for (x, y) in P1][::-1]          # mirror across y = x

    t = np.linspace(0, 1, n)
    B1, d1a, d2a = _bezier(P1, t)
    B3, d1b, d2b = _bezier(P3, t)

    # The arc is part of the inscribed circle (centre (r, r), radius r),
    # from polar angle beta to 90 deg - beta.
    m = max(n // 2, 16)
    ang = np.linspace(math.pi + q.beta, 1.5 * math.pi - q.beta, m)
    Barc = np.stack([r + r * np.cos(ang), r + r * np.sin(ang)], 1)

    pts = np.concatenate([B1, Barc[1:-1], B3], 0)
    kap = np.concatenate([_kappa(d1a, d2a), np.full(m - 2, 1.0 / r), _kappa(d1b, d2b)])
    return pts, kap, q


def circular_corner(r: float, n: int = 400):
    """Plain quarter-circle corner: (0, r) -> (r, 0), centre (r, r)."""
    ang = np.linspace(math.pi, 1.5 * math.pi, n)
    return np.stack([r + r * np.cos(ang), r + r * np.sin(ang)], 1), np.full(n, 1.0 / r)


def superellipse_corner(A: float, n_exp: float, n: int = 600):
    """Superellipse quarter corner inside [0, A]^2, from (0, A) to (A, 0)."""
    t = np.linspace(1e-6, math.pi / 2 - 1e-6, n)
    x = A * np.cos(t) ** (2.0 / n_exp)
    y = A * np.sin(t) ** (2.0 / n_exp)
    pts = np.stack([A - x, A - y], 1)[::-1]
    d1 = np.gradient(pts, axis=0)
    d2 = np.gradient(d1, axis=0)
    return pts, _kappa(d1, d2)


# ==========================================================================
# 3. Fit: the best superellipse that consumes the same edge length p
# ==========================================================================
def fit_superellipse_exponent(r: float = 1.0, s: float = APPLE_SMOOTHING):
    """Semi-axis fixed at A = p; only the exponent n is fitted.

    Returns (n, A, rms) with rms in units of r.
    """
    from scipy.optimize import minimize_scalar

    target, _, q = continuous_corner(r, s, n=400)
    A = q.p

    def cost(n_exp):
        if n_exp <= 2.0:
            return 1e6
        cand, _ = superellipse_corner(A, n_exp, 2000)
        dmin = np.sqrt(((target[:, None, :] - cand[None, :, :]) ** 2).sum(-1)).min(1)
        return float(np.sqrt((dmin ** 2).mean()))

    res = minimize_scalar(cost, bounds=(2.01, 30.0), method="bounded",
                          options={"xatol": 1e-5})
    return float(res.x), float(A), float(res.fun)


# ==========================================================================
# 4. Demo
# ==========================================================================
def rounded_rect_points(w: float, h: float, r: float, s: float, n: int = 160):
    """The whole rounded rectangle as a closed polyline, ready to draw."""
    q = corner_params(r, s, budget=min(w, h) / 2)
    cp, _, _ = continuous_corner(q.r, q.s, n)      # (0,p) -> (p,0), top-left corner
    x, y = cp[:, 0], cp[:, 1]
    tl = np.stack([x, y], 1)
    tr = np.stack([w - x, y], 1)[::-1]
    br = np.stack([w - x, h - y], 1)
    bl = np.stack([x, h - y], 1)[::-1]
    return np.concatenate([tl, tr, br, bl], 0)


def _demo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    r, s = 1.0, APPLE_SMOOTHING
    cont, kc, q = continuous_corner(r, s, 500)
    circ, kk = circular_corner(r, 500)
    n_exp, A, rms = fit_superellipse_exponent(r, s)
    se, kse = superellipse_corner(A, n_exp, 500)

    C = {"circ": "#C0453A", "se": "#2E7D6F", "cont": "#2B5FA8"}
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

    # (1) Outlines plus the Bezier control polygon
    ax = axes[0]
    e = q.a + q.b + q.c
    ctrl = np.array([(0.0, q.p), (0.0, q.p - q.a), (0.0, q.p - (q.a + q.b)), (q.d, q.p - e)])
    ctrl = np.concatenate([ctrl, ctrl[::-1][:, ::-1]], 0)
    ax.plot([-0.25, 0], [q.p, q.p], color="0.8", lw=1)
    ax.plot([0, 0], [q.p, q.p + 0.25], color="0.8", lw=1)
    ax.plot(ctrl[:, 0], ctrl[:, 1], color="0.72", lw=1, ls="-", marker="o", ms=3,
            zorder=1, label="Bezier control polygon")
    ax.plot(circ[:, 0], circ[:, 1], lw=2.0, color=C["circ"], label=f"Circular, r = {r:g}")
    ax.plot(se[:, 0], se[:, 1], lw=2.0, ls="--", color=C["se"],
            label=f"Superellipse  n = {n_exp:.2f}, A = {A:.3f}r")
    ax.plot(cont[:, 0], cont[:, 1], lw=2.6, color=C["cont"],
            label=f"Apple continuous  s = {s:.4f}")
    ax.scatter([q.d, q.p - e], [q.p - e, q.d], s=34, zorder=5, color=C["cont"],
               label="Bezier/arc seam  A0, A1")
    ax.scatter([r], [r], s=30, color="0.45", marker="+", zorder=5)
    ax.annotate("centre (r, r)", (r, r), textcoords="offset points", xytext=(8, -2),
                fontsize=8.5, color="0.4")
    ax.set_aspect("equal"); ax.set_xlim(-0.16, 1.72); ax.set_ylim(1.72, -0.16)
    ax.set_title("Three corners at the same cornerRadius r", fontsize=12)
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.95)

    # (2) Curvature
    ax = axes[1]
    for name, pts, kap, col in [("Circular", circ, kk, C["circ"]),
                                (f"Superellipse n = {n_exp:.2f}", se, kse, C["se"]),
                                ("Apple continuous", cont, kc, C["cont"])]:
        sl = np.concatenate([[0.0], np.cumsum(np.sqrt((np.diff(pts, axis=0) ** 2).sum(1)))])
        pad = 0.4
        ax.plot(np.concatenate([[-pad, 0.0], sl, [sl[-1], sl[-1] + pad]]),
                np.concatenate([[0.0, 0.0], kap, [0.0, 0.0]]), lw=2.2, color=col, label=name)
    ax.axhline(1.0 / r, color="0.8", lw=1, ls=":")
    ax.set_xlabel("arc length s"); ax.set_ylabel("curvature $\\kappa$")
    ax.set_ylim(-0.08, 1.85); ax.legend(fontsize=9, loc="upper right")
    ax.set_title("$\\kappa(s)$ -- the plain corner jumps straight from 0 to 1/r",
                 fontsize=12)
    ax.annotate(f"residual jump x{curvature_ratio(s):.3f}", (0.93, curvature_ratio(s)),
                textcoords="offset points", xytext=(-108, 6), fontsize=8.5,
                color=C["cont"],
                arrowprops=dict(arrowstyle="->", color=C["cont"], lw=0.9))

    # (3) How it actually looks
    ax = axes[2]
    W, R = 168.0, 46.0
    for i, (s_, lab, col) in enumerate([(0.0, "s = 0  (plain round)", C["circ"]),
                                        (APPLE_SMOOTHING,
                                         f"s = {APPLE_SMOOTHING:.3f}  (Apple)", C["cont"]),
                                        (1.0, "s = 1  (fully smoothed)", C["se"])]):
        poly = rounded_rect_points(W, W, R, s_, 200)
        poly[:, 0] += i * (W + 22)
        ax.add_patch(mpatches.Polygon(poly, closed=True, facecolor=col, edgecolor="none"))
        ax.text(i * (W + 22) + W / 2, W + 22, lab, ha="center", fontsize=10)
    ax.set_xlim(-12, 3 * W + 56); ax.set_ylim(W + 40, -14)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"Same r = {R:.0f}px, different smoothing", fontsize=12)

    plt.tight_layout()
    plt.savefig("squircle_comparison.png", dpi=150)

    # Console report
    print(f"Apple smoothing s              = {APPLE_SMOOTHING:.6f}   (p/r = {APPLE_P_OVER_R})")
    print(f"beta = 45 deg * s              = {math.degrees(q.beta):.4f} deg")
    print(f"arc measure 90(1-s)            = {q.arc_measure:.4f} deg")
    print(f"a, b, c, d, l  (x r)           = {q.a:.6f} {q.b:.6f} {q.c:.6f} {q.d:.6f} "
          f"{q.arc_len:.6f}")
    print(f"identity (a+b+c)+l+d-p         = {(q.a+q.b+q.c)+q.arc_len+q.d-q.p:.2e}")
    print(f"arc endpoint A0                = ({q.d:.6f}, {q.arc_len+q.d:.6f})"
          f"   closed form (r(1-cos b), r(1-sin b)) = "
          f"({r*(1-math.cos(q.beta)):.6f}, {r*(1-math.sin(q.beta)):.6f})")
    print(f"kappa_bezier_end / (1/r)       = {curvature_ratio(s):.6f}"
          f"   -> G1, not true G2")
    print(f"best equal-width superellipse  n = {n_exp:.4f}, A = {A:.4f}r, "
          f"rms = {rms:.5f}r  ({rms/q.p*100:.2f}% of p)")
    print()
    print("SVG path (200x200, r = 48, Apple smoothing):")
    print(svg_path(200, 200, 48))


if __name__ == "__main__":
    _demo()
