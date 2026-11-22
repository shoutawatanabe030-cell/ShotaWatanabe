# -*- coding: utf-8 -*-
"""
人物シルエットと地上の風景（逆光の人々が手を取り合い、波動を共鳴させるシーン用）
すべて手続き的に描画する。座標はフル解像度 1920x1080。
"""
import math
import numpy as np
import cv2

W, H = 1920, 1080
SH = 4            # cv2 サブピクセル (2^4)
K = 1 << SH


def P(x, y):
    return (int(round(x * K)), int(round(y * K)))


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


# ---------------------------------------------------------------- figure
class Person:
    """シルエットの人物。local座標は 足元=(0,0)、上向き正、身長=1"""

    def __init__(self, rng, child=False, style=None):
        self.child = child
        self.head_r = 0.075 if child else 0.06 + rng.random() * 0.006
        self.sh_w = (0.085 if child else 0.095 + rng.random() * 0.02)
        self.hip_w = 0.05 + rng.random() * 0.012
        self.leg_t = 0.105 + rng.random() * 0.02
        self.arm_t = 0.075 + rng.random() * 0.012
        styles = ["short", "short", "long", "bun", "pony", "afro", "scarf", "hat", "cap", "braid"]
        self.hair = style or styles[rng.integers(len(styles))]
        self.dress = (not child and rng.random() < 0.3) or (self.hair in ("scarf",) and rng.random() < 0.6)
        self.robe = self.hair == "scarf" and rng.random() < 0.5
        self.phase = rng.random() * 6.28
        self.bulk = 0.95 + rng.random() * 0.15

    def joints(self, pose, t, hand_l=None, hand_r=None):
        """pose: stand / hold / raise / walk / chest / reach / sway"""
        sw = math.sin(t * 1.3 + self.phase) * 0.006
        J = {}
        J["head"] = (sw, 0.915)
        J["neck"] = (sw * 0.7, 0.85)
        J["shl"] = (-self.sh_w + sw * 0.5, 0.80)
        J["shr"] = (self.sh_w + sw * 0.5, 0.80)
        J["hipl"] = (-self.hip_w, 0.5)
        J["hipr"] = (self.hip_w, 0.5)
        if pose == "walk":
            a = math.sin(t * 5.0 + self.phase)
            J["footl"] = (-0.03 + 0.14 * a, 0.02); J["footr"] = (0.03 - 0.14 * a, 0.02)
            J["kneel"] = (-0.03 + 0.09 * a + 0.02, 0.26); J["kneer"] = (0.03 - 0.09 * a + 0.02, 0.26)
        else:
            J["footl"] = (-0.085, 0.015); J["footr"] = (0.085, 0.015)
            J["kneel"] = (-0.075, 0.26); J["kneer"] = (0.075, 0.26)
        ua, fa = 0.19, 0.18
        # 手の目標
        if pose in ("stand", "sway"):
            tl = (-self.sh_w - 0.05, 0.47); tr = (self.sh_w + 0.05, 0.47)
        elif pose == "walk":
            a = math.sin(t * 5.0 + self.phase)
            tl = (-self.sh_w - 0.02 - 0.08 * a, 0.48); tr = (self.sh_w + 0.02 + 0.08 * a, 0.48)
        elif pose == "chest":
            tl = (-self.sh_w - 0.05, 0.47); tr = (-0.02, 0.70)
        elif pose == "reach":
            tl = (-self.sh_w - 0.05, 0.47); tr = (self.sh_w + 0.33, 0.95)
        elif pose == "raise":
            tl = (-self.sh_w - 0.16, 1.18); tr = (self.sh_w + 0.16, 1.18)
        else:  # hold
            tl = (-self.sh_w - 0.2, 0.5); tr = (self.sh_w + 0.2, 0.5)
        if hand_l is not None:
            tl = hand_l
        if hand_r is not None:
            tr = hand_r
        J["hl"], J["el"] = self.ik(J["shl"], tl, ua, fa, -1)
        J["hr"], J["er"] = self.ik(J["shr"], tr, ua, fa, 1)
        return J

    @staticmethod
    def ik(s, t, a, b, side):
        dx, dy = t[0] - s[0], t[1] - s[1]
        d = math.hypot(dx, dy)
        d2 = min(max(d, 1e-4), a + b - 1e-3)
        ang = math.atan2(dy, dx)
        c = (a * a + d2 * d2 - b * b) / (2 * a * d2)
        th = math.acos(max(-1, min(1, c)))
        # 肘は外側・下側に曲げる
        sgn = side if dy < 0.25 else -side
        e_ang = ang - sgn * th
        e = (s[0] + a * math.cos(e_ang), s[1] + a * math.sin(e_ang))
        h = (s[0] + d2 * math.cos(ang), s[1] + d2 * math.sin(ang))
        return h, e

    def draw(self, mask, J, tf, scale):
        """mask: uint8 画像。tf(local)->screen. scale: 身長 px"""
        def S(p):
            return tf(p)

        b = self.bulk
        def cap(a, c, r0, r1):
            capsule(mask, S(a), S(c), r0 * scale * b, r1 * scale * b)

        # legs（太ももから足首へ細く）
        for hp, kn, ft in (("hipl", "kneel", "footl"), ("hipr", "kneer", "footr")):
            cap(J[hp], J[kn], self.leg_t * 0.48, self.leg_t * 0.34)
            cap(J[kn], J[ft], self.leg_t * 0.34, self.leg_t * 0.22)
            fx, fy = J[ft]
            ell(mask, S((fx + (0.02 if ft == "footr" else -0.02), fy)), 0.05 * scale, 0.02 * scale)
        # torso（肩は丸く、腰で少し絞る）
        sl, sr, hl, hr = J["shl"], J["shr"], J["hipl"], J["hipr"]
        poly = [(sl[0], sl[1] + 0.025), (sl[0] + 0.03, sl[1] + 0.045), (sr[0] - 0.03, sr[1] + 0.045), (sr[0], sr[1] + 0.025),
                (sr[0] + 0.01 * b, sr[1] - 0.06), (sr[0] - 0.012 * b, 0.64), (hr[0] + 0.022 * b, 0.53), (hr[0] + 0.024 * b, 0.47),
                (hl[0] - 0.024 * b, 0.47), (hl[0] - 0.022 * b, 0.53), (sl[0] + 0.012 * b, 0.64), (sl[0] - 0.01 * b, sl[1] - 0.06)]
        fill(mask, [S(p) for p in poly])
        for sh in (sl, sr):
            circ(mask, S((sh[0], sh[1] - 0.008)), self.arm_t * 0.52 * scale * b)
        if self.dress or self.robe:
            hem = 0.05 if self.robe else 0.24
            wd = 0.17 if self.robe else 0.16
            fill(mask, [S(p) for p in [(hl[0] - 0.03, 0.58), (hr[0] + 0.03, 0.58), (wd, hem), (0.0, hem - 0.01), (-wd, hem)]])
        # neck + head
        cap(J["neck"], (J["neck"][0], 0.78), 0.024, 0.028)
        hx, hy = J["head"]; r = self.head_r
        ell(mask, S((hx, hy)), r * 0.88 * scale, r * 1.06 * scale)
        self.draw_hair(mask, S, scale, hx, hy, r)
        # arms
        for s_, e_, h_ in (("shl", "el", "hl"), ("shr", "er", "hr")):
            cap(J[s_], J[e_], self.arm_t * 0.55, self.arm_t * 0.42)
            cap(J[e_], J[h_], self.arm_t * 0.42, self.arm_t * 0.3)
            circ(mask, S(J[h_]), 0.026 * scale)

    def draw_hair(self, mask, S, sc, hx, hy, r):
        hs = self.hair
        R = r * sc
        if hs == "short":
            ell(mask, S((hx, hy + r * 0.2)), R * 0.98, R * 0.92)
        elif hs == "long":
            fill(mask, [S(p) for p in [(hx - r * 1.0, hy + r * 0.3), (hx + r * 1.0, hy + r * 0.3), (hx + r * 1.15, hy - r * 2.5), (hx - r * 1.15, hy - r * 2.5)]])
            ell(mask, S((hx, hy + r * 0.2)), R * 1.05, R * 1.02)
        elif hs == "bun":
            ell(mask, S((hx, hy + r * 0.2)), R * 0.98, R * 0.92)
            circ(mask, S((hx - r * 0.25, hy + r * 1.2)), R * 0.5)
        elif hs == "pony":
            ell(mask, S((hx, hy + r * 0.2)), R * 0.98, R * 0.92)
            ell(mask, S((hx - r * 1.15, hy - r * 0.5)), R * 0.32, R * 1.0, -25)
        elif hs == "braid":
            ell(mask, S((hx, hy + r * 0.2)), R * 0.98, R * 0.92)
            for k in range(4):
                circ(mask, S((hx + r * 0.95, hy - r * (0.4 + 0.5 * k))), R * 0.26)
        elif hs == "afro":
            circ(mask, S((hx, hy + r * 0.3)), R * 1.4)
        elif hs == "scarf":
            fill(mask, [S(p) for p in [(hx - r * 1.15, hy + r * 0.2), (hx + r * 1.15, hy + r * 0.2), (hx + r * 1.9, hy - r * 2.2), (hx - r * 1.9, hy - r * 2.2)]])
            ell(mask, S((hx, hy + r * 0.22)), R * 1.15, R * 1.12)
        elif hs == "hat":
            ell(mask, S((hx, hy + r * 0.72)), R * 2.2, R * 0.26)
            ell(mask, S((hx, hy + r * 0.95)), R * 0.9, R * 0.72)
        elif hs == "cap":
            ell(mask, S((hx, hy + r * 0.3)), R * 1.0, R * 0.85)
            ell(mask, S((hx + r * 0.85, hy + r * 0.45)), R * 0.75, R * 0.17)


def capsule(mask, p0, p1, r0, r1):
    """先細りのカプセル（画面座標px）"""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 1e-3:
        circ(mask, p0, max(r0, r1)); return
    nx, ny = -dy / L, dx / L
    poly = [(x0 + nx * r0, y0 + ny * r0), (x1 + nx * r1, y1 + ny * r1), (x1 - nx * r1, y1 - ny * r1), (x0 - nx * r0, y0 - ny * r0)]
    fill(mask, poly)
    circ(mask, p0, r0); circ(mask, p1, r1)


def circ(mask, c, r, val=255):
    cv2.circle(mask, P(*c), max(1, int(round(r * K))), val, -1, cv2.LINE_AA, SH)


def ell(mask, c, rx, ry, ang=0, val=255):
    cv2.ellipse(mask, P(*c), (max(1, int(round(rx * K))), max(1, int(round(ry * K)))), ang, 0, 360, val, -1, cv2.LINE_AA, SH)


def fill(mask, pts, val=255):
    cv2.fillPoly(mask, [np.array([P(x, y) for x, y in pts], np.int32)], val, cv2.LINE_AA, SH)


def tf_upright(x0, y0, s):
    return lambda p: (x0 + p[0] * s, y0 - p[1] * s)


def tf_rot(x0, y0, s, ang):
    """ang: 人物の上方向（画面座標でのラジアン, 0=上）"""
    c, si = math.cos(ang), math.sin(ang)
    return lambda p: (x0 + (p[0] * c + p[1] * si) * s, y0 - (p[1] * c - p[0] * si) * s)


# ---------------------------------------------------------------- crowd helpers
def line_of_people(rng, n, x0, x1, ground, height, jitter=0.08, children=0.2):
    ppl = []
    xs = np.linspace(x0, x1, n)
    for i in range(n):
        child = rng.random() < children
        p = Person(rng, child=child)
        h = height * (0.62 if child else (0.92 + rng.random() * 0.16))
        ppl.append(dict(p=p, x=float(xs[i] + rng.normal(0, jitter * (x1 - x0) / n)), y=ground, h=h))
    return ppl


def draw_chain(mask, ppl, t, pose="hold", raise_amt=0.0):
    """隣同士で手をつないだ人の列を描く。各人の手の位置(画面)を返す"""
    hands = []
    n = len(ppl)
    for i, q in enumerate(ppl):
        p, x, y, h = q["p"], q["x"], q["y"], q["h"]
        tf = tf_upright(x, y, h)
        inv = lambda X, Y: ((X - x) / h, (y - Y) / h)
        hl = hr = None
        hy = 0.5 + 0.68 * raise_amt
        if i > 0:
            o = ppl[i - 1]
            mx = (o["x"] + x) / 2; my = y - (o["h"] + h) / 2 * hy
            hl = inv(mx, my)
        if i < n - 1:
            o = ppl[i + 1]
            mx = (o["x"] + x) / 2; my = y - (o["h"] + h) / 2 * hy
            hr = inv(mx, my)
        base = "raise" if raise_amt > 0.5 else ("stand" if pose == "hold" else pose)
        J = p.joints(base, t, hl, hr)
        p.draw(mask, J, tf, h)
        hands.append([tf(J["hl"]), tf(J["hr"])])
    return hands


# ---------------------------------------------------------------- environments
def fbm1d(n, seed, octaves=6, rough=0.55):
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    amp = 1.0
    for o in range(octaves):
        k = 2 ** o * 3
        pts = rng.random(k + 3)
        xs = np.linspace(0, k, n)
        out += amp * np.interp(xs, np.arange(k + 3), pts)
        amp *= rough
    out -= out.min(); out /= out.max()
    return out


def sky_gradient(top, mid, bottom, horizon_y, h=H, w=W):
    y = np.arange(h, dtype=np.float32)[:, None]
    u = np.clip(y / max(horizon_y, 1), 0, 1)
    top, mid, bottom = [np.array(c, np.float32) for c in (top, mid, bottom)]
    c1 = top * (1 - u[..., None]) + mid * u[..., None]
    v = np.clip((y - horizon_y * 0.6) / (horizon_y * 0.4 + 1), 0, 1)
    img = c1 * (1 - v[..., None] ** 2) + bottom * (v[..., None] ** 2)
    return np.broadcast_to(img, (h, w, 3)).astype(np.float32).copy()


def stars_layer(rng, n=2600, band=True, h=H, w=W):
    img = np.zeros((h, w, 3), np.float32)
    xs = rng.random(n) * w; ys = rng.random(n) * h
    if band:
        m = n // 2
        xs[:m] = rng.random(m) * w
        ys[:m] = h * 0.15 + xs[:m] * 0.35 + rng.normal(0, h * 0.07, m)
    b = rng.pareto(2.5, n) * 0.25 + 0.1
    for x, y, bb in zip(xs, ys, b):
        if 0 <= x < w - 1 and 0 <= y < h - 1:
            img[int(y), int(x)] += min(bb, 2.0)
    img = cv2.GaussianBlur(img, (0, 0), 0.7) * 2.2
    if band:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = (yy - (h * 0.15 + xx * 0.35)) / (h * 0.11)
        glow = np.exp(-d * d)[..., None] * np.array([0.07, 0.065, 0.1], np.float32)
        img += glow
    return img


def radial_glow(cx, cy, r, color, h=H, w=W, power=1.6):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    g = 1.0 / (1 + d ** 2) ** power
    return g[..., None] * np.array(color, np.float32)


def ridge_mask(base_y, amp, seed, w=W, h=H, octaves=6, rough=0.55):
    prof = base_y - amp * fbm1d(w, seed, octaves, rough)
    yy = np.arange(h, dtype=np.float32)[:, None]
    m = np.clip(yy - prof[None, :] + 0.5, 0, 1)
    return m.astype(np.float32), prof


def city_skyline(rng, base_y, w=W, h=H, max_h=380):
    mask = np.zeros((h, w), np.float32)
    win = np.zeros((h, w, 3), np.float32)
    x = -20
    while x < w + 20:
        bw = int(rng.integers(40, 140)); bh = int(rng.integers(60, max_h))
        if rng.random() < 0.12:
            bh = int(bh * 1.6)
        top = base_y - bh
        cv2.rectangle(mask, (x, int(top)), (x + bw, h), 1.0, -1)
        if rng.random() < 0.25:
            cv2.line(mask, (x + bw // 2, int(top)), (x + bw // 2, int(top - rng.integers(20, 70))), 1.0, 2)
        # 窓
        for wy in range(int(top) + 10, base_y + 200, 14):
            for wx in range(x + 6, x + bw - 6, 11):
                if rng.random() < 0.33:
                    c = np.array([1.0, 0.78, 0.45] if rng.random() < 0.8 else [0.7, 0.85, 1.0], np.float32)
                    win[wy:wy + 6, wx:wx + 5] = c * (0.4 + rng.random() * 0.6)
        x += bw + int(rng.integers(0, 10))
    win *= mask[..., None]
    return mask, win


def acacia(mask, x, y, s):
    cv2.line(mask, P(x, y), P(x + 0.05 * s, y - 0.55 * s), 1.0, max(2, int(0.05 * s)), cv2.LINE_AA, SH)
    cv2.line(mask, P(x + 0.05 * s, y - 0.5 * s), P(x - 0.25 * s, y - 0.75 * s), 1.0, max(2, int(0.03 * s)), cv2.LINE_AA, SH)
    cv2.line(mask, P(x + 0.05 * s, y - 0.5 * s), P(x + 0.32 * s, y - 0.78 * s), 1.0, max(2, int(0.03 * s)), cv2.LINE_AA, SH)
    ell(mask, (x + 0.03 * s, y - 0.82 * s), 0.55 * s, 0.11 * s, 0, 1.0)


# ---------------------------------------------------------------- compositing helpers
def rim_light(m, ldir, k=3.0, blur=1.2):
    """m: 0..1 マスク。ldir: 光源方向(画面上の単位ベクトル)"""
    M = np.float32([[1, 0, -ldir[0] * k], [0, 1, -ldir[1] * k]])
    # m_up(p) = m(p + k*ldir)
    sh = cv2.warpAffine(m, np.float32([[1, 0, -ldir[0] * k], [0, 1, -ldir[1] * k]]), (m.shape[1], m.shape[0]))
    rim = np.clip(m - sh, 0, 1)
    if blur > 0:
        rim = cv2.GaussianBlur(rim, (0, 0), blur)
    return rim


def composite_silhouette(img, m, rimcol, ldir, silcol=(0.012, 0.014, 0.024), rimk=3.0, rim_strength=1.0):
    # 背景が明るいところの縁だけ光らせる
    lum = cv2.GaussianBlur(img.mean(2), (0, 0), 6)
    rimw = np.clip(lum * 4.0, 0, 1)
    m3 = m[..., None]
    img *= (1 - m3)
    img += m3 * np.array(silcol, np.float32)
    rim = rim_light(m, ldir, rimk) * rimw
    img += rim[..., None] * np.array(rimcol, np.float32) * rim_strength
    return img


def energy_polyline(glow, pts, color, width, t, pulse_speed=1.0, pulses=True):
    """手と手を結ぶ光の線。glow: float画像"""
    if len(pts) < 2:
        return
    canvas = np.zeros(glow.shape[:2], np.uint8)
    arr = np.array([P(x, y) for x, y in pts], np.int32)
    cv2.polylines(canvas, [arr], False, 255, width, cv2.LINE_AA, SH)
    glow += (canvas.astype(np.float32) / 255.0)[..., None] * np.array(color, np.float32)
    if pulses:
        seg = np.array(pts, np.float32)
        L = np.r_[0, np.cumsum(np.hypot(*np.diff(seg, axis=0).T))]
        tot = L[-1]
        for k in range(6):
            s = ((t * pulse_speed * 260 + k * tot / 6) % tot)
            j = min(np.searchsorted(L, s) - 1, len(seg) - 2); j = max(j, 0)
            u = (s - L[j]) / max(L[j + 1] - L[j], 1e-3)
            x, y = seg[j] * (1 - u) + seg[j + 1] * u
            glow_canvas = np.zeros(glow.shape[:2], np.uint8)
            cv2.circle(glow_canvas, P(x, y), int(width * 3 * K), 255, -1, cv2.LINE_AA, SH)
            glow += (glow_canvas.astype(np.float32) / 255.0)[..., None] * np.array(color, np.float32) * 1.5


def ground_rings(glow, cx, cy, rx, ry, t, beats, color, life=2.4, maxr=1.6):
    canvas = np.zeros(glow.shape[:2], np.float32)
    for b in beats:
        age = t - b
        if 0 <= age < life:
            s = 0.15 + maxr * age / life
            a = (1 - age / life) ** 1.5
            cv2.ellipse(canvas, P(cx, cy), (int(rx * s * K), int(ry * s * K)), 0, 0, 360, a, 3, cv2.LINE_AA, SH)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 1.5) * 2.0
    glow += canvas[..., None] * np.array(color, np.float32)


def rising_particles(glow, rng_seed, t, n, x0, x1, y0, y1, color, speed=60, size=2):
    rng = np.random.default_rng(rng_seed)
    xs = rng.random(n) * (x1 - x0) + x0
    ph = rng.random(n)
    sp = speed * (0.6 + rng.random(n) * 0.8)
    span = (y1 - y0)
    ys = y1 - ((ph * span + t * sp) % span)
    xs = xs + np.sin(t * 0.8 + ph * 20) * 12
    tw = 0.5 + 0.5 * np.sin(t * 3 + ph * 30)
    fade = np.clip((ys - y0) / (span * 0.3), 0, 1)
    canvas = np.zeros(glow.shape[:2], np.float32)
    for x, y, a in zip(xs, ys, tw * fade):
        cv2.circle(canvas, P(x, y), int(size * K), float(a), -1, cv2.LINE_AA, SH)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 1.2) * 1.6
    glow += canvas[..., None] * np.array(color, np.float32)
