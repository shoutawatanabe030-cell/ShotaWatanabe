# -*- coding: utf-8 -*-
"""地上のシーン（人々）。各関数は線形RGBの float 画像 (H,W,3) を返す"""
import math
import numpy as np
import cv2
from people import (Person, W, H, P as PT, K, SH, smooth, tf_upright, tf_rot, line_of_people, draw_chain,
                    sky_gradient, stars_layer, radial_glow, ridge_mask, city_skyline, acacia, composite_silhouette,
                    energy_polyline, ground_rings, rising_particles, capsule, circ, ell, fill, fbm1d)

_cache = {}
BEAT = 60 / 128.5714


def cached(key, fn):
    if key not in _cache:
        if len(_cache) > 24:
            _cache.clear()
        _cache[key] = fn()
    return _cache[key]


def hue_rgb(h, s=0.7, v=1.0):
    import colorsys
    return np.array(colorsys.hsv_to_rgb(h % 1.0, s, v), np.float32)


# ---------------------------------------------------------------- environments
def env(name, seed):
    def build():
        rng = np.random.default_rng(seed)
        e = dict(ldir=(0.0, -1.0), rim=(1.0, 0.75, 0.45))
        if name == "dusk_hill":
            bg = sky_gradient((0.04, 0.05, 0.16), (0.5, 0.28, 0.36), (1.0, 0.6, 0.3), 740)
            bg += radial_glow(960, 740, 300, (1.0, 0.68, 0.32), power=1.2) * 0.9
            far, _ = ridge_mask(745, 70, seed + 1, rough=0.5)
            bg = bg * (1 - far[..., None] * 0.65) + far[..., None] * np.array([0.12, 0.07, 0.12], np.float32)
            g, prof = ridge_mask(800, 45, seed + 2)
            e.update(bg=bg, ground=g, gcol=(0.012, 0.012, 0.025), gy=800, prof=prof)
        elif name == "night_hill":
            bg = sky_gradient((0.01, 0.015, 0.05), (0.05, 0.08, 0.2), (0.22, 0.35, 0.55), 790) + stars_layer(rng)
            bg += radial_glow(960, 820, 500, (0.25, 0.35, 0.6), power=1.0) * 0.35
            g, prof = ridge_mask(820, 40, seed + 2)
            e.update(bg=bg, ground=g, gcol=(0.008, 0.008, 0.018), gy=820, prof=prof, rim=(0.55, 0.75, 1.0))
        elif name == "sunrise_ridge":
            bg = sky_gradient((0.18, 0.28, 0.5), (0.95, 0.62, 0.42), (1.2, 0.85, 0.5), 700)
            bg += radial_glow(1150, 700, 180, (1.4, 1.0, 0.6), power=1.5) * 1.2
            for k, (y, a, c) in enumerate([(700, 120, (0.55, 0.45, 0.55)), (730, 90, (0.32, 0.25, 0.35)), (760, 70, (0.16, 0.12, 0.2))]):
                m, _ = ridge_mask(y, a, seed + 10 + k, rough=0.5)
                bg = bg * (1 - m[..., None]) + m[..., None] * np.array(c, np.float32)
            g, prof = ridge_mask(810, 30, seed + 2, octaves=4)
            e.update(bg=bg, ground=g, gcol=(0.02, 0.015, 0.02), gy=810, prof=prof, rim=(1.2, 0.85, 0.5))
        elif name == "city_night":
            bg = sky_gradient((0.02, 0.025, 0.07), (0.12, 0.1, 0.22), (0.45, 0.28, 0.3), 700) + stars_layer(rng, 800, band=False) * 0.5
            m, win = city_skyline(rng, 760)
            bg = bg * (1 - m[..., None]) + m[..., None] * np.array([0.03, 0.03, 0.06], np.float32) + win * 0.9
            bg += radial_glow(960, 780, 700, (0.5, 0.3, 0.25), power=0.8) * 0.15
            g = np.zeros((H, W), np.float32); g[800:] = 1
            e.update(bg=bg, ground=g, gcol=(0.01, 0.01, 0.02), gy=800, prof=np.full(W, 800.0), rim=(1.0, 0.7, 0.5))
        elif name == "desert_dusk":
            bg = sky_gradient((0.12, 0.08, 0.25), (0.85, 0.42, 0.3), (1.1, 0.65, 0.3), 690)
            bg += radial_glow(700, 700, 220, (1.2, 0.7, 0.3), power=1.3) * 0.8
            for k, (y, a, c) in enumerate([(700, 50, (0.45, 0.22, 0.2)), (740, 60, (0.25, 0.12, 0.13))]):
                m, _ = ridge_mask(y, a, seed + 20 + k, octaves=3, rough=0.4)
                bg = bg * (1 - m[..., None]) + m[..., None] * np.array(c, np.float32)
            g, prof = ridge_mask(800, 70, seed + 2, octaves=3, rough=0.4)
            e.update(bg=bg, ground=g, gcol=(0.03, 0.015, 0.02), gy=800, prof=prof, rim=(1.2, 0.7, 0.4))
        elif name == "beach_dawn":
            bg = sky_gradient((0.15, 0.22, 0.42), (0.85, 0.6, 0.55), (1.15, 0.8, 0.55), 640)
            bg += radial_glow(1250, 630, 160, (1.5, 1.05, 0.6), power=1.5) * 1.0
            sea = np.zeros((H, W), np.float32); sea[640:] = 1
            yy = np.arange(H, dtype=np.float32)[:, None]
            seac = np.array([0.25, 0.28, 0.42], np.float32) * (1 - np.clip((yy - 640) / 300, 0, 1))[..., None] + np.array([0.08, 0.1, 0.18], np.float32)
            bg = bg * (1 - sea[..., None]) + sea[..., None] * seac
            g = np.zeros((H, W), np.float32); g[830:] = 1
            g = cv2.GaussianBlur(g, (0, 0), 2)
            e.update(bg=bg, ground=g, gcol=(0.03, 0.025, 0.035), gy=830, prof=np.full(W, 830.0), rim=(1.3, 0.9, 0.6), sea=True, sun=(1250, 630))
        elif name == "aurora_snow":
            bg = sky_gradient((0.01, 0.02, 0.06), (0.03, 0.08, 0.16), (0.15, 0.25, 0.35), 760) + stars_layer(rng, 1800)
            g, prof = ridge_mask(800, 25, seed + 2, octaves=4)
            e.update(bg=bg, ground=g, gcol=(0.16, 0.2, 0.28), gy=800, prof=prof, rim=(0.5, 1.0, 0.8), aurora=True)
        elif name == "window_night":
            bg = sky_gradient((0.02, 0.025, 0.06), (0.06, 0.06, 0.14), (0.2, 0.14, 0.2), 900)
            bok = np.zeros((H, W, 3), np.float32)
            for _ in range(160):
                x, y = rng.random() * W, 350 + rng.random() * 600
                c = np.array([1.0, 0.7, 0.4] if rng.random() < 0.7 else [0.5, 0.7, 1.0], np.float32)
                cv2.circle(bok, (int(x), int(y)), int(8 + rng.random() * 26), tuple(float(v) for v in c * (0.15 + rng.random() * 0.35)), -1, cv2.LINE_AA)
            bg += cv2.GaussianBlur(bok, (0, 0), 4)
            frame = np.ones((H, W), np.float32)
            cv2.rectangle(frame, (330, 90), (1590, 1080), 0.0, -1)
            frame[90:1080, 948:972] = 1.0
            frame[520:540, 330:1590] = 1.0
            e.update(bg=bg, ground=frame, gcol=(0.012, 0.012, 0.02), gy=1000, prof=np.full(W, 1000.0), rim=(1.0, 0.75, 0.5))
        elif name == "space":
            from people import stars_layer as sl
            bg = sky_gradient((0.0, 0.0, 0.01), (0.01, 0.01, 0.03), (0.02, 0.02, 0.05), 1000) + sl(rng, 3000)
            e.update(bg=bg, ground=np.zeros((H, W), np.float32), gcol=(0, 0, 0), gy=H, prof=np.full(W, float(H)), rim=(0.6, 0.8, 1.0))
        return e
    return cached(("env", name, seed), build)


def dynamic_bg(e, t, P):
    img = e["bg"].copy()
    if e.get("aurora"):
        a = aurora(t)
        img += a * (0.8 + 0.4 * P.get("bass", 0.5))
    if e.get("sea"):
        sx, sy = e["sun"]
        yy, xx = np.mgrid[640:H, 0:W].astype(np.float32)
        sh = np.sin(yy * 0.5 + t * 3.0 + np.sin(xx * 0.01 + t) * 2) * 0.5 + 0.5
        band = np.exp(-((xx - sx) / (60 + (yy - 640) * 0.5)) ** 2)
        img[640:] += (band * sh ** 6)[..., None] * np.array([1.4, 1.0, 0.6], np.float32) * 0.8
    gm = e["ground"][..., None]
    img = img * (1 - gm) + gm * np.array(e["gcol"], np.float32)
    return img


def aurora(t):
    def build():
        xs = np.arange(W, dtype=np.float32)
        return xs
    xs = cached(("aurora_x",), build)
    img = np.zeros((H, W, 3), np.float32)
    for k, (c, y0, amp) in enumerate([((0.2, 1.0, 0.55), 330, 90), ((0.4, 0.6, 1.0), 260, 70), ((0.8, 0.3, 0.9), 220, 50)]):
        cy = y0 + amp * np.sin(xs * 0.004 + t * 0.25 + k) + 30 * np.sin(xs * 0.013 - t * 0.4 + k * 2)
        yy = np.arange(H, dtype=np.float32)[:, None]
        d = (yy - cy[None, :])
        curtain = np.exp(-np.maximum(d, 0) / 140) * np.exp(-np.maximum(-d, 0) / 18) * (d > -60)
        stri = 0.6 + 0.4 * np.sin(xs * 0.05 + t * 1.5 + k)[None, :]
        img += (curtain * stri)[..., None] * np.array(c, np.float32) * 0.35
    return img


def ground_y(e, x):
    xi = int(np.clip(x, 0, W - 1))
    return float(e["prof"][xi])


# ---------------------------------------------------------------- post helpers
def camera(img, z, px=0.0, py=0.0, cx=W / 2, cy=H / 2):
    if abs(z - 1) < 1e-3 and abs(px) < 0.5 and abs(py) < 0.5:
        return img
    M = np.float32([[z, 0, cx * (1 - z) + px], [0, z, cy * (1 - z) + py]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def add_glow(img, glow, k=1.0, sigma=6):
    img += glow * 0.8 * k
    small = cv2.resize(glow, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    img += cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma / 2), (W, H)) * 1.6 * k
    return img


def aura_ring(glow, x, y, t, col, period=2.2, phase=0.0, maxr=160, a=1.0):
    age = ((t + phase) % period) / period
    r = 20 + maxr * age
    cv2.circle(glow, PT(x, y), int(r * K), tuple(float(c) * a * (1 - age) ** 1.5 for c in col), 2, cv2.LINE_AA, SH)


# ---------------------------------------------------------------- hands close-up
def draw_hand(mask, wx, wy, ang, s, curl=0.0, mirror=False, reach=0.0):
    """手のシルエット。手首(wx,wy)、指先方向 ang(rad, 0=右)、手のひら長 s"""
    c, si = math.cos(ang), math.sin(ang)
    my = -1 if mirror else 1
    def T(x, y):
        y = y * my
        return (wx + (x * c - y * si) * s, wy + (x * si + y * c) * s)
    # 腕
    capsule(mask, T(-3.2, 0.0), T(0.0, 0.0), 0.36 * s, 0.3 * s)
    fill(mask, [T(0, -0.36), T(0.95, -0.42), T(1.05, 0.0), T(0.95, 0.4), T(0.1, 0.42)])
    # 指（人差し指〜小指）
    for k, (yy, ln) in enumerate([(-0.3, 0.82), (-0.1, 0.92), (0.1, 0.86), (0.28, 0.7)]):
        cu = curl * (1.0 - (reach if k == 0 else 0))
        a1 = cu * 1.2; a2 = cu * 1.5
        p0 = (0.98, yy)
        p1 = (p0[0] + ln * 0.5 * math.cos(a1), p0[1] + ln * 0.5 * math.sin(a1) * 0.3 + cu * 0.1)
        p2 = (p1[0] + ln * 0.45 * math.cos(a1 + a2), p1[1] + ln * 0.45 * math.sin(a1 + a2) * 0.6)
        capsule(mask, T(*p0), T(*p1), 0.1 * s, 0.09 * s)
        capsule(mask, T(*p1), T(*p2), 0.09 * s, 0.075 * s)
    # 親指
    capsule(mask, T(0.25, -0.35), T(0.65, -0.75 + curl * 0.3), 0.12 * s, 0.1 * s)
    return T(1.75 - curl * 0.8, -0.1)


# ---------------------------------------------------------------- scene functions
def sc_alone(tl, t, P, env_name="dusk_hill", seed=1, x=640, hue=0.6, pose="sway", height=360, z=(1.0, 1.08), others=()):
    """ひとりで立つ人（バラバラの声）"""
    e = env(env_name, seed)
    img = dynamic_bg(e, t, P)
    rng = np.random.default_rng(seed + 100)
    mask = np.zeros((H, W), np.uint8)
    glow = np.zeros((H, W, 3), np.float32)
    figs = [(x, height, hue)] + list(others)
    for k, (fx, fh, fhue) in enumerate(figs):
        gy = ground_y(e, fx) + 8
        p = Person(np.random.default_rng(seed + 10 + k))
        J = p.joints(pose, t)
        p.draw(mask, J, tf_upright(fx, gy, fh), fh)
        col = hue_rgb(fhue, 0.75, 1.0)
        for q in range(2):
            aura_ring(glow, fx, gy - fh * 0.55, tl, col * 0.9, period=2.6 + k * 0.7, phase=q * 1.3 + k, maxr=fh * 0.7)
        circ_glow = np.zeros((H, W), np.uint8)
        cv2.circle(circ_glow, PT(fx, gy - fh * 0.72), int(fh * 0.05 * K), 255, -1, cv2.LINE_AA, SH)
        glow += (circ_glow.astype(np.float32) / 255)[..., None] * col * (0.5 + 0.5 * math.sin(t * 2 + k))
    m = mask.astype(np.float32) / 255
    img = composite_silhouette(img, m, e["rim"], e["ldir"], rimk=2.5)
    img = add_glow(img, glow, 0.9)
    u = tl / max(P["shot_len"], 1e-3)
    return camera(img, z[0] + (z[1] - z[0]) * u, cx=x, cy=600)


def sc_walkers(tl, t, P, env_name="city_night", seed=3, n=9):
    """すれ違う人々（混ざり合わない色の波）"""
    e = env(env_name, seed)
    img = dynamic_bg(e, t, P)
    rng = np.random.default_rng(seed)
    mask = np.zeros((H, W), np.uint8)
    glow = np.zeros((H, W, 3), np.float32)
    for k in range(n):
        dirn = 1 if k % 2 == 0 else -1
        depth = rng.random()
        h = 240 + 220 * depth
        gy = e["gy"] + 20 + 160 * depth
        speed = (70 + 60 * rng.random()) * dirn * (0.6 + depth)
        x = (rng.random() * (W + 400) + speed * t) % (W + 400) - 200
        p = Person(np.random.default_rng(seed * 10 + k))
        J = p.joints("walk", t * (0.9 + 0.3 * rng.random()))
        if dirn < 0:
            tf = lambda q, x=x, gy=gy, h=h: (x - q[0] * h, gy - q[1] * h)
        else:
            tf = tf_upright(x, gy, h)
        p.draw(mask, J, tf, h)
        col = hue_rgb(rng.random(), 0.7, 1.0)
        aura_ring(glow, x, gy - h * 0.55, t, col * 0.7, period=1.8 + rng.random(), phase=rng.random() * 3, maxr=h * 0.45)
    m = mask.astype(np.float32) / 255
    img = composite_silhouette(img, m, e["rim"], (0.0, -1.0), rimk=2.0)
    img = add_glow(img, glow, 0.8)
    u = tl / max(P["shot_len"], 1e-3)
    return camera(img, 1.04 + 0.04 * u, px=-30 * u)


def sc_pullback(tl, t, P, env_name="night_hill", seed=5, hue=0.58):
    """強がるたびに遠くなって: ひとりの人からカメラが離れていく"""
    e = env(env_name, seed)
    img = dynamic_bg(e, t, P)
    mask = np.zeros((H, W), np.uint8)
    glow = np.zeros((H, W, 3), np.float32)
    x = 960; gy = ground_y(e, x) + 6; h = 300
    p = Person(np.random.default_rng(seed + 3), style="short")
    J = p.joints("stand", t)
    p.draw(mask, J, tf_upright(x, gy, h), h)
    m = mask.astype(np.float32) / 255
    img = composite_silhouette(img, m, e["rim"], e["ldir"], rimk=2.5)
    col = hue_rgb(hue, 0.6, 1.0)
    aura_ring(glow, x, gy - h * 0.6, t, col * 0.6, period=3.0, maxr=120)
    img = add_glow(img, glow, 0.8)
    u = tl / max(P["shot_len"], 1e-3)
    return camera(img, 1.9 - 0.85 * smooth(u), cx=x, cy=gy - h * 0.55)


def sc_chest(tl, t, P, env_name="window_night", seed=7, dim=True):
    """本当の気持ち隠してた: 胸に手を当てる人（胸の光が揺らぐ）"""
    e = env(env_name, seed)
    img = dynamic_bg(e, t, P)
    mask = np.zeros((H, W), np.uint8)
    x = 1180; gy = 1720; h = 1350
    p = Person(np.random.default_rng(seed + 4), style="long")
    J = p.joints("chest", t)
    p.draw(mask, J, tf_upright(x, gy, h), h)
    m = mask.astype(np.float32) / 255
    img = composite_silhouette(img, m, e["rim"], (-0.7, -0.7), rimk=3.5, rim_strength=1.3)
    glow = np.zeros((H, W, 3), np.float32)
    hx, hy = tf_upright(x, gy, h)((-0.02, 0.7))
    flick = 0.35 + 0.25 * math.sin(t * 7) * math.sin(t * 3.1) if dim else 0.8 + 0.5 * P["beat"]
    cv2.circle(glow, PT(hx, hy), int(26 * K), (1.0 * flick, 0.7 * flick, 0.45 * flick), -1, cv2.LINE_AA, SH)
    img = add_glow(img, glow, 1.0, sigma=10)
    u = tl / max(P["shot_len"], 1e-3)
    return camera(img, 1.0 + 0.06 * u, px=-20 * u)


def sc_hands(tl, t, P, seed=9, mode="reach"):
    """差し出した手を / 握り返せば: 手のクローズアップ"""
    rng = np.random.default_rng(seed)
    bg = cached(("handbg", seed), lambda: (sky_gradient((0.02, 0.02, 0.06), (0.1, 0.07, 0.15), (0.35, 0.2, 0.25), 900)
                                          + radial_glow(960, 520, 520, (0.6, 0.4, 0.3), power=1.0) * 0.4
                                          + stars_layer(np.random.default_rng(seed), 1200, band=False) * 0.6))
    img = bg.copy()
    mask = np.zeros((H, W), np.uint8)
    glow = np.zeros((H, W, 3), np.float32)
    u = tl / max(P["shot_len"], 1e-3)
    if mode == "reach":
        gap = 260 * (1 - smooth(u * 1.15)) + 18
        s = 150
        tipL = draw_hand(mask, 960 - gap / 2 - 1.75 * s, 560 + 30 * (1 - u), -0.12, s, curl=0.35, reach=1.0)
        tipR = draw_hand(mask, 960 + gap / 2 + 1.75 * s, 520 - 30 * (1 - u), math.pi + 0.1, s, curl=0.35, mirror=True, reach=1.0)
        cx, cy = 960, 540
        touch = smooth((u - 0.75) / 0.15)
        spark = 0.25 + 1.6 * touch
        cv2.circle(glow, PT(cx, cy), int((10 + 30 * touch) * K), (1.2 * spark, 0.9 * spark, 0.6 * spark), -1, cv2.LINE_AA, SH)
        if touch > 0:
            for q in range(3):
                aura_ring(glow, cx, cy, tl, np.array([1.0, 0.8, 0.5]) * touch, period=1.2, phase=q * 0.4, maxr=420)
    else:
        s = 160
        draw_hand(mask, 960 - 2.2 * s, 600, -0.25, s, curl=0.85)
        draw_hand(mask, 960 + 2.2 * s, 600, math.pi + 0.25, s, curl=0.85, mirror=True)
        circ(mask, (960, 560), 0.55 * s)
        cx, cy = 960, 560
        burst = 0.8 + 0.8 * P["beat"]
        cv2.circle(glow, PT(cx, cy - 10), int(60 * K), (1.3 * burst, 1.0 * burst, 0.6 * burst), -1, cv2.LINE_AA, SH)
        for q in range(4):
            aura_ring(glow, cx, cy, tl, np.array([1.0, 0.8, 0.5]), period=1.6, phase=q * 0.4, maxr=700)
        rising_particles(glow, seed, tl, 140, 300, 1620, 0, 900, (1.0, 0.8, 0.5), speed=90)
    m = mask.astype(np.float32) / 255
    img = composite_silhouette(img, m, (1.0, 0.75, 0.5), (0.0, -1.0), rimk=4.0, rim_strength=1.4)
    img = add_glow(img, glow, 1.0, sigma=12)
    return camera(img, 1.0 + 0.05 * u)


def sc_line(tl, t, P, env_name="sunrise_ridge", seed=11, n=9, raise_keys=((0, 0), (1, 0)), height=330, z=(1.0, 1.06),
            energy=1.0, x0=220, x1=1700):
    """手をつなぐ人々の列。エネルギーが手から手へ流れる"""
    e = env(env_name, seed)
    img = dynamic_bg(e, t, P)
    rng = np.random.default_rng(seed)
    mask = np.zeros((H, W), np.uint8)
    glow = np.zeros((H, W, 3), np.float32)
    ppl = line_of_people(rng, n, x0, x1, 0, height, jitter=0.04)
    for q in ppl:
        q["y"] = ground_y(e, q["x"]) + 6
    u = tl / max(P["shot_len"], 1e-3)
    ra = float(np.interp(u, [k[0] for k in raise_keys], [k[1] for k in raise_keys]))
    hands = draw_chain(mask, ppl, t, raise_amt=smooth(ra))
    m = mask.astype(np.float32) / 255
    img = composite_silhouette(img, m, e["rim"], e["ldir"], rimk=2.5)
    chain = []
    for hl, hr in hands:
        chain += [hl, hr]
    col = np.array([1.0, 0.82, 0.5]) * (1 - P["coh"] * 0.0)
    if energy > 0:
        energy_polyline(glow, chain[1:-1], col * energy * (0.7 + 0.6 * P["beat"]), 3, t, pulse_speed=1.2)
        for (hl, hr) in hands[:-1]:
            pass
        rising_particles(glow, seed, tl, 90, x0, x1, 200, 800, (1.0, 0.85, 0.55), speed=70)
        for q in range(2):
            aura_ring(glow, (x0 + x1) / 2, chain[len(chain) // 2][1], tl, col * energy * 0.7, period=BEAT * 2, phase=q * BEAT, maxr=900)
    img = add_glow(img, glow, 1.0)
    return camera(img, z[0] + (z[1] - z[0]) * u, cy=650)


def sc_circle(tl, t, P, env_name="night_hill", seed=13, n=14, rx=520, ry=120, cy=800, height=250, column=1.0, raise_keys=((0, 0), (1, 0))):
    """輪になって手をつなぐ人々。中心から光の柱、地面に波紋"""
    e = env(env_name, seed)
    img = dynamic_bg(e, t, P)
    rng = np.random.default_rng(seed)
    glow = np.zeros((H, W, 3), np.float32)
    u = tl / max(P["shot_len"], 1e-3)
    rot = t * 0.08
    ppl = []
    for k in range(n):
        a = rot + k * 2 * np.pi / n
        x = 960 + rx * math.sin(a); depth = math.cos(a)  # +1 奥
        y = cy - ry * depth
        sc = 1.0 - 0.3 * depth
        ppl.append(dict(p=Person(np.random.default_rng(seed * 7 + k), child=(k % 5 == 3)), x=x, y=y, h=height * sc * (0.65 if k % 5 == 3 else 1.0), depth=depth, a=a))
    ra = smooth(float(np.interp(u, [k[0] for k in raise_keys], [k[1] for k in raise_keys])))
    # 奥の人 → 光の柱 → 手前の人
    def draw_set(sel):
        mask = np.zeros((H, W), np.uint8)
        order = sorted(sel, key=lambda q: -q["depth"])
        hands = []
        for q in order:
            i = ppl.index(q)
            L = ppl[(i - 1) % n]; R = ppl[(i + 1) % n]
            hy = 0.5 + 0.68 * ra
            inv = lambda X, Y, q=q: ((X - q["x"]) / q["h"], (q["y"] - Y) / q["h"])
            hl = inv((L["x"] + q["x"]) / 2, (L["y"] + q["y"]) / 2 - (L["h"] + q["h"]) / 2 * hy)
            hr = inv((R["x"] + q["x"]) / 2, (R["y"] + q["y"]) / 2 - (R["h"] + q["h"]) / 2 * hy)
            J = q["p"].joints("hold", t, hl, hr)
            tf = tf_upright(q["x"], q["y"], q["h"])
            q["p"].draw(mask, J, tf, q["h"])
            hands.append(tf(J["hl"]))
        return mask.astype(np.float32) / 255, hands
    back = [q for q in ppl if q["depth"] > 0]; front = [q for q in ppl if q["depth"] <= 0]
    mb, hb = draw_set(back)
    img = composite_silhouette(img, mb, e["rim"], e["ldir"], rimk=2.0)
    # 光の柱と地面の波紋
    beats = [b for b in P["beats_near"]]
    ground_rings(glow, 960, cy, rx * 0.9, ry * 0.9, t, beats, np.array([1.0, 0.85, 0.55]) * (0.6 + 0.6 * P["coh"]), life=BEAT * 4, maxr=1.8)
    if column > 0:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        colm = np.exp(-((xx - 960) / (40 + 30 * P["beat"])) ** 2) * np.clip((cy - yy) / cy, 0, 1) ** 0.6 * (yy < cy)
        img += colm[..., None] * np.array([1.0, 0.85, 0.6], np.float32) * column * (0.5 + 0.5 * P["beat"])
        cv2.circle(glow, PT(960, cy - 40), int(70 * K), (1.4 * column, 1.1 * column, 0.7 * column), -1, cv2.LINE_AA, SH)
        rising_particles(glow, seed, tl, 120, 700, 1220, 0, cy, (1.0, 0.85, 0.6), speed=120)
    mf, hf = draw_set(front)
    img = composite_silhouette(img, mf, (1.0, 0.85, 0.6), (0.0, -1.0), rimk=2.0, rim_strength=0.6)
    # 手から手への光
    allh = sorted([(q["a"], q) for q in ppl], key=lambda z: z[0])
    pts = []
    for k in range(n + 1):
        q = ppl[k % n]; R = ppl[(k + 1) % n]
        hy = 0.5 + 0.68 * ra
        pts.append(((q["x"] + R["x"]) / 2, (q["y"] + R["y"]) / 2 - (q["h"] + R["h"]) / 2 * hy))
    energy_polyline(glow, pts, np.array([1.0, 0.82, 0.5]) * (0.5 + 0.5 * P["beat"]), 2, t, pulse_speed=0.8)
    img = add_glow(img, glow, 1.0)
    return camera(img, 1.0 + 0.07 * u, cy=cy - 150)


def sc_two(tl, t, P, env_name="dusk_hill", seed=15, mode="touch"):
    """二人: 手のひらを合わせる(touch) / 抱きしめる(embrace)"""
    e = env(env_name, seed)
    img = dynamic_bg(e, t, P)
    mask = np.zeros((H, W), np.uint8)
    glow = np.zeros((H, W, 3), np.float32)
    u = tl / max(P["shot_len"], 1e-3)
    h = 520
    gy = ground_y(e, 960) + 10
    pa = Person(np.random.default_rng(seed + 1), style="hat")
    pb = Person(np.random.default_rng(seed + 2), style="scarf")
    if mode == "touch":
        gap = 340 - 60 * smooth(u)
        xa, xb = 960 - gap / 2 - 30, 960 + gap / 2 + 30
        meet = (960, gy - h * 0.78)
        Ja = pa.joints("stand", t, None, ((meet[0] - xa) / h, (gy - meet[1]) / h))
        Jb = pb.joints("stand", t, ((meet[0] - xb) / h, (gy - meet[1]) / h), None)
        pa.draw(mask, Ja, tf_upright(xa, gy, h), h); pb.draw(mask, Jb, tf_upright(xb, gy, h * 0.94), h * 0.94)
        k = 0.4 + 1.2 * smooth((u - 0.3) / 0.3)
        cv2.circle(glow, PT(*meet), int(22 * K), (1.2 * k, 0.95 * k, 0.6 * k), -1, cv2.LINE_AA, SH)
        for q in range(3):
            aura_ring(glow, meet[0], meet[1], tl, np.array([1.0, 0.8, 0.55]) * k, period=1.5, phase=q * 0.5, maxr=650)
        cx, cy = 960, gy - h * 0.6
    else:
        xa, xb = 900, 1020
        sa = tf_upright(xa, gy, h); sb = tf_upright(xb, gy, h * 0.94)
        # 互いの肩に腕をまわす
        Ja = pa.joints("stand", t, None, ((xb + 70 - xa) / h, 0.78))
        Jb = pb.joints("stand", t, ((xa - 70 - xb) / (h * 0.94), 0.8), None)
        pa.draw(mask, Ja, sa, h); pb.draw(mask, Jb, sb, h * 0.94)
        cx, cy = 960, gy - h * 0.62
        k = 0.8 + 0.5 * P["beat"]
        cv2.circle(glow, PT(cx, cy), int(50 * K), (1.2 * k, 0.85 * k, 0.6 * k), -1, cv2.LINE_AA, SH)
        for q in range(2):
            aura_ring(glow, cx, cy, tl, np.array([1.0, 0.75, 0.6]), period=2.0, phase=q, maxr=700)
    m = mask.astype(np.float32) / 255
    img = composite_silhouette(img, m, e["rim"], e["ldir"], rimk=3.0)
    img = add_glow(img, glow, 1.0, sigma=10)
    return camera(img, 1.0 + 0.07 * u, cx=cx, cy=cy)


def sc_share(tl, t, P, env_name="desert_dusk", seed=17, n=7):
    """分け合おう: 光の玉が手から手へ渡されていく"""
    e = env(env_name, seed)
    img = dynamic_bg(e, t, P)
    rng = np.random.default_rng(seed)
    mask = np.zeros((H, W), np.uint8)
    glow = np.zeros((H, W, 3), np.float32)
    ppl = line_of_people(rng, n, 260, 1660, 0, 340, jitter=0.02)
    for q in ppl:
        q["y"] = ground_y(e, q["x"]) + 6
    u = tl / max(P["shot_len"], 1e-3)
    pos = u * (n - 1)
    hands = []
    for i, q in enumerate(ppl):
        d = abs(i - pos)
        lift = smooth(1 - d)
        hx = (0.12) ; hy = 0.62 + 0.25 * lift
        J = q["p"].joints("stand", t, (-0.05, hy), (0.12 + 0.12 * lift, hy))
        tf = tf_upright(q["x"], q["y"], q["h"])
        q["p"].draw(mask, J, tf, q["h"])
        hands.append(tf(J["hr"]))
    m = mask.astype(np.float32) / 255
    img = composite_silhouette(img, m, e["rim"], e["ldir"], rimk=2.5)
    i0 = int(min(pos, n - 2)); fr = pos - i0
    ox = hands[i0][0] * (1 - fr) + hands[i0 + 1][0] * fr
    oy = hands[i0][1] * (1 - fr) + hands[i0 + 1][1] * fr - 60 * math.sin(fr * math.pi)
    cv2.circle(glow, PT(ox, oy), int(26 * K), (1.5, 1.2, 0.7), -1, cv2.LINE_AA, SH)
    for i in range(int(pos) + 1):
        hx, hy = hands[min(i, n - 1)]
        cv2.circle(glow, PT(hx, hy), int(9 * K), (1.0, 0.8, 0.5), -1, cv2.LINE_AA, SH)
    rising_particles(glow, seed, tl, 60, 200, 1700, 300, 800, (1.0, 0.8, 0.5), speed=50)
    img = add_glow(img, glow, 1.0)
    return camera(img, 1.02 + 0.04 * u, px=40 * (0.5 - u))


def sc_lanterns(tl, t, P, env_name="night_hill", seed=19, n=60):
    """小さな優しさが波になって: たくさんの人が灯す小さな光が空へ"""
    e = env(env_name, seed)
    img = dynamic_bg(e, t, P)
    rng = np.random.default_rng(seed)
    mask = np.zeros((H, W), np.uint8)
    glow = np.zeros((H, W, 3), np.float32)
    u = tl / max(P["shot_len"], 1e-3)
    for k in range(n):
        depth = rng.random()
        x = rng.random() * W
        h = 60 + 160 * depth
        gy = ground_y(e, x) + 20 + 220 * depth
        p = Person(np.random.default_rng(seed * 3 + k), child=rng.random() < 0.2)
        J = p.joints("stand", t, None, (0.12, 0.9))
        tf = tf_upright(x, gy, h)
        p.draw(mask, J, tf, h)
        lx, ly = tf((0.12, 0.98))
        tk = rng.random() * 2.0
        lit = smooth((u * 3.0 - tk) / 0.3)
        cv2.circle(glow, PT(lx, ly), int((3 + 5 * depth) * K), (1.2 * lit, 0.8 * lit, 0.4 * lit), -1, cv2.LINE_AA, SH)
        # 灯が空へ昇る
        rise = max(0.0, u * 3.0 - tk - 0.3)
        if rise > 0:
            ry = ly - rise * 260; rxx = lx + math.sin(rise * 2 + k) * 20
            cv2.circle(glow, PT(rxx, ry), int((3 + 4 * depth) * K), (1.2, 0.85, 0.45), -1, cv2.LINE_AA, SH)
    m = mask.astype(np.float32) / 255
    img = composite_silhouette(img, m, e["rim"], e["ldir"], rimk=1.5)
    img = add_glow(img, glow, 1.2)
    return camera(img, 1.0 + 0.05 * u, py=-30 * u)


def sc_reach_sky(tl, t, P, env_name="night_hill", seed=21):
    """ほら手を伸ばして: 子どもが空へ手を伸ばす"""
    e = env(env_name, seed)
    img = dynamic_bg(e, t, P)
    mask = np.zeros((H, W), np.uint8)
    glow = np.zeros((H, W, 3), np.float32)
    u = tl / max(P["shot_len"], 1e-3)
    x = 900; gy = ground_y(e, x) + 6; h = 420
    p = Person(np.random.default_rng(seed + 5), child=True, style="pony")
    reach_y = 0.6 + 0.55 * smooth(u * 1.5)
    J = p.joints("stand", t, None, (0.3, reach_y))
    tf = tf_upright(x, gy, h)
    p.draw(mask, J, tf, h)
    m = mask.astype(np.float32) / 255
    img = composite_silhouette(img, m, e["rim"], e["ldir"], rimk=2.5)
    hx, hy = tf(J["hr"])
    sx, sy = hx + 260, hy - 300
    k = 0.6 + 0.6 * smooth((u - 0.3) / 0.4)
    cv2.circle(glow, PT(sx, sy), int(14 * K), (1.4 * k, 1.2 * k, 0.9 * k), -1, cv2.LINE_AA, SH)
    energy_polyline(glow, [(hx, hy), (sx, sy)], np.array([0.9, 0.8, 0.6]) * smooth((u - 0.4) / 0.4), 2, t, pulses=False)
    img = add_glow(img, glow, 1.0, sigma=10)
    return camera(img, 1.0 + 0.08 * u, cx=x, cy=gy - h * 0.6)


def sc_globe_ring(tl, t, P, scene3d=None, seed=23, n=34, R=300, raise_keys=((0, 0), (1, 0))):
    """地球のまわりに世界中の人々が立ち、手をつないで輪になる"""
    e = env("space", seed)
    img = e["bg"].copy()
    u = tl / max(P["shot_len"], 1e-3)
    cx, cy = 960, 520
    if scene3d is not None:
        img += scene3d.earth_disc(t, cx, cy, R, P)
    mask = np.zeros((H, W), np.uint8)
    glow = np.zeros((H, W, 3), np.float32)
    rot = t * 0.05
    ra = smooth(float(np.interp(u, [k[0] for k in raise_keys], [k[1] for k in raise_keys])))
    h = 92
    pts = []
    for k in range(n):
        a = rot + k * 2 * np.pi / n
        bx = cx + R * math.sin(a); by = cy - R * math.cos(a)
        p = Person(np.random.default_rng(seed * 5 + k), child=(k % 6 == 2))
        hh = h * (0.7 if k % 6 == 2 else 1.0)
        half = math.sin(np.pi / n) * R / hh
        J = p.joints("hold", t, (-half, 0.5 + 0.6 * ra), (half, 0.5 + 0.6 * ra))
        tf = tf_rot(bx, by, hh, a)
        p.draw(mask, J, tf, hh)
        pts.append(tf(J["hr"]))
    pts.append(pts[0])
    m = mask.astype(np.float32) / 255
    img = composite_silhouette(img, m, (0.6, 0.85, 1.2), (0.0, -1.0), silcol=(0.01, 0.012, 0.02), rimk=1.5, rim_strength=0.0)
    # 縁を地球の光で照らす
    energy_polyline(glow, pts, np.array([1.0, 0.85, 0.55]) * (0.6 + 0.6 * P["beat"]), 2, t, pulse_speed=1.5)
    for q in range(3):
        age = ((tl + q * BEAT) % (BEAT * 3)) / (BEAT * 3)
        cv2.circle(glow, PT(cx, cy), int((R + 80 + 600 * age) * K), tuple(float(c) * (1 - age) ** 2 for c in (1.0, 0.85, 0.55)), 3, cv2.LINE_AA, SH)
    img = add_glow(img, glow, 1.0)
    return camera(img, 1.0 + 0.1 * u, cy=cy)


def sc_lookup(tl, t, P, env_name="aurora_snow", seed=25, n=5):
    """空を見上げる人々（オーロラ / 星空）"""
    e = env(env_name, seed)
    img = dynamic_bg(e, t, P)
    rng = np.random.default_rng(seed)
    mask = np.zeros((H, W), np.uint8)
    glow = np.zeros((H, W, 3), np.float32)
    u = tl / max(P["shot_len"], 1e-3)
    ppl = line_of_people(rng, n, 600, 1320, 0, 300, jitter=0.15)
    for q in ppl:
        q["y"] = ground_y(e, q["x"]) + 8
    hands = draw_chain(mask, ppl, t, raise_amt=0.0)
    m = mask.astype(np.float32) / 255
    img = composite_silhouette(img, m, e["rim"], e["ldir"], rimk=2.0, rim_strength=0.6)
    chain = []
    for hl, hr in hands:
        chain += [hl, hr]
    energy_polyline(glow, chain[1:-1], np.array([0.6, 1.0, 0.8]) * 0.6, 2, t)
    img = add_glow(img, glow, 1.0)
    return camera(img, 1.0 + 0.05 * u, py=20 * u)
