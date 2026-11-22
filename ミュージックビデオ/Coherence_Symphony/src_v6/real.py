# -*- coding: utf-8 -*-
"""
実際の写真（ご本人・イベント・デバイスの写真）を入れるカット（v6）。
実在の方の顔がゆがまないよう、写真そのものは変形させず、
ゆっくりしたズーム・パン、背景との奥行きのずれ、光の粒・光の帯・拍に合わせた縁の光で動かす。
- 横長の写真: 画面いっぱい（cover）
- 縦長の写真: 同じ写真をぼかして背景に敷き、中央に写真を置く（card）
"""
import os, math
import numpy as np
import cv2

W, H = 1920, 1080
REALDIR = os.environ.get("MV_REAL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "real"))
BEAT = 60 / 128.5714
_cache = {}


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def P(x, y):
    return (int(round(x * 16)), int(round(y * 16)))


def _lin(im):
    v = np.clip(im, 0, 0.985)
    return (-np.log(1 - v) / 1.15).astype(np.float32)


def load(name):
    if name in _cache:
        return _cache[name]
    if len(_cache) > 6:
        _cache.pop(next(iter(_cache)))
    im = cv2.imread(os.path.join(REALDIR, name), cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
    sh, sw = im.shape[:2]
    ent = dict(src=_lin(im), sw=sw, sh=sh)
    if True:
        # 背景用: 画面を覆う大きさにして強くぼかし、暗くする
        s = max(W * 1.2 / sw, H * 1.2 / sh)
        bg = cv2.resize(im, (int(sw * s), int(sh * s)), interpolation=cv2.INTER_AREA)
        bg = cv2.GaussianBlur(bg, (0, 0), 40)
        g = bg.mean(2, keepdims=True)
        bg = (bg * 0.6 + g * 0.4) * 0.42
        ent["bg"] = _lin(bg)
        # 角を丸めたマスク
        r = int(0.03 * sw)
        m = np.zeros((sh, sw), np.uint8)
        cv2.rectangle(m, (r, 0), (sw - r, sh), 255, -1)
        cv2.rectangle(m, (0, r), (sw, sh - r), 255, -1)
        for cx, cy in ((r, r), (sw - r, r), (r, sh - r), (sw - r, sh - r)):
            cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
        ent["mask"] = m.astype(np.float32) / 255.0
    _cache[name] = ent
    return ent


def _grad():
    if "_grad" not in _cache:
        y = np.arange(H, dtype=np.float32)
        u = np.clip((y - 640) / 440, 0, 1)
        _cache["_grad"] = (1 - 0.55 * u * u * (3 - 2 * u))[:, None, None].astype(np.float32)
    return _cache["_grad"]


def _particles(out, t, seed, n, col, region=(0, 1, 0, 1)):
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = region
    xs = rng.random(n); ph = rng.random(n); sp = 0.04 + rng.random(n) * 0.07
    yy = y1 - ((ph + t * sp) % 1.0) * (y1 - y0)
    xx = x0 + xs * (x1 - x0) + 0.008 * np.sin(t * 0.9 + ph * 30)
    tw = 0.5 + 0.5 * np.sin(t * 3 + ph * 40)
    canvas = np.zeros((H, W), np.float32)
    for a, b, w in zip(xx, yy, tw):
        cv2.circle(canvas, P(a * W, b * H), int(2.2 * 16), float(w), -1, cv2.LINE_AA, 4)
    out += cv2.GaussianBlur(canvas, (0, 0), 1.3)[..., None] * np.array(col, np.float32) * 1.2


def _sweep(out, u, col, k=0.09):
    xs = np.arange(W, dtype=np.float32)[None, :]
    ys = np.arange(H, dtype=np.float32)[:, None]
    pos = -500 + (W + 1000) * u
    d = (xs + ys * 0.35 - pos) / 300
    out += (np.exp(-d * d) * k)[..., None] * np.array(col, np.float32)


def shot(tl, t, G, img, layout=None, z=(1.0, 1.07), c0=(0.5, 0.5), c1=(0.5, 0.5), shot_len=4.0,
         glow=(1.0, 0.82, 0.55), particles=60, sweep=True, card_h=0.9, card_y=0.47, seed=0, bottom=0.55):
    e = load(img)
    src, sw, sh = e["src"], e["sw"], e["sh"]
    u = tl / max(shot_len, 1e-3)
    eu = float(smooth(u)) * 0.5 + u * 0.5
    beat = G.get("beat", 0.0); bass = G.get("bass", 0.5)
    layout = layout or ("card" if sw / sh < 1.25 else "cover")
    if layout == "cover":
        zz = z[0] + (z[1] - z[0]) * eu
        cx = (c0[0] + (c1[0] - c0[0]) * eu) * sw
        cy = (c0[1] + (c1[1] - c0[1]) * eu) * sh
        s = zz * max(W / sw, H / sh)
        halfw, halfh = W / (2 * s), H / (2 * s)
        cx = min(max(cx, halfw), sw - halfw); cy = min(max(cy, halfh), sh - halfh)
        M = np.float32([[s, 0, W / 2 - s * cx], [0, s, H / 2 - s * cy]])
        out = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    else:
        bg = e["bg"]; bh, bw = bg.shape[:2]
        # 背景はカードと逆向きにゆっくり動かす（奥行きのずれ）
        bz = 1.0 + 0.05 * eu
        bx = W / 2 - 40 * (eu - 0.5); by = H / 2 + 15 * (eu - 0.5)
        Mb = np.float32([[bz, 0, bx - bz * bw / 2], [0, bz, by - bz * bh / 2]])
        out = cv2.warpAffine(bg, Mb, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        zz = z[0] + (z[1] - z[0]) * eu
        s = card_h * H / sh * zz
        ccx = W / 2 + 18 * (eu - 0.5); ccy = card_y * H
        Mc = np.float32([[s, 0, ccx - s * sw / 2], [0, s, ccy - s * sh / 2]])
        card = cv2.warpAffine(src, Mc, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        m = cv2.warpAffine(e["mask"], Mc, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # 影と縁の光（拍に合わせて脈打つ）
        small = cv2.resize(m, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        shadow = cv2.resize(cv2.GaussianBlur(small, (0, 0), 6), (W, H))
        out *= (1 - 0.55 * shadow)[..., None]
        halo = cv2.resize(cv2.GaussianBlur(small, (0, 0), 10), (W, H)) - m
        k = 0.35 + 0.45 * beat * (0.4 + bass)
        out += np.clip(halo, 0, 1)[..., None] * np.array(glow, np.float32) * k
        out = out * (1 - m[..., None]) + card * m[..., None]
    if particles:
        _particles(out, t, seed + 17, particles, glow)
    if sweep:
        _sweep(out, u, glow)
    g = _grad()
    if bottom != 0.55:
        g = 1 - (1 - g) * (bottom / 0.55)
    out *= g
    return out
