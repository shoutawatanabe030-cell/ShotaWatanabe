# -*- coding: utf-8 -*-
"""
実写風の人物画像（Canva AI で生成）を動かすカット。
ゆっくりしたズーム・パン（Ken Burns）に、拍に合わせた波動の輪・光の粒・グローを重ねる。
"""
import os, math
import numpy as np
import cv2

W, H = 1920, 1080
IMGDIR = os.environ.get("MV_PHOTOS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "photos"))
BEAT = 60 / 128.5714
_cache = {}


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def load(name):
    """表示用の色(0..1)を、最終トーンマップ(1-exp(-1.15x))の逆変換で線形に戻して保持"""
    if name not in _cache:
        keys = [k for k in _cache if k != "_grad"]
        if len(keys) > 10:
            _cache.pop(keys[0])
        path = os.path.join(IMGDIR, name)
        if not os.path.exists(path):  # リポジトリには容量の都合で JPEG 版を置いている
            path = os.path.splitext(path)[0] + ".jpg"
        im = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
        v = np.clip(im, 0, 0.985)
        lin = -np.log(1 - v) / 1.15
        _cache[name] = lin.astype(np.float32)
    return _cache[name]


def _bottom_grad():
    if "_grad" not in _cache:
        y = np.arange(H, dtype=np.float32)
        u = np.clip((y - 640) / 440, 0, 1)
        _cache["_grad"] = (1 - 0.55 * u * u * (3 - 2 * u))[:, None, None].astype(np.float32)
    return _cache["_grad"]


def P(x, y):
    return (int(round(x * 16)), int(round(y * 16)))


def shot(tl, t, G, img, z=(1.0, 1.08), c0=(0.5, 0.5), c1=(0.5, 0.5), fx=(), grade=None, shot_len=4.0):
    """
    img: ファイル名, z: ズーム(開始, 終了) 1.0 = 画面いっぱい, c0/c1: 画像上の注視点(正規化)
    fx: [("rings", x, y, color, maxr), ("glow", x, y, color, r), ("particles", x0, x1, y0, y1, color, n), ("sweep", color)]
    """
    src = load(img)
    sh, sw = src.shape[:2]
    u = tl / max(shot_len, 1e-3)
    e = float(smooth(u)) * 0.5 + u * 0.5
    zz = z[0] + (z[1] - z[0]) * e
    cx = (c0[0] + (c1[0] - c0[0]) * e) * sw
    cy = (c0[1] + (c1[1] - c0[1]) * e) * sh
    s = zz * max(W / sw, H / sh)
    # 画面外がはみ出さないよう注視点を制限
    halfw, halfh = W / (2 * s), H / (2 * s)
    cx = min(max(cx, halfw), sw - halfw); cy = min(max(cy, halfh), sh - halfh)
    M = np.float32([[s, 0, W / 2 - s * cx], [0, s, H / 2 - s * cy]])
    out = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def to_screen(x, y):
        return (M[0, 0] * x * sw + M[0, 2], M[1, 1] * y * sh + M[1, 2])

    if grade is not None:
        g = np.array(grade, np.float32)
        out *= g
    # テロップが読みやすいよう、画面下部をやわらかく暗くする
    out *= _bottom_grad()
    glow = np.zeros((H, W, 3), np.float32)
    beat = G.get("beat", 0.0); bass = G.get("bass", 0.5); coh = G.get("coh", 0.0)
    for f in fx:
        kind = f[0]
        if kind == "rings":
            _, x, y, col, maxr = f
            X, Y = to_screen(x, y)
            col = np.array(col, np.float32)
            period = BEAT * 2
            for q in range(3):
                age = ((t + q * period / 3) % period) / period
                r = 30 + maxr * age
                a = (1 - age) ** 1.6 * (0.6 + 0.6 * bass)
                cv2.circle(glow, P(X, Y), int(r * 16), tuple(float(c) * a for c in col), 3, cv2.LINE_AA, 4)
        elif kind == "glow":
            _, x, y, col, r = f
            X, Y = to_screen(x, y)
            k = 0.6 + 0.8 * beat * (0.4 + bass)
            cv2.circle(glow, P(X, Y), int(r * 16), tuple(float(c) * k for c in col), -1, cv2.LINE_AA, 4)
        elif kind == "particles":
            _, x0, x1, y0, y1, col, n = f
            rng = np.random.default_rng(int(x0 * 1000 + y0 * 100 + n))
            xs = rng.random(n); ph = rng.random(n); sp = 0.05 + rng.random(n) * 0.08
            yy = y1 - ((ph + t * sp) % 1.0) * (y1 - y0)
            xx = x0 + xs * (x1 - x0) + 0.01 * np.sin(t * 0.9 + ph * 30)
            tw = 0.5 + 0.5 * np.sin(t * 3 + ph * 40)
            canvas = np.zeros((H, W), np.float32)
            for a, b, w in zip(xx, yy, tw):
                X, Y = to_screen(a, b)
                cv2.circle(canvas, P(X, Y), int(2.2 * 16), float(w), -1, cv2.LINE_AA, 4)
            glow += cv2.GaussianBlur(canvas, (0, 0), 1.3)[..., None] * np.array(col, np.float32) * 1.4
        elif kind == "sweep":
            _, col = f
            xs = np.arange(W, dtype=np.float32)[None, :]
            ys = np.arange(H, dtype=np.float32)[:, None]
            pos = -400 + (W + 800) * u
            d = (xs + ys * 0.35 - pos) / 260
            glow += (np.exp(-d * d) * 0.25)[..., None] * np.array(col, np.float32)
    if glow.any():
        out += glow * 0.7
        small = cv2.resize(glow, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        out += cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (W, H)) * 1.5
    return out
