# -*- coding: utf-8 -*-
"""
実写風の人物画像を「生きて動く」カットにする（v4）。
- ピントの合った部分（人物）と画面の上下から奥行きを推定し、カメラの動きで手前と奥がずれる立体的な動き（パララックス）
- 人物の呼吸・ゆれ、空や雲の流れ、水面のゆらぎ、光の帯の流れ、拍に合わせた光の脈動
- 雨、漂うボケた光、湯気、落ち葉、ちりなどの粒子
"""
import os, math
import numpy as np
import cv2

W, H = 1920, 1080
IMGDIR = os.environ.get("MV_PHOTOS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src_v3", "photos"))
BEAT = 60 / 128.5714
AW, AH = 640, 360          # 補助マップの解像度
_cache = {}


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def P(x, y):
    return (int(round(x * 16)), int(round(y * 16)))


def _grid():
    if "_grid" not in _cache:
        X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        y = np.arange(H, dtype=np.float32)
        u = np.clip((y - 640) / 440, 0, 1)
        grad = (1 - 0.55 * u * u * (3 - 2 * u))[:, None, None].astype(np.float32)
        _cache["_grid"] = (X, Y, grad)
    return _cache["_grid"]


def load(name, anim):
    key = (name, anim.get("depth", "auto"), anim.get("sky"), anim.get("water"), anim.get("flow", 0) > 0)
    if key in _cache:
        return _cache[key]
    keys = [k for k in _cache if isinstance(k, tuple)]
    if len(keys) > 8:
        _cache.pop(keys[0])
    path = os.path.join(IMGDIR, name)
    if not os.path.exists(path):
        path = os.path.splitext(path)[0] + ".jpg"
    im = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
    lin = (-np.log(1 - np.clip(im, 0, 0.985)) / 1.15).astype(np.float32)
    small = cv2.resize(im, (AW, AH), interpolation=cv2.INTER_AREA)
    gray = small.mean(2)
    # ピントの合った部分 = 手前（人物）
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    sharp = cv2.GaussianBlur(lap, (0, 0), 5)
    sharp = np.clip(sharp / (np.percentile(sharp, 97) + 1e-6), 0, 1)
    fg = cv2.GaussianBlur(sharp, (0, 0), 9)
    fg = np.clip(fg / (fg.max() + 1e-6) * 1.3, 0, 1)
    yy = np.linspace(0, 1, AH, dtype=np.float32)[:, None] * np.ones((1, AW), np.float32)
    mode = anim.get("depth", "auto")
    wfg = {"auto": 0.65, "y": 0.25, "fg": 0.85}[mode]
    depth = np.clip(wfg * fg + (1 - wfg) * yy ** 1.2, 0, 1)
    depth = cv2.GaussianBlur(depth, (0, 0), 6)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    # 人物など質感のある部分は、空・水の動きから外す
    tex = cv2.GaussianBlur(lap, (0, 0), 2)
    tex = np.clip(tex / (np.percentile(tex, 90) + 1e-6), 0, 1)
    protect = cv2.dilate((np.maximum(fg, tex) > 0.25).astype(np.float32), np.ones((15, 15), np.uint8))
    protect = cv2.GaussianBlur(protect, (0, 0), 4)
    sky = np.zeros_like(depth)
    if anim.get("sky") is not None:
        hy = anim["sky"]
        sky = smooth((hy - yy) / 0.08) * np.clip(1 - protect, 0, 1)
    water = np.zeros_like(depth)
    if anim.get("water") is not None:
        w0, w1 = anim["water"][:2]
        x0, x1 = (anim["water"][2:4] if len(anim["water"]) >= 4 else (0.0, 1.0))
        xx = np.linspace(0, 1, AW, dtype=np.float32)[None, :] * np.ones((AH, 1), np.float32)
        water = smooth((yy - w0) / 0.04) * smooth((w1 - yy) / 0.04) * smooth((xx - x0) / 0.04) * smooth((x1 - xx) / 0.04)
        water *= np.clip(1 - protect, 0, 1)
    lum = small.max(2)
    bright = np.clip((lum - 0.72) / 0.2, 0, 1) * (small[..., 0] + small[..., 1] > small[..., 2] * 1.3).astype(np.float32)
    if anim.get("flow", 0) > 0:
        bright = np.clip((lum - 0.6) / 0.25, 0, 1)
    bright = cv2.GaussianBlur(bright, (0, 0), 2)
    aux = np.stack([depth, sky, water, bright], -1).astype(np.float32)
    # ボケた光の素材（明るく、奥にある点）
    rng = np.random.default_rng(abs(hash(name)) % 2 ** 32)
    cand = np.argwhere((lum > 0.55) & (depth < 0.55))
    bok = []
    if len(cand):
        for i in rng.choice(len(cand), size=min(70, len(cand)), replace=False):
            y_, x_ = cand[i]
            c = small[y_, x_]
            bok.append((x_ / AW, y_ / AH, c, rng.random() * 6.28, 0.4 + rng.random()))
    ent = dict(src=lin, aux=aux, sw=lin.shape[1], sh=lin.shape[0], bokeh=bok, small=small)
    _cache[key] = ent
    return ent


def shot(tl, t, G, img, z=(1.0, 1.06), c0=(0.5, 0.5), c1=(0.5, 0.5), fx=(), grade=None, shot_len=4.0, anim=None):
    anim = dict(anim or {})
    e = load(img, anim)
    src, aux, sw, sh = e["src"], e["aux"], e["sw"], e["sh"]
    X, Y, grad = _grid()
    u = tl / max(shot_len, 1e-3)
    eu = float(smooth(u)) * 0.5 + u * 0.5
    zz = z[0] + (z[1] - z[0]) * eu
    cx = (c0[0] + (c1[0] - c0[0]) * eu) * sw
    cy = (c0[1] + (c1[1] - c0[1]) * eu) * sh
    s = zz * max(W / sw, H / sh) * 1.04   # 余白を少し確保（パララックス用）
    halfw, halfh = W / (2 * s), H / (2 * s)
    cx = min(max(cx, halfw), sw - halfw); cy = min(max(cy, halfh), sh - halfh)
    tx, ty = W / 2 - s * cx, H / 2 - s * cy
    sx = (X - tx) / s; sy = (Y - ty) / s
    A = cv2.remap(aux, sx * (AW / sw), sy * (AH / sh), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    D, SKY, WAT, BR = A[..., 0], A[..., 1], A[..., 2], A[..., 3]
    # パララックス（カメラが横に動くと手前ほど大きくずれる）
    par = anim.get("par", 1.0)
    dirx, diry = anim.get("pdir", (1.0, 0.25))
    px = par * 0.022 * W * (eu - 0.5) * 2 * dirx
    py = par * 0.022 * W * (eu - 0.5) * 2 * diry
    dz = par * 0.05 * eu
    offX = D * px + D * dz * (X - W / 2)
    offY = D * py + D * dz * (Y - H / 2)
    # 人物の呼吸・ゆれ
    sway = anim.get("sway", 1.0)
    if sway > 0:
        fgw = np.clip((D - 0.45) * 2.2, 0, 1)
        ph = anim.get("phase", 0.0)
        offX += fgw * sway * 3.0 * math.sin(2 * math.pi * t / 3.7 + ph)
        offY += fgw * sway * 2.2 * math.sin(2 * math.pi * t / 4.4 + ph) * ((Y - H * 0.25) / H)
    sxp = (X - offX - tx) / s
    syp = (Y - offY - ty) / s
    # 空・雲の流れ
    if anim.get("sky") is not None:
        sxp = sxp + SKY * anim.get("skyspeed", 6.0) * tl
    # 水面のゆらぎ
    if anim.get("water") is not None:
        wa = anim.get("wamp", 2.2)
        sxp = sxp + WAT * wa * 0.6 * np.sin(sxp * 0.031 + t * 1.3)
        syp = syp + WAT * wa * np.sin(syp * 0.11 + t * 2.4 + sxp * 0.004)
    # 光の帯の流れ
    fl = anim.get("flow", 0.0)
    if fl > 0:
        sxp = sxp + BR * fl * np.sin(syp * 0.013 + sxp * 0.009 - t * 2.4)
        syp = syp + BR * fl * np.cos(sxp * 0.011 - syp * 0.01 - t * 2.0)
    out = cv2.remap(src, sxp, syp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    beat = G.get("beat", 0.0); bass = G.get("bass", 0.5)
    # 拍に合わせて明るい部分が脈打つ
    pk = anim.get("pulse", 0.35)
    if pk > 0:
        out += out * (BR * (pk * beat * (0.5 + bass)))[..., None]
    if grade is not None:
        out *= np.array(grade, np.float32)
    def to_screen(xn, yn):
        return (s * xn * sw + tx, s * yn * sh + ty)
    glow = np.zeros((H, W, 3), np.float32)
    # 漂うボケた光
    if anim.get("bokeh", 0) > 0:
        canvas = np.zeros((H, W, 3), np.float32)
        for (bx, by, c, ph, sz) in e["bokeh"]:
            xx = bx + 0.012 * math.sin(t * 0.35 * sz + ph)
            yy_ = by - 0.01 * t * sz * 0.5 + 0.006 * math.cos(t * 0.4 + ph)
            X0, Y0 = to_screen(xx, yy_)
            a = (0.5 + 0.5 * math.sin(t * 0.8 * sz + ph)) * anim["bokeh"]
            r = (14 + 26 * sz) * s
            cv2.circle(canvas, P(X0, Y0), int(r * 16), tuple(float(v) * a * 0.35 for v in c), -1, cv2.LINE_AA, 4)
        out += cv2.GaussianBlur(canvas, (0, 0), 6)
    # 雨
    if anim.get("rain"):
        x0, x1, y0, y1 = anim["rain"]
        rng = np.random.default_rng(7)
        n = 260
        rx = rng.random(n); ry = rng.random(n); sp = 0.9 + rng.random(n) * 0.6; ln = 25 + rng.random(n) * 45
        canvas = np.zeros((H, W), np.float32)
        for a, b, v, L in zip(rx, ry, sp, ln):
            X0 = (x0 + a * (x1 - x0)) * W + 0.0
            Y0 = (y0 + ((b + t * v * 0.9) % 1.0) * (y1 - y0)) * H
            cv2.line(canvas, P(X0, Y0), P(X0 - L * 0.18, Y0 + L), 0.22, 1, cv2.LINE_AA, 4)
        out += cv2.GaussianBlur(canvas, (0, 0), 0.8)[..., None] * np.array([0.8, 0.85, 0.95], np.float32)
    # 湯気
    if anim.get("steam"):
        n2 = _noise()
        for (mx, my) in anim["steam"]:
            Xm, Ym = to_screen(mx, my)
            hgt, wd = 260 * s, 70 * s
            x0 = int(max(Xm - wd * 1.5, 0)); x1 = int(min(Xm + wd * 1.5, W))
            y0 = int(max(Ym - hgt, 0)); y1 = int(min(Ym, H))
            if x1 <= x0 or y1 <= y0:
                continue
            gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            v = (Ym - gy) / hgt
            nx = ((gx - Xm) * 0.6 + 900 + 60 * np.sin(v * 5 + t)) % 512
            ny = (gy * 0.5 + t * 90) % 512
            nz = cv2.remap(n2, nx.astype(np.float32), ny.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            col = np.exp(-((gx - Xm - 30 * np.sin(v * 4 + t * 0.7) * v) / (wd * (0.4 + v))) ** 2) * np.sin(np.clip(v, 0, 1) * np.pi)
            out[y0:y1, x0:x1] += (np.clip(nz - 0.45, 0, 1) * 1.6 * col * 0.55)[..., None] * np.array([0.95, 0.92, 0.88], np.float32)
    # 落ち葉
    if anim.get("leaves"):
        rng = np.random.default_rng(11)
        canvas = np.zeros((H, W, 3), np.float32)
        for k in range(40):
            x = (rng.random() * W + math.sin(t * 0.9 + k) * 60 + t * 25) % W
            y = (rng.random() * H + t * (60 + 40 * rng.random())) % H
            ang = (t * 90 + k * 40) % 360
            c = (0.9, 0.5 + 0.2 * rng.random(), 0.15)
            cv2.ellipse(canvas, P(x, y), (int(9 * 16), int(4 * 16)), ang, 0, 360, c, -1, cv2.LINE_AA, 4)
        out = out * (1 - np.clip(canvas.sum(2, keepdims=True) * 3, 0, 1) * 0.0) + canvas * 0.55
    # もや
    if anim.get("mist"):
        n2 = _noise()
        yy0 = int(anim["mist"] * H)
        gy, gx = np.mgrid[yy0:H, 0:W].astype(np.float32)
        nz = cv2.remap(n2, ((gx * 0.25 + t * 30) % 512).astype(np.float32), ((gy * 0.5) % 512).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        out[yy0:] += (np.clip(nz - 0.4, 0, 1) * 0.25)[..., None] * np.array([0.8, 0.82, 0.9], np.float32)
    # 既存の波動エフェクト（輪・グロー・粒子・光のスイープ）
    for f in fx:
        kind = f[0]
        if kind == "rings":
            _, x, y, col, maxr = f
            Xc, Yc = to_screen(x, y)
            period = BEAT * 2
            for q in range(3):
                age = ((t + q * period / 3) % period) / period
                r = 30 + maxr * age
                a = (1 - age) ** 1.6 * (0.6 + 0.6 * bass)
                cv2.circle(glow, P(Xc, Yc), int(r * 16), tuple(float(c) * a for c in col), 3, cv2.LINE_AA, 4)
        elif kind == "glow":
            _, x, y, col, r = f
            Xc, Yc = to_screen(x, y)
            k = 0.6 + 0.8 * beat * (0.4 + bass)
            cv2.circle(glow, P(Xc, Yc), int(r * 16), tuple(float(c) * k for c in col), -1, cv2.LINE_AA, 4)
        elif kind == "particles":
            _, x0, x1, y0, y1, col, n = f
            rng = np.random.default_rng(int(x0 * 1000 + y0 * 100 + n))
            xs = rng.random(n); ph = rng.random(n); sp = 0.05 + rng.random(n) * 0.08
            yy = y1 - ((ph + t * sp) % 1.0) * (y1 - y0)
            xx = x0 + xs * (x1 - x0) + 0.01 * np.sin(t * 0.9 + ph * 30)
            tw = 0.5 + 0.5 * np.sin(t * 3 + ph * 40)
            canvas = np.zeros((H, W), np.float32)
            for a, b, w in zip(xx, yy, tw):
                Xc, Yc = to_screen(a, b)
                cv2.circle(canvas, P(Xc, Yc), int(2.2 * 16), float(w), -1, cv2.LINE_AA, 4)
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
    out *= grad
    return out


def _noise():
    if "_noise" not in _cache:
        rng = np.random.default_rng(3)
        fy = np.fft.fftfreq(512)[:, None]; fx = np.fft.rfftfreq(512)[None, :]
        f = np.sqrt(fx ** 2 + fy ** 2); f[0, 0] = 1
        amp = f ** -1.3; amp[0, 0] = 0
        n = np.fft.irfft2(amp * np.exp(2j * np.pi * rng.random(amp.shape)), s=(512, 512))
        n = (n - n.min()) / (n.max() - n.min())
        _cache["_noise"] = n.astype(np.float32)
    return _cache["_noise"]
