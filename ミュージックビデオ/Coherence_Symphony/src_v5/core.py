# -*- coding: utf-8 -*-
"""地球の内部（断面）: 4分の1を切り取った地球。核が鼓動し、波が地表・宇宙へ広がる"""
import math
import numpy as np
import cv2

W, H = 1920, 1080


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)


class Core:
    def __init__(self, scene):
        self.sc = scene
        self.tex = scene.earth_tex
        rng = np.random.default_rng(31)
        from mv_core import fft_noise
        self.n1 = fft_noise(512, 512, 2.4, 41)
        self.n2 = fft_noise(512, 512, 2.0, 42)
        self.cache = {}

    def grid(self, cx, cy, R):
        key = (int(cx), int(cy), int(R))
        if key in self.cache:
            return self.cache[key]
        x0 = max(int(cx - R * 1.9), 0); x1 = min(int(cx + R * 1.9), W)
        y0 = max(int(cy - R * 1.9), 0); y1 = min(int(cy + R * 1.9), H)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        g = ((xx + 0.5 - cx) / R, -(yy + 0.5 - cy) / R, (x0, x1, y0, y1))
        if len(self.cache) > 8:
            self.cache.clear()
        self.cache[key] = g
        return g

    def render(self, t, P, cx=960, cy=500, R=360, yaw=-0.42, pitch=0.32, cut=1.0, pulse=1.0, waves=1.0,
               rise=0.0, glow=1.0):
        """線形RGBの画像を返す（背景は透明=黒、宇宙背景は別途）"""
        img = np.zeros((H, W, 3), np.float32)
        x, y, (x0, x1, y0, y1) = self.grid(cx, cy, R)
        d2 = x * x + y * y
        inside = d2 < 1
        zf = np.sqrt(np.clip(1 - d2, 0, 1))
        # 視点空間 → 地球ローカル（yaw, pitch）
        M = rot_x(pitch) @ rot_y(yaw)  # local = M^T view ... view = M local
        Mt = M.T
        pv = np.stack([x, y, zf], -1)
        pl = pv @ M  # = M^T p
        # 切り取る範囲: ローカルで x>0 かつ z>0（手前の4分の1）
        n1 = np.array([1, 0, 0], np.float32) @ Mt.T  # 平面法線（視点空間）
        n1 = (M @ np.array([1, 0, 0], np.float32)); n2 = (M @ np.array([0, 0, 1], np.float32))
        removed = inside & (pl[..., 0] > 0) & (pl[..., 2] > 0) & (cut > 0.5)
        # 光線 p(s) = (x, y, zf - s)
        def plane_s(n):
            # dot((x,y,zf-s), n) = 0 → s = (x n0 + y n1 + zf n2)/n2
            with np.errstate(divide="ignore", invalid="ignore"):
                return (x * n[0] + y * n[1] + zf * n[2]) / n[2]
        s1 = plane_s(n1); s2 = plane_s(n2)
        s1 = np.where(s1 > 0, s1, np.inf); s2 = np.where(s2 > 0, s2, np.inf)
        use1 = s1 < s2
        s = np.minimum(s1, s2)
        s = np.where(np.isfinite(s), s, 0.0).astype(np.float32)
        px, py, pz = x, y, zf - s
        rr = np.sqrt(px * px + py * py + pz * pz)
        face = removed & (rr <= 1.0)
        # 断面上の座標（ノイズ用）
        p_loc = np.stack([px, py, pz], -1) @ M
        ang = np.arctan2(p_loc[..., 1], np.where(use1, p_loc[..., 2], p_loc[..., 0]))
        beat = P["beat"]; bass = P["bass"]
        # 層
        r = np.clip(rr, 0, 1)
        ic = r < 0.19; oc = (r >= 0.19) & (r < 0.55); lm = (r >= 0.55) & (r < 0.89); um = (r >= 0.89) & (r < 0.985)
        crust = r >= 0.985
        flow = t * 0.15
        nu = ((ang / (2 * np.pi) + 0.5) * 512 + flow * 80 / (r + 0.3)).astype(np.float32)
        nv = (r * 512 * 0.9).astype(np.float32)
        nz1 = cv2.remap(self.n1, nu, nv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        nz2 = cv2.remap(self.n2, nu * 1.7, nv * 1.3 + t * 20, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        hb = pulse * (0.6 + 0.8 * beat * (0.5 + bass))  # 鼓動
        col = np.zeros(r.shape + (3,), np.float32)
        col += ic[..., None] * (np.array([1.6, 1.35, 0.9]) * (1.2 + 1.2 * hb) * (1.2 - r / 0.19 * 0.4)[..., None])
        oc_t = ((r - 0.19) / 0.36)[..., None]
        col += oc[..., None] * ((np.array([1.4, 0.75, 0.2]) * (1 - oc_t) + np.array([1.0, 0.38, 0.08]) * oc_t) * (0.7 + 0.6 * nz1[..., None]) * (1 + 0.5 * hb))
        lm_t = ((r - 0.55) / 0.34)[..., None]
        col += lm[..., None] * ((np.array([0.85, 0.25, 0.06]) * (1 - lm_t) + np.array([0.45, 0.12, 0.05]) * lm_t) * (0.55 + 0.7 * nz2[..., None]))
        col += um[..., None] * (np.array([0.32, 0.12, 0.06]) * (0.6 + 0.6 * nz2[..., None]))
        col += crust[..., None] * np.array([0.25, 0.2, 0.16])
        # 境界の光
        for rb, cc in ((0.19, (1.5, 1.2, 0.7)), (0.55, (1.2, 0.6, 0.2)), (0.89, (0.6, 0.25, 0.1))):
            col += (np.exp(-((r - rb) / 0.006) ** 2))[..., None] * np.array(cc) * 0.6
        # 核から広がる波
        if waves > 0.01:
            period = 60 / 128.5714 * 2
            for k in range(3):
                age = ((t + k * period / 3) % period) / period
                rw = age * 1.25
                col += (np.exp(-((r - rw) / 0.03) ** 2) * (1 - age) ** 1.5 * waves)[..., None] * np.array([1.4, 1.1, 0.6])
        # 平面の陰影
        L = np.array([-0.4, 0.5, 0.75], np.float32); L /= np.linalg.norm(L)
        nf = np.where(use1[..., None], n1[None, None], n2[None, None])
        nf = nf * np.sign((nf * np.array([0, 0, 1.0])).sum(-1, keepdims=True) + 1e-6)
        sh = np.clip((nf * L).sum(-1), 0, 1) * 0.4 + 0.75
        col *= sh[..., None]
        # 地表（切り取られていない部分）
        surf = inside & ~removed
        nw_local = pl  # ローカル座標の法線 → 地球回転を加える
        nl = nw_local @ rot_y(t * 0.05)
        lat = np.arcsin(np.clip(nl[..., 1], -1, 1)); lon = np.arctan2(nl[..., 0], nl[..., 2])
        th_, tw_ = self.tex.shape[:2]
        mx = ((lon / (2 * np.pi) + 0.5) * tw_).astype(np.float32); my = ((0.5 - lat / np.pi) * th_).astype(np.float32)
        tex = cv2.remap(self.tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        ndl = np.clip(pv @ L, 0, 1)
        stex = tex * (0.15 + 1.0 * ndl[..., None])
        rim = (1 - zf) ** 2.5
        stex += rim[..., None] * np.array([0.3, 0.6, 1.0]) * 0.7
        out = np.where(face[..., None], col, 0) + np.where(surf[..., None], stex, 0)
        # 切り口の縁で欠けた背景（宇宙）が見える部分は黒
        a = np.clip((1 - np.sqrt(d2)) * R + 0.5, 0, 1)[..., None]
        sub = img[y0:y1, x0:x1]
        sub += out * a
        # 外側の波（地表→宇宙へ）
        dpx = np.sqrt(d2)
        if waves > 0.01 or rise > 0.01:
            period = 60 / 128.5714 * 2
            ring = np.zeros_like(dpx)
            for k in range(3):
                age = ((t + k * period / 3) % period) / period
                rw = 1.0 + age * 0.9
                ring += np.exp(-((dpx - rw) / 0.012) ** 2) * (1 - age) ** 2
            outside = dpx > 1.0
            sub += (ring * outside * (0.4 * waves + 0.8 * rise))[..., None] * np.array([1.0, 0.8, 0.5], np.float32)
        halo = np.exp(-np.maximum(dpx - 1, 0) / 0.05) * (dpx > 0.98)
        sub += halo[..., None] * np.array([0.3, 0.55, 1.0], np.float32) * 0.6 * glow
        return img
