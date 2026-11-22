# -*- coding: utf-8 -*-
"""
Coherence Symphony - Music Video renderer
テーマ「波動による全宇宙の調和」
地球を中心に、銀河・惑星・星々が周波数（波動）によってひとつに調和していく。

numpy + OpenCV で 1920x1080 / 30fps のフレームを生成し ffmpeg に流す。
使い方:
  python3 mv.py still 10 85 150        # 指定秒の静止画を preview/ に出力
  python3 mv.py render out.mp4         # 全編を並列レンダリングして音声と合成
"""
import os, sys, math, bisect, subprocess, time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.environ.get("MV_ASSETS", os.path.join(HERE, "..", "assets"))
CITIES = [(35.7, 139.7), (39.9, 116.4), (28.6, 77.2), (19.1, 72.9), (-6.2, 106.8), (-33.9, 151.2), (25.2, 55.3),
          (30.0, 31.2), (-1.3, 36.8), (6.5, 3.4), (-26.2, 28.0), (41.0, 29.0), (55.8, 37.6), (48.9, 2.35),
          (51.5, -0.1), (64.1, -21.9), (40.7, -74.0), (41.9, -87.6), (34.1, -118.2), (19.4, -99.1), (4.7, -74.1),
          (-12.0, -77.0), (-23.6, -46.6), (-34.6, -58.4), (21.3, -157.9), (37.6, 127.0), (14.6, 121.0),
          (13.8, 100.5), (35.7, 51.4), (9.0, 38.7), (-4.3, 15.3), (61.2, -149.9), (49.3, -123.1), (-33.4, -70.6),
          (-36.8, 174.8), (-31.95, 115.86), (1.35, 103.8), (31.2, 121.5), (43.7, -79.4), (59.3, 18.1)]
AUDIO = os.environ.get("MV_AUDIO", "")
sys.path.insert(0, HERE)
from lyrics import CARDS

W, H = 1920, 1080
HW, HH = W // 2, H // 2
FPS = 30
F = 1500.0                 # 焦点距離 (px)
CX, CY = W / 2, H / 2 - 50  # 主点（地球を少し上に置いてテロップと重ならないように）

# ---------------------------------------------------------------- helpers
def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)

def kf(t, keys):
    ts = [k[0] for k in keys]
    if t <= ts[0]:
        return float(keys[0][1])
    if t >= ts[-1]:
        return float(keys[-1][1])
    i = bisect.bisect_right(ts, t) - 1
    (t0, v0), (t1, v1) = keys[i], keys[i + 1]
    u = float(smooth((t - t0) / (t1 - t0)))
    return v0 + (v1 - v0) * u

def spline(t, keys):
    """時間を考慮した Catmull-Rom（カメラ用、キーで止まらない）"""
    ts = [k[0] for k in keys]; vs = [k[1] for k in keys]; n = len(ts)
    if t <= ts[0]:
        return vs[0]
    if t >= ts[-1]:
        return vs[-1]
    i = bisect.bisect_right(ts, t) - 1
    i0, i2, i3 = max(i - 1, 0), i + 1, min(i + 2, n - 1)
    h = ts[i2] - ts[i]
    m1 = (vs[i2] - vs[i0]) / max(ts[i2] - ts[i0], 1e-6) * h
    m2 = (vs[i3] - vs[i]) / max(ts[i3] - ts[i], 1e-6) * h
    u = (t - ts[i]) / h
    h00 = 2 * u**3 - 3 * u**2 + 1; h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2; h11 = u**3 - u**2
    return h00 * vs[i] + h10 * m1 + h01 * vs[i2] + h11 * m2

def hsv2rgb(h, s, v):
    h = (h % 1.0) * 6
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p = v * (1 - s); q = v * (1 - s * f); tt = v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, tt, v]); g = np.choose(i, [tt, v, v, q, p, p])
    b = np.choose(i, [p, p, tt, v, v, q])
    return np.stack([r, g, b], -1).astype(np.float32)

def fft_noise(h, w, beta, seed):
    """周期的（タイル可能）な 1/f ノイズ 0..1"""
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.rfftfreq(w)[None, :]
    f = np.sqrt(fx**2 + fy**2); f[0, 0] = 1.0
    amp = f ** (-beta / 2.0); amp[0, 0] = 0
    spec = amp * np.exp(2j * np.pi * rng.random(amp.shape))
    n = np.fft.irfft2(spec, s=(h, w))
    n = (n - n.min()) / (n.max() - n.min())
    return n.astype(np.float32)

def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)

def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)

# ---------------------------------------------------------------- timeline
# 構成（128.6BPM）: 0 静寂 / 8.5 Intro歌 / 24 ビート開始 / 38 Verse1 / 68 Pre / 82 Chorus1
# 108 間奏 / 113 Verse2 / 128 Pre2 / 141 Chorus2 / 168 Bridge / 186 静かな部分 / 192 盛り上げ
# 200 Final Chorus / 229 Outro歌 / 244 インスト終盤 / 265.6 曲終わり
CAM_D = [(0, 130), (8.5, 95), (23.5, 44), (38, 25), (52, 21), (67, 17.5), (82, 10.5), (95, 8.8),
         (108, 9.2), (113, 13), (128, 10.8), (141, 9.6), (155, 15), (168, 19), (180, 14.5),
         (186, 5.2), (192, 4.7), (200, 17), (214, 20), (228, 21), (243, 24), (255, 25), (261, 29), (265.5, 52), (268, 56)]
CAM_EL = [(0, 0.22), (23.5, 0.3), (38, 0.38), (67, 0.45), (82, 0.5), (108, 0.62), (113, 0.36),
          (128, 0.42), (141, 0.5), (168, 0.62), (180, 0.92), (186, 0.26), (192, 0.22), (200, 1.12),
          (228, 1.18), (243, 0.72), (255, 0.9), (268, 1.02)]
CAM_AZ = [(0, 0), (38, 0.3), (82, 0.6), (113, 1.4), (141, 1.8), (186, 2.6), (200, 3.4), (244, 4.0), (268, 4.6)]

COH = [(0, 0), (67, 0), (81.5, 0.6), (83, 1.0), (108, 1.0), (112, 0.45), (127, 0.45), (129, 0.9),
       (140, 0.9), (141.5, 1), (168, 1), (170, 0.9), (186, 0.95), (200, 1), (268, 1)]
RING = [(0, 0), (67, 0), (82, 1), (108, 1), (112, 0.35), (127, 0.35), (140, 0.9), (142, 1), (268, 1)]
HEART = [(0, 0), (126, 0), (128.6, 7.5), (139.8, 0)]
MAND = [(0, 0), (196, 0), (202, 6.5), (228, 6.5), (232, 0), (244, 0), (247, 7.0), (256, 5.0), (265, 8.0)]
GALC = [(0, 0), (141, 0), (162, 1), (268, 1)]
INCL = [(0, 0), (192, 0), (203, 1), (228, 1), (233, 0.35), (243, 0.35), (250, 1), (268, 1)]
WAVE = [(0, 0), (18, 0), (30, 0.5), (38, 0.7), (67, 0.7), (82, 1), (108, 1), (110, 0.5), (113, 0.7),
        (141, 1), (168, 1), (172, 0.85), (186, 0.85), (187, 0.25), (192, 0.25), (200, 1), (228, 1),
        (232, 0.6), (243, 0.6), (246, 1.2), (265, 1.2), (267.5, 0)]
SHELL = [(0, 0.3), (23, 0.35), (24, 0.5), (38, 0.35), (67, 0.45), (82, 1), (108, 1), (110, 0.4),
         (141, 1), (168, 1), (170, 0.6), (186, 0.6), (187, 0.3), (192, 0.45), (200, 1.1), (228, 1.1),
         (230, 0.4), (243, 0.4), (245, 1.3), (265.5, 1.3), (266, 0)]
SPEC = [(0, 0), (80, 0), (83, 1), (108, 1), (111, 0), (140, 0), (142, 1), (168, 1), (170, 0),
        (199, 0), (201, 1), (228, 1), (231, 0), (244, 0), (246, 1), (265.5, 1), (266.5, 0)]
PLAN = [(0, 0), (20, 0), (30, 1), (268, 1)]
WARM = [(0, 0), (167, 0), (172, 1), (190, 1), (197, 0)]
GOLD = [(0, 0), (196, 0), (202, 0.6), (268, 0.8)]
NEB = [(0, 0.25), (8, 0.45), (24, 0.9), (268, 1)]
FADE = [(0, 0), (1.2, 1), (265.6, 1), (267.5, 0)]
RAYS = [(0, 0), (199, 0), (201, 0.6), (228, 0.6), (231, 0), (244, 0), (246, 1), (265.5, 1), (266, 0)]
STREAK = [(0, 0), (193, 0), (197.5, 0.6), (199.9, 1), (200.25, 0), (268, 0)]
IMPACTS = [24.0, 82.4, 141.2, 200.0, 244.4]
HEYS = [217.1, 221.0]


class Feat:
    def __init__(self):
        d = np.load(os.path.join(ASSETS, "audio_feat.npz"))
        self.n = d["bass"].shape[0]
        b = d["bass"]; self.bass = np.clip(b / np.percentile(b, 95), 0, 1.6)
        r = d["rms"]; self.rms = np.clip(r / np.percentile(r, 95), 0, 1.6)
        hi = d["high"]; self.high = np.clip(hi / np.percentile(hi, 95), 0, 1.6)
        self.mel = d["mel"]  # (64, n) dB
        self.beats = d["beats"]
        # 平滑化したエネルギー
        k = np.ones(9) / 9
        self.bass_s = np.convolve(self.bass, k, mode="same")
        self.rms_s = np.convolve(self.rms, np.ones(31) / 31, mode="same")

    def at(self, arr, t):
        i = int(np.clip(round(t * FPS), 0, self.n - 1))
        return float(arr[i])

    def beat_env(self, t, tau=0.16):
        j = bisect.bisect_right(self.beats, t) - 1
        if j < 0:
            return 0.0
        return math.exp(-(t - self.beats[j]) / tau)

    def mel_at(self, t):
        i = int(np.clip(round(t * FPS), 0, self.n - 1))
        lo = max(i - 1, 0)
        return self.mel[:, lo:i + 2].mean(1)


# ---------------------------------------------------------------- scene
class Scene:
    def __init__(self):
        self.feat = Feat()
        rng = np.random.default_rng(7)
        self._init_sky(rng)
        self._init_souls(rng)
        self._init_galaxies(rng)
        self._init_planets(rng)
        self._init_sources(rng)
        self._init_shells()
        self._init_ripples(rng)
        self._init_grid()
        self.telop = Telop()

    # ---- sky
    def _init_sky(self, rng):
        n = 16000
        v = rng.normal(size=(n, 3)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        # 天の川の帯に星を集める
        nb = np.array([0.35, 0.88, 0.32], np.float32); nb /= np.linalg.norm(nb)
        m = n // 3
        band = rng.normal(size=(m, 3)).astype(np.float32)
        band -= (band @ nb)[:, None] * nb[None]
        band /= np.linalg.norm(band, axis=1, keepdims=True)
        band += nb[None] * rng.normal(0, 0.09, size=(m, 1)).astype(np.float32)
        band /= np.linalg.norm(band, axis=1, keepdims=True)
        v[:m] = band
        self.star_dir = v
        mag = rng.pareto(2.2, n).astype(np.float32) * 0.18 + 0.05
        self.star_b = np.clip(mag, 0, 3.0)
        temp = rng.random(n)
        col = np.where(temp[:, None] < 0.15, [1.0, 0.75, 0.55],
                       np.where(temp[:, None] < 0.4, [0.7, 0.82, 1.0], [0.95, 0.95, 1.0])).astype(np.float32)
        self.star_col = col
        self.star_tw = rng.random(n).astype(np.float32) * 6.28
        self.star_tf = (0.5 + rng.random(n) * 2.5).astype(np.float32)
        self.galnorm = nb
        # 星雲テクスチャ（周期ノイズ）
        cache = os.path.join(ASSETS, "nebula_v4.npz")
        if os.path.exists(cache):
            c = np.load(cache)
            self.neb_a, self.neb_b, self.neb_f, self.clouds = c["a"], c["b"], c["f"], c["clouds"]
        else:
            hh, ww = 1024, 2048
            m1 = fft_noise(hh, ww, 3.8, 11); m2 = fft_noise(hh, ww, 3.8, 12)
            d1 = fft_noise(hh, ww, 3.0, 1); d2 = fft_noise(hh, ww, 2.8, 2); d3 = fft_noise(hh, ww, 2.8, 3)
            M1 = smooth((m1 - 0.52) / 0.3); M2 = smooth((m2 - 0.56) / 0.3)
            ridged = 1 - np.abs(2 * d3 - 1)
            self.neb_a = (M1 * (0.25 + 1.2 * np.clip(d1 - 0.3, 0, 1) ** 1.5)).astype(np.float32)
            self.neb_b = (M2 * (0.2 + 1.3 * np.clip(d2 - 0.35, 0, 1) ** 1.5)).astype(np.float32)
            self.neb_f = (np.maximum(M1, M2) * ridged ** 6).astype(np.float32)
            cl = fft_noise(1024, 2048, 2.7, 4)
            self.clouds = (np.clip((cl - 0.5) * 2.4, 0, 1) ** 1.2).astype(np.float32)
            np.savez(cache, a=self.neb_a, b=self.neb_b, f=self.neb_f, clouds=self.clouds)

    # ---- particles that gather into harmonic rings
    def _init_souls(self, rng):
        n = 14000
        shells = np.arange(9)
        p = (shells + 2.0); p /= p.sum()
        sh = rng.choice(shells, size=n, p=p)
        self.s_shell = sh
        self.s_R = (2.4 + 1.5 * sh + rng.normal(0, 0.07, n)).astype(np.float32)
        self.s_th0 = (rng.random(n) * 2 * np.pi).astype(np.float32)
        self.s_om = (0.55 / np.sqrt(2.4 + 1.5 * sh)).astype(np.float32)
        self.s_y = rng.normal(0, 0.04, n).astype(np.float32)
        d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        r = 6 + 34 * rng.random(n) ** 0.8
        home = d * r[:, None]; home[:, 1] *= 0.5
        self.s_home = home.astype(np.float32)
        self.s_df = (0.05 + rng.random((n, 3)) * 0.2).astype(np.float32) * 2 * np.pi
        self.s_dp = (rng.random((n, 3)) * 2 * np.pi).astype(np.float32)
        self.s_delay = (rng.random(n) * 0.5).astype(np.float32)
        self.s_b = (np.exp(rng.normal(0, 0.5, n)) * 0.45).astype(np.float32)
        self.s_chaos_col = hsv2rgb(rng.random(n), 0.75, 1.0)
        u = (sh / 8.0)[:, None]
        gold = np.array([1.0, 0.78, 0.42]); cyan = np.array([0.45, 0.82, 1.0]); vio = np.array([0.72, 0.55, 1.0])
        hc = np.where(u < 0.6, gold * (1 - u / 0.6) + cyan * (u / 0.6), cyan * (1 - (u - 0.6) / 0.4) + vio * ((u - 0.6) / 0.4))
        self.s_harm_col = hc.astype(np.float32)
        self.s_tw = (rng.random(n) * 2 * np.pi).astype(np.float32)
        self.s_tf = (1 + rng.random(n) * 3).astype(np.float32)
        # 傾いたリング（アーミラリ天球のように）
        self.shell_incl = np.array([0.0, 1.1, -0.9, 0.7, -1.25, 0.5, 1.35, -0.6, 0.95], np.float32)
        self.shell_node = (np.arange(9) * 2.39996).astype(np.float32)

    def _init_galaxies(self, rng):
        self.gals = []
        centers = [(38, 10, -30), (-46, -6, -18), (20, -14, 44), (-28, 17, 36)]
        radii = [15, 13, 16, 12]
        targets = [17.5, 20.5, 23.5, 26.5]
        for gi, (c, R, Rt) in enumerate(zip(centers, radii, targets)):
            n = 4000
            nrm = rng.normal(size=3); nrm /= np.linalg.norm(nrm)
            e1 = np.cross(nrm, [0, 1, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(nrm, e1)
            arm = rng.integers(0, 2, n)
            rr = np.clip(rng.exponential(0.32, n), 0.02, 1.2)
            th = arm * np.pi + np.log(rr + 0.05) / math.tan(0.35) + rng.normal(0, 0.28, n)
            bulge = rng.random(n) < 0.18
            rr[bulge] = np.abs(rng.normal(0, 0.09, bulge.sum()))
            th[bulge] = rng.random(bulge.sum()) * 2 * np.pi
            z = rng.normal(0, 0.03, n) * (1 + 2 * bulge)
            col = np.tile(np.array([0.6, 0.75, 1.0], np.float32), (n, 1))
            col[bulge] = [1.0, 0.85, 0.6]
            knots = (~bulge) & (rng.random(n) < 0.06)
            col[knots] = [1.0, 0.45, 0.72]
            inner = (~bulge) & (rr < 0.25)
            col[inner] = [1.0, 0.9, 0.75]
            b = (np.exp(rng.normal(0, 0.45, n)) * 0.3 * (1 + 1.5 * bulge)).astype(np.float32)
            self.gals.append(dict(c=np.array(c, np.float32), R=R, Rt=Rt, e1=e1.astype(np.float32),
                                  e2=e2.astype(np.float32), nrm=nrm.astype(np.float32), r=rr.astype(np.float32),
                                  th=th.astype(np.float32), z=z.astype(np.float32), col=col, b=b,
                                  om=(0.07 / (rr + 0.25)).astype(np.float32),
                                  delay=(rng.random(n) * 0.5).astype(np.float32), gi=gi))

    def _init_planets(self, rng):
        specs = [
            # name, orbit, radius, tex-colors, ring
            ("moon", 1.9, 0.22, [(0.55, 0.55, 0.58), (0.38, 0.38, 0.4), (0.7, 0.7, 0.72)], False),
            ("venus", 3.15, 0.36, [(0.95, 0.85, 0.6), (0.85, 0.7, 0.45), (1.0, 0.92, 0.75)], False),
            ("mars", 4.65, 0.30, [(0.78, 0.36, 0.2), (0.55, 0.22, 0.12), (0.9, 0.55, 0.35)], False),
            ("jupiter", 7.6, 0.78, [(0.88, 0.75, 0.6), (0.7, 0.45, 0.3), (0.98, 0.92, 0.82)], False),
            ("saturn", 10.6, 0.6, [(0.92, 0.84, 0.62), (0.78, 0.66, 0.45), (1.0, 0.95, 0.8)], True),
            ("neptune", 13.6, 0.46, [(0.3, 0.5, 0.95), (0.2, 0.35, 0.8), (0.55, 0.75, 1.0)], False),
        ]
        self.planets = []
        for i, (name, a, rad, cols, ring) in enumerate(specs):
            tex = self._planet_tex(rng, cols, banded=name in ("jupiter", "saturn", "neptune"), seed=10 + i)
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            chaos = d * (18 + 12 * rng.random()); chaos[1] *= 0.5
            p = dict(name=name, a=a, rad=rad, tex=tex, ring=ring, ph=rng.random() * 6.28,
                     om=0.45 / a ** 0.8, chaos=chaos.astype(np.float32), delay=rng.random() * 0.4,
                     spin=0.2 + rng.random() * 0.3, tilt=rng.normal(0, 0.3))
            if ring:
                m = 3000
                rr = 1.45 + rng.random(m) * 0.9
                rr = rr[(rr < 1.95) | (rr > 2.03)]  # カッシーニの間隙
                th = rng.random(rr.size) * 2 * np.pi
                p["ring_r"] = rr.astype(np.float32); p["ring_th"] = th.astype(np.float32)
                p["ring_b"] = (0.12 + rng.random(rr.size) * 0.1).astype(np.float32)
            self.planets.append(p)
        # 地球テクスチャ
        cache = os.path.join(ASSETS, "earth_tex_hi.npy")
        if os.path.exists(cache):
            self.earth_tex = np.load(cache)
        else:
            im = cv2.imread(os.path.join(ASSETS, "bmng.jpg"))[:, :, ::-1]
            im = im.astype(np.float32) / 255
            hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV); hsv[..., 1] = np.clip(hsv[..., 1] * 1.15, 0, 1)
            im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            self.earth_tex = (im ** 0.9).astype(np.float32)
            np.save(cache, self.earth_tex)
        tex = self.earth_tex
        self.earth_ocean = np.clip((tex[..., 2] - np.maximum(tex[..., 0], tex[..., 1])) * 6, 0, 1).astype(np.float32)
        ncache = os.path.join(ASSETS, "night_tex.npy")
        if os.path.exists(ncache):
            self.night_tex = np.load(ncache)
        else:
            small = cv2.resize(tex, (2700, 1350), interpolation=cv2.INTER_AREA)
            oc = cv2.resize(self.earth_ocean, (2700, 1350), interpolation=cv2.INTER_AREA)
            br = small.mean(2); sat = small.max(2) - small.min(2)
            land = (oc < 0.3).astype(np.float32)
            ice = ((br > 0.6) & (sat < 0.12)).astype(np.float32)
            desert = np.clip((small[..., 0] - small[..., 2] - 0.18) * 5, 0, 1)
            yy = np.linspace(90, -90, 1350)[:, None]
            latw = np.clip(np.cos(np.radians(yy)) * 1.4 - 0.3, 0, 1)
            n1 = fft_noise(1350, 2700, 1.6, 21); n2 = fft_noise(1350, 2700, 0.6, 22)
            dens = land * (1 - ice) * (1 - 0.8 * desert) * latw * np.clip((n1 - 0.45) * 3, 0, 1)
            lights = dens * np.clip((n2 - 0.5) * 6, 0, 1)
            for (la, lo) in CITIES:
                x = int((lo / 360 + 0.5) * 2700); y = int((0.5 - la / 180) * 1350)
                cv2.circle(lights, (x, y), 6, 1.0, -1)
            lights = cv2.GaussianBlur(lights.astype(np.float32), (0, 0), 1.2) * 2.5
            self.night_tex = np.clip(lights, 0, 2).astype(np.float32)
            np.save(ncache, self.night_tex)

    def _planet_tex(self, rng, cols, banded, seed):
        h, w = 256, 512
        nz = fft_noise(h, w, 2.2, seed)
        c0, c1, c2 = [np.array(c, np.float32) for c in cols]
        if banded:
            lat = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
            nz2 = fft_noise(h, w, 2.8, seed + 100)
            band = 0.5 + 0.5 * np.sin(lat * (5 + seed % 4) * np.pi + 5.0 * (nz2 - 0.5) + seed)
            band = 0.25 + 0.5 * band + 0.25 * nz
        else:
            band = np.clip((nz - 0.35) * 2.5, 0, 1)
        t = band[..., None]
        tex = c1 * (1 - t) + c0 * t
        tex = tex + (c2 - tex) * np.clip(nz - 0.7, 0, 1)[..., None] * 2
        return np.clip(tex, 0, 1).astype(np.float32)

    def _init_sources(self, rng):
        n = 6
        ang = rng.random(n) * 2 * np.pi
        rad = 6 + rng.random(n) * 15
        self.src_chaos = np.stack([np.cos(ang) * rad, np.sin(ang) * rad], 1).astype(np.float32)
        self.src_k = (1.2 + rng.random(n) * 2.6).astype(np.float32)
        self.src_w = (1.5 + rng.random(n) * 3.0).astype(np.float32)
        self.src_ph = (rng.random(n) * 2 * np.pi).astype(np.float32)
        self.src_col = np.array([[1.0, 0.35, 0.45], [1.0, 0.68, 0.25], [0.35, 1.0, 0.6],
                                 [0.25, 0.7, 1.0], [0.62, 0.38, 1.0], [1.0, 0.42, 0.9]], np.float32)
        self.K0, self.W0 = 2.1, 2.6
        self.heart_sign = np.array([-1, -1, -1, 1, 1, 1], np.float32)

    def _init_shells(self):
        f = self.feat
        self.shell_list = []
        for j, b in enumerate(f.beats):
            amp = kf(b, SHELL)
            every = 1 if amp >= 0.7 else (2 if amp >= 0.4 else 4)
            if j % every:
                continue
            strength = amp * (0.35 + 0.8 * f.at(f.bass_s, b))
            if b < 23:
                strength = amp * 0.9
            self.shell_list.append((b, strength))
        self.shell_t = np.array([s[0] for s in self.shell_list])

    def _init_ripples(self, rng):
        ev = []
        for (t0, t1, n) in [(168.5, 186.0, 70)]:
            ts = np.sort(t0 + rng.random(n) * (t1 - t0))
            for t in ts:
                a = rng.random() * 2 * np.pi; r = 3 + rng.random() * 22
                ev.append((t, math.cos(a) * r, math.sin(a) * r, 0.6 + rng.random() * 0.6))
        self.ripples = ev
        self.ripple_t = np.array([e[0] for e in ev])

    def _init_grid(self):
        xs = ((np.arange(HW) + 0.5) * 2 - CX) / F
        ys = ((np.arange(HH) + 0.5) * 2 - CY) / F
        self.gx, self.gy = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        self.vignette = (1 - 0.38 * np.clip(r / 1.3, 0, 1) ** 2.2).astype(np.float32)[..., None]

    # ------------------------------------------------------------ camera
    def camera(self, t):
        D = math.exp(spline(t, [(a, math.log(b)) for a, b in CAM_D]))
        el = spline(t, CAM_EL)
        az = 0.035 * t + spline(t, CAM_AZ)
        roll = 0.06 * math.sin(t * 0.045) + 0.03 * math.sin(t * 0.13)
        C = np.array([D * math.cos(el) * math.sin(az), D * math.sin(el), D * math.cos(el) * math.cos(az)], np.float32)
        fwd = -C / np.linalg.norm(C)
        right = np.cross(fwd, np.array([0, 1, 0], np.float32)); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        cr, sr = math.cos(roll), math.sin(roll)
        right, up = right * cr + up * sr, up * cr - right * sr
        return dict(C=C, fwd=fwd.astype(np.float32), right=right.astype(np.float32), up=up.astype(np.float32), D=D)

    def project(self, cam, P):
        v = P - cam["C"][None]
        z = v @ cam["fwd"]; x = v @ cam["right"]; y = v @ cam["up"]
        zz = np.maximum(z, 1e-3)
        return CX + F * x / zz, CY - F * y / zz, z

    # ------------------------------------------------------------ main render
    def render(self, t):
        img, P = self.render_linear(t)
        return self.post(img, P, t)

    def render_linear(self, t, cam_fn=None, over=None):
        f = self.feat
        cam = cam_fn(t) if cam_fn else self.camera(t)
        P = dict(coh=kf(t, COH), ring=kf(t, RING), heart=kf(t, HEART), mand=kf(t, MAND), galc=kf(t, GALC),
                 incl=kf(t, INCL), wave=kf(t, WAVE), spec=kf(t, SPEC), plan=kf(t, PLAN), warm=kf(t, WARM),
                 gold=kf(t, GOLD), neb=kf(t, NEB), fade=kf(t, FADE), streak=kf(t, STREAK), rays=kf(t, RAYS))
        bass = f.at(f.bass_s, t); beat = f.beat_env(t); rms = f.at(f.rms_s, t)
        P["bass"], P["beat"], P["rms"] = bass, beat, rms
        imp = sum(math.exp(-(t - ti) / 0.45) for ti in IMPACTS if t >= ti)
        hey = sum(math.exp(-(t - ti) / 0.3) for ti in HEYS if t >= ti)
        P["flash"] = min(imp + hey * 0.8, 1.5)
        P.setdefault("night", 0.0); P.setdefault("net", 0.0); P.setdefault("kind", 0.0); P.setdefault("spin", 0.0)
        P.setdefault("light", (-0.45, 0.38, 0.81)); P.setdefault("cities", 0.0); P.setdefault("netsync", 0.0)
        P.setdefault("netstart", 0); P.setdefault("rayk", 1.0); P.setdefault("pvis", 1.0); P.setdefault("halo", 1.0)
        if over:
            for k, v in over.items():
                P[k] = v(t) if callable(v) else v

        # earth on screen
        eu, ev, ez = self.project(cam, np.zeros((1, 3), np.float32))
        eu, ev, ez = float(eu[0]), float(ev[0]), float(ez[0])
        if ez > 1.02:
            er = F / math.sqrt(max(ez * ez - 1, 1e-3))
        else:
            eu, ev, er = -1e5, -1e5, 0.5
        P["earth"] = (eu, ev, ez, er)
        self._cam = cam

        back = np.zeros((HH, HW, 3), np.float32)
        front = np.zeros((HH, HW, 3), np.float32)
        SKIP = os.environ.get("MV_SKIP", "").split(",")
        if "neb" not in SKIP: self.draw_nebula(back, cam, P, t)
        if "waves" not in SKIP: self.draw_waves(back, front, cam, P, t)
        if "shells" not in SKIP: self.draw_shells(front, cam, P, t)
        if "halo" not in SKIP and P["halo"] > 0.01: self.draw_halo(front, P, t)

        full = np.zeros((H, W, 3), np.float32)
        self.draw_stars(full, cam, P, t)

        # bodies (opaque)
        bodies = self.collect_bodies(cam, P, t)
        brgb = np.zeros((H, W, 3), np.float32); balpha = np.zeros((H, W, 1), np.float32)
        for bd in sorted(bodies, key=lambda b: -b["z"]):
            self.draw_sphere(brgb, balpha, cam, bd, P, t)

        # particles
        pfull = np.zeros((H, W, 3), np.float32)
        pglow = np.zeros((HH, HW, 3), np.float32)
        if "particles" not in SKIP and P["pvis"] > 0.01: self.draw_particles(pfull, pglow, cam, P, t, bodies)
        pglow = cv2.GaussianBlur(pglow, (0, 0), 1.6)
        front += pglow
        if P.get("galcore", 0) > 0.01:
            self.draw_galcores(front, cam, P)
        if P["rays"] > 0.01:
            front += self.rays(front, P) * (0.55 * P["rays"] * (0.6 + 0.6 * P["bass"]))

        img = cv2.resize(back, (W, H), interpolation=cv2.INTER_LINEAR) + full
        img = img * (1 - balpha) + brgb
        img += cv2.resize(front, (W, H), interpolation=cv2.INTER_LINEAR) + pfull
        if P["spec"] > 0.01:
            self.draw_spectrum(img, P, t)

        if P["net"] > 0.001 or P["cities"] > 0.001 or P["kind"] > 0.001:
            self.draw_network(img, cam, P, t)
        if P["streak"] > 0.01:
            img = self.zoom_blur(img, P)
        return img, P

    def post(self, img, P, t):
        # bloom
        small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        bright = np.maximum(small - 0.55, 0)
        b1 = cv2.GaussianBlur(bright, (0, 0), 6); b2 = cv2.GaussianBlur(bright, (0, 0), 22)
        bloom = cv2.resize(b1 * 0.55 + b2 * 0.5, (W, H), interpolation=cv2.INTER_LINEAR)
        img += bloom
        if P["flash"] > 0.01:
            img += np.array([1.0, 0.92, 0.8], np.float32) * (0.18 * P["flash"])
        # end flash
        ef = kf(t, [(264.9, 0), (265.55, 1), (266.4, 0)])
        img = 1 - np.exp(-img * 1.15)
        img = img * self.vignette
        if ef > 0:
            img = img + (1 - img) * ef * 0.85
        img *= P["fade"]
        self.telop.draw(img, t, P)
        return np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------ earth surface: cities / network
    def earth_rot(self, t, P):
        return rot_z(0.41) @ rot_y(-0.045 * t - 2.2 + P["spin"])

    def _init_cities(self):
        lat = np.radians([c[0] for c in CITIES]); lon = np.radians([c[1] for c in CITIES])
        self.city_loc = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], 1).astype(np.float32)
        n = len(CITIES)
        d = np.arccos(np.clip(self.city_loc @ self.city_loc.T, -1, 1))
        edges = set()
        for i in range(n):
            for j in np.argsort(d[i])[1:4]:
                edges.add((min(i, j), max(i, j)))
        rng = np.random.default_rng(5)
        for _ in range(14):
            i, j = rng.integers(0, n, 2)
            if i != j and d[i, j] > 0.8:
                edges.add((min(i, j), max(i, j)))
        self.edges = sorted(edges)
        self.city_d = d
        self.city_ph = rng.random(n) * 6.28
        self.city_hue = rng.random(n)

    def _bfs(self, start):
        n = len(CITIES)
        adj = [[] for _ in range(n)]
        for i, j in self.edges:
            adj[i].append(j); adj[j].append(i)
        depth = [-1] * n; depth[start] = 0; q = [start]
        while q:
            i = q.pop(0)
            for j in adj[i]:
                if depth[j] < 0:
                    depth[j] = depth[i] + 1; q.append(j)
        return depth

    def draw_network(self, img, cam, P, t):
        if not hasattr(self, "city_loc"):
            self._init_cities()
        Rm = self.earth_rot(t, P)
        C = cam["C"]
        world = self.city_loc @ Rm.T
        def vis(X):
            v = X - C[None]
            L = np.linalg.norm(v, axis=1)
            dvec = v / L[:, None]
            b = (C[None] * dvec).sum(1)
            c = (C * C).sum() - 1.0
            disc = b * b - c
            th = -b - np.sqrt(np.maximum(disc, 0))
            return ~((disc > 0) & (th > 1e-3) & (th < L - 2e-3))
        def proj(X):
            u, v, z = self.project(cam, X)
            return u, v, z
        canvas = np.zeros((H, W, 3), np.float32)
        cv8 = np.zeros((H, W), np.uint8)
        coh = P["coh"]; beat = P["beat"]
        warm = np.array([1.0, 0.78, 0.45]); gold = np.array([1.0, 0.9, 0.65])
        # 都市の光（バラバラ→同期）
        if P["cities"] > 0.001:
            vv = vis(world * 1.003)
            u, v, z = proj(world * 1.003)
            for i in np.where(vv & (z > 0))[0]:
                tw = 0.5 + 0.5 * math.sin(t * (1.5 + self.city_hue[i] * 3) + self.city_ph[i])
                sync = P["netsync"]
                a = (tw * (1 - sync) + sync * (0.6 + 0.6 * beat)) * P["cities"]
                col = hsv2rgb(np.array([self.city_hue[i]]), 0.6 * (1 - sync), 1.0)[0] * (1 - sync) + gold * sync
                r = 2.5 + 3.0 * a
                cv2.circle(canvas, (int(u[i] * 4), int(v[i] * 4)), int(r * 4), tuple(float(c) * a * 1.4 for c in col), -1, cv2.LINE_AA, 2)
                # 共鳴の輪
                age = (t * 0.8 + self.city_ph[i]) % 1.0 if sync < 0.5 else (t * 128.57 / 60 / 2) % 1.0
                rr = 6 + 40 * age
                cv2.circle(canvas, (int(u[i] * 4), int(v[i] * 4)), int(rr * 4), tuple(float(c) * a * (1 - age) * 0.8 for c in col), 2, cv2.LINE_AA, 2)
        # 都市を結ぶ光の弧
        if P["net"] > 0.001:
            depth = self._bfs(int(P["netstart"]))
            maxd = max(depth) + 1
            for (i, j) in self.edges:
                di, dj = depth[i], depth[j]
                if di > dj:
                    i, j, di, dj = j, i, dj, di
                appear = (di + 0.5) / maxd
                prog = float(np.clip((P["net"] - appear * 0.85) / 0.15 * 1.0, 0, 1))
                if prog <= 0:
                    continue
                a0, a1 = self.city_loc[i], self.city_loc[j]
                ang = self.city_d[i, j]
                m = max(8, int(ang * 40))
                s_ = np.linspace(0, prog, m)
                sin_a = math.sin(max(ang, 1e-4))
                pts = (np.sin((1 - s_) * ang)[:, None] * a0 + np.sin(s_ * ang)[:, None] * a1) / sin_a
                hgt = 1.0 + 0.04 + 0.22 * ang * np.sin(np.pi * s_)
                Xw = (pts * hgt[:, None]) @ Rm.T
                vv = vis(Xw)
                u, v, z = proj(Xw)
                ok = vv & (z > 0.05)
                col = warm * (1 - coh) + gold * coh
                inten = 0.55 + 0.5 * beat * coh
                for k in range(m - 1):
                    if ok[k] and ok[k + 1]:
                        cv2.line(cv8, (int(u[k] * 4), int(v[k] * 4)), (int(u[k + 1] * 4), int(v[k + 1] * 4)), 255, 2, cv2.LINE_AA, 2)
                # パルス
                if prog >= 1:
                    for q in range(2):
                        sp = ((t * 0.35 + q * 0.5 + (i * 7 + j * 3) * 0.13) % 1.0)
                        k = int(sp * (m - 1))
                        if ok[k]:
                            cv2.circle(canvas, (int(u[k] * 4), int(v[k] * 4)), 16, tuple(float(c) * 1.6 for c in col), -1, cv2.LINE_AA, 2)
            lines = cv8.astype(np.float32) / 255.0
            col = warm * (1 - coh) + gold * coh
            canvas += lines[..., None] * col.astype(np.float32) * (0.8 + 0.6 * beat * coh) * min(P["net"] * 3, 1)
        # 小さな優しさの波紋（地表を広がる）
        if P["kind"] > 0.001:
            rng = np.random.default_rng(11)
            n = 60
            lat = np.arcsin(rng.uniform(-0.8, 0.85, n)); lon = rng.uniform(-np.pi, np.pi, n)
            t0 = rng.random(n) * 6.0
            for k in range(n):
                age = ((t - t0[k]) % 6.0) / 6.0
                rad = 0.02 + 0.5 * age
                c0 = np.array([math.cos(lat[k]) * math.sin(lon[k]), math.sin(lat[k]), math.cos(lat[k]) * math.cos(lon[k])])
                e1 = np.cross(c0, [0, 1, 0]); e1 /= np.linalg.norm(e1) + 1e-6; e2 = np.cross(c0, e1)
                th = np.linspace(0, 2 * np.pi, 48)
                ring = (math.cos(rad) * c0[None] + math.sin(rad) * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)) * 1.004
                Xw = ring @ Rm.T
                face = (Xw * (C[None] - Xw)).sum(1) > 0
                u, v, z = proj(Xw)
                a = (1 - age) ** 1.5 * P["kind"]
                colk = np.array([1.0, 0.75, 0.6]) * (1 - P["gold"]) + gold * P["gold"]
                for q in range(47):
                    if face[q] and face[q + 1] and z[q] > 0:
                        cv2.line(canvas, (int(u[q] * 4), int(v[q] * 4)), (int(u[q + 1] * 4), int(v[q + 1] * 4)),
                                 tuple(float(c) * a for c in colk), 2, cv2.LINE_AA, 2)
        img += cv2.GaussianBlur(canvas, (0, 0), 2.5) * 1.3 + canvas * 0.8

    # ------------------------------------------------------------ layers
    def sky_dirs(self, cam):
        d = (cam["fwd"][None, None] + self.gx[..., None] * cam["right"][None, None]
             - self.gy[..., None] * cam["up"][None, None])
        d /= np.linalg.norm(d, axis=2, keepdims=True)
        return d

    def draw_nebula(self, out, cam, P, t):
        d = self.sky_dirs(cam)
        self._dirs = d
        lon = np.arctan2(d[..., 0], d[..., 2]); lat = np.arcsin(np.clip(d[..., 1], -1, 1))
        th_, tw_ = self.neb_a.shape
        mx = ((lon / (2 * np.pi) + 0.5) * tw_ + t * 1.0).astype(np.float32)
        my = ((0.5 - lat / np.pi) * th_).astype(np.float32)
        a = cv2.remap(self.neb_a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        b = cv2.remap(self.neb_b, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        fl = cv2.remap(self.neb_f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        bl = np.abs(d @ self.galnorm)
        band = np.exp(-(bl / 0.28) ** 2)
        coh, warm, gold = P["coh"], P["warm"], P["gold"]
        # パレット（混沌→調和→暖色）
        c_a = np.array([0.16, 0.08, 0.38]) * (1 - coh) + np.array([0.03, 0.13, 0.32]) * coh
        c_b = np.array([0.36, 0.06, 0.28]) * (1 - coh) + np.array([0.15, 0.06, 0.32]) * coh
        c_f = np.array([0.35, 0.5, 0.9]) * 0.35
        c_a = c_a * (1 - warm) + np.array([0.28, 0.06, 0.24]) * warm
        c_b = c_b * (1 - warm) + np.array([0.34, 0.08, 0.2]) * warm
        k = P["neb"] * (0.75 + 0.25 * P["rms"]) * (1 - 0.25 * coh)
        dens = (0.55 + 0.9 * band)
        out += (a[..., None] * c_a + b[..., None] * c_b + fl[..., None] * c_f * 0.5).astype(np.float32) * (dens * k)[..., None]
        out += (band[..., None] * np.array([0.045, 0.045, 0.075], np.float32) * k)

    def draw_waves(self, back, front, cam, P, t):
        amp = P["wave"] * (0.7 + 0.45 * P["bass"] + 0.35 * P["beat"] * P["coh"])
        self._src_pos = self.src_chaos.copy()
        if amp < 0.01:
            return
        C = cam["C"]
        dx = cam["fwd"][0] + self.gx * cam["right"][0] - self.gy * cam["up"][0]
        dy = cam["fwd"][1] + self.gx * cam["right"][1] - self.gy * cam["up"][1]
        dz = cam["fwd"][2] + self.gx * cam["right"][2] - self.gy * cam["up"][2]
        with np.errstate(divide="ignore", invalid="ignore"):
            th = -C[1] / dy
        valid = th > 0
        th = np.where(valid, th, 0).astype(np.float32)
        px = C[0] + th * dx; pz = C[2] + th * dz
        norm = np.sqrt(dx * dx + dy * dy + dz * dz)
        dist = th * norm
        rr = np.sqrt(px * px + pz * pz)
        graz = np.clip(np.abs(dy / norm) * 5.0, 0, 1)
        env = (valid * np.exp(-dist / 30.0) * smooth((rr - 1.05) / 0.6) * (1 - smooth((rr - 26) / 8)) * graz).astype(np.float32)

        coh = P["coh"]
        # 波源の配置：混沌 → 中心（地球）/ 2つの鼓動 / 曼荼羅の円
        n = 6
        ang = np.arange(n) * (2 * np.pi / n) + t * 0.05
        struct = np.stack([np.cos(ang), np.sin(ang)], 1) * P["mand"]
        struct[:, 0] += self.heart_sign * P["heart"]
        pos = self.src_chaos * (1 - coh) + struct * coh
        drift = np.stack([np.sin(t * 0.1 + np.arange(n)), np.cos(t * 0.13 + np.arange(n) * 2)], 1) * 2.0 * (1 - coh)
        pos = pos + drift
        self._src_pos = pos.astype(np.float32)
        k = self.src_k * (1 - coh) + self.K0 * coh
        w = self.src_w * (1 - coh) + self.W0 * coh
        ph = self.src_ph * (1 - coh)
        Fs = np.zeros_like(px); As = np.zeros_like(px); colr = np.zeros(px.shape + (3,), np.float32)
        indiv = np.zeros_like(px)
        mv = float(smooth(P["mand"] / 5.0)) * coh
        for i in range(n):
            d = np.sqrt((px - pos[i, 0]) ** 2 + (pz - pos[i, 1]) ** 2)
            c = np.cos(k[i] * d - w[i] * t + ph[i])
            at = 1.0 / (1 + 0.05 * d)
            Fs += c * at; As += at
            cp = np.maximum(c, 0); cp *= cp; cp *= cp
            colr += (cp * at)[..., None] * self.src_col[i]
            if mv > 0.01:
                # 円周上の波源それぞれの波紋（フラワー・オブ・ライフ状に重なる）
                ring_i = np.exp(-((d - ((w[i] * t) % (2 * np.pi)) / k[i] - np.round(d * k[i] / (2 * np.pi) - ((w[i] * t) % (2 * np.pi)) / (2 * np.pi)) * 2 * np.pi / k[i]) * k[i] * 2.2) ** 2)
                indiv += ring_i * at * np.exp(-d / 9.0)
        Fn = Fs / As
        crest = np.maximum(Fn, 0) ** 10
        u = smooth((rr - 2) / 18)[..., None]
        warm, gold = P["warm"], P["gold"]
        inner = np.array([1.0, 0.82, 0.55]) * (1 - warm) + np.array([1.0, 0.55, 0.5]) * warm
        outer = np.array([0.45, 0.8, 1.0]) * (1 - warm) + np.array([1.0, 0.7, 0.4]) * warm
        outer = outer * (1 - gold * 0.5) + np.array([1.0, 0.85, 0.6]) * gold * 0.5
        ccol = (inner * (1 - u) + outer * u).astype(np.float32)
        rgb = colr / As[..., None] * (1.3 * (1 - 0.9 * coh)) + crest[..., None] * ccol * (0.25 + 0.95 * coh)
        if mv > 0.01:
            fcol = np.array([1.0, 0.8, 0.5], np.float32) * (1 - P["gold"] * 0.3) + np.array([0.6, 0.85, 1.0], np.float32) * P["gold"] * 0.3
            rgb += (indiv / As * 2.2 * mv)[..., None] * fcol
        rgb += (0.5 + 0.5 * Fn)[..., None] ** 4 * ccol * 0.03
        # 小さな優しさの波紋（ブリッジ / 終盤）
        j0 = bisect.bisect_left(self.ripple_t, t - 4.5); j1 = bisect.bisect_right(self.ripple_t, t)
        if j1 > j0:
            rc = np.array([1.0, 0.7, 0.55]) * (1 - gold) + np.array([1.0, 0.88, 0.6]) * gold
            rip = np.zeros_like(px)
            for (t0, x0, z0, a0) in self.ripples[j0:j1]:
                age = t - t0
                d = np.sqrt((px - x0) ** 2 + (pz - z0) ** 2)
                R = age * 2.6
                rip += a0 * (1 - age / 4.5) ** 2 * np.exp(-((d - R) / (0.25 + 0.06 * R)) ** 2)
            rgb += rip[..., None] * rc.astype(np.float32) * 1.4
        rgb *= (env * amp)[..., None]
        # 地球の前後で分ける
        eu, ev, ez, er = P["earth"]
        yy = (self.gy * F + CY); xx = (self.gx * F + CX)
        inside = ((xx - eu) ** 2 + (yy - ev) ** 2) < (er * 1.02) ** 2
        frontmask = inside & (dist < ez)
        back += rgb * (~inside)[..., None]
        front += rgb * frontmask[..., None]

    def draw_shells(self, out, cam, P, t):
        eu, ev, ez, er = P["earth"]
        D = cam["D"]
        j0 = bisect.bisect_left(self.shell_t, t - 3.2); j1 = bisect.bisect_right(self.shell_t, t)
        if j1 <= j0:
            return
        yy = self.gy * F + CY; xx = self.gx * F + CX
        dd = np.sqrt((xx - eu) ** 2 + (yy - ev) ** 2) * 0.5  # half-res px
        coh, warm = P["coh"], P["warm"]
        col_a = np.array([0.5, 0.85, 1.0]) * (1 - coh) + np.array([1.0, 0.85, 0.6]) * coh
        col_a = (col_a * (1 - warm) + np.array([1.0, 0.6, 0.55]) * warm).astype(np.float32)
        acc = np.zeros_like(dd)
        for (b, s) in self.shell_list[j0:j1]:
            age = t - b
            R = 1.0 + age * 0.24 * D
            if R > 0.85 * D:
                continue
            rs = 0.5 * F * R / math.sqrt(max(D * D - R * R, 1e-3))
            if rs > 1400:
                continue
            fade = (1 - age / 3.2) ** 2 * (1 - smooth((R / D - 0.6) / 0.25))
            wdt = 1.6 + 0.012 * rs
            x = (dd - rs) / wdt
            acc += s * fade * (np.exp(-x * x) + 0.18 * np.exp(-(x / 6) ** 2))
        out += acc[..., None] * col_a * 0.55

    def draw_halo(self, out, P, t):
        eu, ev, ez, er = P["earth"]
        yy = self.gy * F + CY; xx = self.gx * F + CX
        dn = np.sqrt((xx - eu) ** 2 + (yy - ev) ** 2) / max(er, 0.5)
        o = np.maximum(dn - 1, 0)
        edge = smooth((dn - 0.96) / 0.06)
        coh = P["coh"]
        k = 0.55 + 0.9 * coh * (0.55 + 0.45 * P["beat"] * P["bass"]) + 0.3 * P["flash"]
        tight = np.exp(-o / 0.07) * edge
        wide = np.exp(-o / 0.55) * edge
        huge = np.exp(-o / 3.0) * 0.12 * coh
        cy = np.array([0.35, 0.65, 1.0], np.float32); gd = np.array([1.0, 0.8, 0.5], np.float32)
        out += (tight[..., None] * cy * 0.9 + wide[..., None] * (cy * (1 - coh) + gd * coh) * 0.35
                + huge[..., None] * gd) * k
        # 遠景でも地球が「光の点」として見えるように
        if er < 40:
            pt = np.exp(-(dn * max(er, 0.5) / 7.0) ** 2) * (1 - er / 40) * 1.2
            out += pt[..., None] * np.array([0.6, 0.85, 1.0], np.float32)

    def draw_stars(self, full, cam, P, t):
        d = self.star_dir
        z = d @ cam["fwd"]
        m = z > 0.05
        d = d[m]; z = z[m]
        u = CX + F * (d @ cam["right"]) / z
        v = CY - F * (d @ cam["up"]) / z
        b = self.star_b[m] * (0.8 + 0.2 * np.sin(self.star_tw[m] + t * self.star_tf[m]))
        b = b * min(1.0, 0.3 + t / 4.0)
        w = self.star_col[m] * b[:, None]
        splat(full, u, v, w)

    def collect_bodies(self, cam, P, t):
        bodies = []
        eu, ev, ez, er = P["earth"]
        if ez > 1.05:
            bodies.append(dict(kind="earth", pos=np.zeros(3, np.float32), rad=1.0, u=eu, v=ev, z=ez, r=er))
        if P["plan"] > 0.01:
            ring = P["ring"]
            for p in self.planets:
                ang = p["ph"] + p["om"] * t
                orb = np.array([math.cos(ang) * p["a"], 0.06 * p["a"] * math.sin(ang * 0.7), math.sin(ang) * p["a"]], np.float32)
                ci = float(smooth((ring * 1.4 - p["delay"]) / 1.0))
                dr = np.array([math.sin(t * 0.07 + p["ph"]), 0.4 * math.cos(t * 0.05 + p["ph"]), math.cos(t * 0.06 + p["ph"])], np.float32) * 2
                pos = (p["chaos"] + dr) * (1 - ci) + orb * ci
                u, v, z = self.project(cam, pos[None])
                z = float(z[0])
                if z - p["rad"] < 0.3:
                    continue
                r = F * p["rad"] / math.sqrt(max(z * z - p["rad"] ** 2, 1e-3))
                bodies.append(dict(kind="planet", p=p, pos=pos, rad=p["rad"], u=float(u[0]), v=float(v[0]), z=z, r=r))
        return bodies

    def draw_sphere(self, rgb, alpha, cam, bd, P, t):
        u0, v0, r = bd["u"], bd["v"], bd["r"]
        if r < 0.6:
            return
        x0 = int(max(math.floor(u0 - r - 2), 0)); x1 = int(min(math.ceil(u0 + r + 2), W))
        y0 = int(max(math.floor(v0 - r - 2), 0)); y1 = int(min(math.ceil(v0 + r + 2), H))
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        x = (xx + 0.5 - u0) / r; y = -(yy + 0.5 - v0) / r
        d2 = x * x + y * y
        a = np.clip((1 - np.sqrt(d2)) * r + 0.5, 0, 1)
        nz = np.sqrt(np.clip(1 - d2, 0, 1))
        # world normal
        R_, U_, Fw = cam["right"], cam["up"], cam["fwd"]
        nw = x[..., None] * R_ + y[..., None] * U_ - nz[..., None] * Fw
        L = np.array(P["light"], np.float32); L /= np.linalg.norm(L)
        ndl = x * L[0] + y * L[1] + nz * L[2]
        if bd["kind"] == "earth":
            Rm = self.earth_rot(t, P)
            nl = nw @ Rm  # = Rm^T n
            lat = np.arcsin(np.clip(nl[..., 1], -1, 1)); lon = np.arctan2(nl[..., 0], nl[..., 2])
            th_, tw_ = self.earth_tex.shape[:2]
            mx = ((lon / (2 * np.pi) + 0.5) * tw_).astype(np.float32)
            my = ((0.5 - lat / np.pi) * th_).astype(np.float32)
            col = cv2.remap(self.earth_tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            ocean = cv2.remap(self.earth_ocean, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            ch, cw = self.clouds.shape
            cmx = ((lon / (2 * np.pi) + 0.5) * cw + t * 1.5).astype(np.float32)
            cmy = ((0.5 - lat / np.pi) * ch).astype(np.float32)
            cl = cv2.remap(self.clouds, cmx, cmy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            col = col * (1 - cl[..., None] * 0.75) + cl[..., None] * 0.95
            diff = np.clip((ndl + 0.3) / 1.3, 0, 1)
            hv = L + np.array([0, 0, 1], np.float32); hv /= np.linalg.norm(hv)
            spec = np.clip(x * hv[0] + y * hv[1] + nz * hv[2], 0, 1) ** 50 * ocean * (1 - cl) * 0.8
            coh = P["coh"]
            night = np.array([0.03, 0.06, 0.12], np.float32) + np.array([0.08, 0.06, 0.02], np.float32) * coh
            shade = col * (0.1 + 1.05 * diff[..., None]) + night * (1 - diff[..., None]) + spec[..., None]
            rim = (1 - nz) ** 2.4
            atm = np.array([0.3, 0.6, 1.0], np.float32) * (1 - coh * 0.3) + np.array([0.5, 0.45, 0.3], np.float32) * coh * 0.3
            shade += rim[..., None] * atm * (0.5 + 0.8 * diff[..., None])
            shade *= 1.0 + 0.25 * P["beat"] * P["bass"] * coh
            if P["night"] > 0.001:
                nlt = cv2.remap(self.night_tex, mx * (self.night_tex.shape[1] / tw_), my * (self.night_tex.shape[0] / th_),
                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                nightk = (1 - np.clip(diff * 1.6, 0, 1)) ** 2 * P["night"] * (0.85 + 0.3 * P["beat"] * P["netsync"])
                ncol = np.array([1.0, 0.72, 0.38], np.float32) * (1 - P["netsync"]) + np.array([1.0, 0.88, 0.6], np.float32) * P["netsync"]
                shade += (nlt * nightk)[..., None] * ncol * (1 - cl[..., None] * 0.6)
        else:
            p = bd["p"]
            Rm = rot_z(p["tilt"]) @ rot_y(-p["spin"] * t)
            nl = nw @ Rm
            lat = np.arcsin(np.clip(nl[..., 1], -1, 1)); lon = np.arctan2(nl[..., 0], nl[..., 2])
            th_, tw_ = p["tex"].shape[:2]
            mx = ((lon / (2 * np.pi) + 0.5) * tw_).astype(np.float32)
            my = ((0.5 - lat / np.pi) * th_).astype(np.float32)
            col = cv2.remap(p["tex"], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            diff = np.clip((ndl + 0.15) / 1.15, 0, 1)
            shade = col * (0.05 + 1.0 * diff[..., None])
            rim = (1 - nz) ** 3
            shade += rim[..., None] * col * 0.5 * diff[..., None]
            shade *= P["plan"]
        a3 = a[..., None]
        sub_rgb = rgb[y0:y1, x0:x1]; sub_a = alpha[y0:y1, x0:x1]
        sub_rgb[:] = sub_rgb * (1 - a3) + shade.astype(np.float32) * a3
        sub_a[:] = sub_a * (1 - a3) + a3 * (1.0 if bd["kind"] == "earth" else P["plan"])

    def draw_particles(self, pfull, pglow, cam, P, t, bodies):
        f = self.feat
        coh, ring = P["coh"], P["ring"]
        beat, bass = P["beat"], P["bass"]
        # --- souls
        n = self.s_R.size
        ci = smooth((ring * 1.5 - self.s_delay) / 1.0)
        th = self.s_th0 + self.s_om * t * (1 + 0.6 * P["gold"])
        rx = self.s_R * np.cos(th); rz = self.s_R * np.sin(th)
        ry = self.s_y
        ringpos = np.stack([rx, ry, rz], 1)
        if P["incl"] > 0.001:
            inc = self.shell_incl * P["incl"]
            node = self.shell_node + t * 0.08
            I = inc[self.s_shell]; N = node[self.s_shell]
            ci_, si_ = np.cos(I), np.sin(I)
            y2 = ry * ci_ - rz * si_; z2 = ry * si_ + rz * ci_
            cn, sn = np.cos(N), np.sin(N)
            ringpos = np.stack([rx * cn + z2 * sn, y2, -rx * sn + z2 * cn], 1)
        drift = np.sin(self.s_df * t + self.s_dp) * 1.3
        chaos = self.s_home + drift
        pos = chaos * (1 - ci)[:, None] + ringpos * ci[:, None]
        col = self.s_chaos_col * (1 - ci)[:, None] + self.s_harm_col * ci[:, None]
        gold = P["gold"]
        if gold > 0:
            col = col * (1 - 0.35 * gold) + np.array([1.0, 0.88, 0.65], np.float32) * 0.35 * gold
        tw = 0.6 + 0.4 * np.sin(self.s_tw + self.s_tf * t)
        pulse = 0.75 + 0.6 * beat * (0.4 + bass)
        crest = np.maximum(np.cos(self.K0 * self.s_R - self.W0 * t), 0) ** 4
        b = self.s_b * (1.9 * tw * (1 - ci) + ci * pulse * (1 + 0.9 * coh * crest))
        b = b * (0.6 + 0.4 * P["plan"]) * (1 + 0.5 * P["flash"]) * (1 + 0.8 * P["incl"])
        allpos = [pos]; allcol = [col]; allb = [b]; allg = [1.0 + 0.8 * (1 - ci)]
        # 波源（バラバラな声）を光の点として描く
        if coh < 0.999 and P["wave"] > 0.01:
            sp = self._src_pos
            spos = np.stack([sp[:, 0], np.full(6, 0.05, np.float32), sp[:, 1]], 1).astype(np.float32)
            allpos.append(spos); allcol.append(self.src_col)
            allb.append(np.full(6, 2.2 * (1 - coh) * min(P["wave"] * 1.5, 1), np.float32)); allg.append(np.full(6, 6.0, np.float32))
        # --- galaxies
        galc = P["galc"]
        for g in self.gals:
            gth = g["th"] + g["om"] * t
            loc_r = g["r"] * g["R"]
            gp = (g["c"][None] + (loc_r * np.cos(gth))[:, None] * g["e1"] + (loc_r * np.sin(gth))[:, None] * g["e2"]
                  + (g["z"] * g["R"])[:, None] * g["nrm"])
            if galc > 0.001:
                cg = smooth((galc * 1.5 - g["delay"]) / 1.0)
                Rt = g["Rt"] + (g["r"] - 0.3) * 2.2
                ang = gth * 0.5 + g["gi"] * 1.3 + t * 0.03
                tx = Rt * np.cos(ang); tz = Rt * np.sin(ang); ty = g["z"] * 3
                if P["incl"] > 0.001:
                    I = (0.5 + 0.35 * g["gi"]) * (1 if g["gi"] % 2 else -1) * P["incl"]
                    ci_, si_ = math.cos(I), math.sin(I)
                    ty, tz = ty * ci_ - tz * si_, ty * si_ + tz * ci_
                tp = np.stack([tx, ty, tz], 1)
                gp = gp * (1 - cg)[:, None] + tp * cg[:, None]
                gcol = g["col"] * (1 - 0.5 * cg)[:, None] + self.s_harm_col[:g["col"].shape[0]] * (0.5 * cg)[:, None]
                gb = g["b"] * (1 + cg * (1.5 + 0.8 * beat * bass))
            else:
                gcol = g["col"]; gb = g["b"]
            allpos.append(gp); allcol.append(gcol); allb.append(gb * (0.5 + 0.5 * P["plan"]) * P.get("galb", 1.0)); allg.append(np.full(gp.shape[0], P.get("galg", 1.0), np.float32))
        # --- saturn ring particles
        for bd in bodies:
            if bd["kind"] == "planet" and bd["p"]["ring"]:
                p = bd["p"]; rr = p["ring_r"] * p["rad"]; th = p["ring_th"] + t * 0.15
                lx = rr * np.cos(th); lz = rr * np.sin(th)
                tilt = 0.45
                ly = -lz * math.sin(tilt); lz2 = lz * math.cos(tilt)
                rp = bd["pos"][None] + np.stack([lx, ly, lz2], 1)
                allpos.append(rp.astype(np.float32))
                allcol.append(np.tile(np.array([0.95, 0.85, 0.65], np.float32), (rr.size, 1)))
                allb.append(p["ring_b"] * 2.2 * P["plan"]); allg.append(np.full(rr.size, 0.5, np.float32))
        pos = np.concatenate(allpos).astype(np.float32)
        col = np.concatenate(allcol).astype(np.float32)
        b = np.concatenate(allb).astype(np.float32)
        gf = np.concatenate([np.broadcast_to(np.asarray(x, np.float32), (a.shape[0],)) for x, a in zip(allg, allpos)])
        u, v, z = self.project(cam, pos)
        m = (z > 0.6) & (u > -20) & (u < W + 20) & (v > -20) & (v < H + 20)
        # occlusion by bodies
        for bd in bodies:
            hid = (z > bd["z"]) & (((u - bd["u"]) ** 2 + (v - bd["v"]) ** 2) < (bd["r"] * 0.98) ** 2)
            m &= ~hid
        u, v, z, col, b, gf = u[m], v[m], z[m], col[m], b[m] * P["pvis"], gf[m]
        dist_k = np.clip((14.0 / z) ** 0.65, 0.3, 2.2)
        bb = b * dist_k
        w = col * bb[:, None]
        splat(pfull, u, v, w * 0.8)
        g = np.clip((8.0 / z), 0.3, 2.5) * gf
        splat(pglow, u * 0.5 - 0.25, v * 0.5 - 0.25, w * (0.5 * g)[:, None])

    def draw_spectrum(self, img, P, t):
        eu, ev, ez, er = P["earth"]
        mel = self.feat.mel_at(t)
        nb = 128
        canvas = np.zeros((H, W, 3), np.uint8)
        base = er * 1.42 + 14
        L = min(40 + er * 0.4, 160)
        for i in range(nb):
            a = i / nb
            m = 1 - abs(2 * a - 1)
            band = int(m * 46) + 1
            lvl = np.clip((mel[band] + 52) / 48, 0, 1) ** 1.6
            ln = 3 + L * lvl * P["spec"]
            ang = a * 2 * np.pi - np.pi / 2 + t * 0.05
            ca, sa = math.cos(ang), math.sin(ang)
            p0 = (int((eu + ca * base) * 4), int((ev + sa * base) * 4))
            p1 = (int((eu + ca * (base + ln)) * 4), int((ev + sa * (base + ln)) * 4))
            q = m
            c = (np.array([0.45, 0.85, 1.0]) * (1 - q) + np.array([1.0, 0.8, 0.45]) * q)
            c = c * (1 - P["warm"]) + np.array([1.0, 0.6, 0.55]) * P["warm"]
            cv2.line(canvas, p0, p1, tuple(int(x * 255) for x in c), 2, cv2.LINE_AA, shift=2)
        img += canvas.astype(np.float32) * (0.45 / 255.0) * P["spec"]

    def draw_galcores(self, out, cam, P):
        yy = self.gy * F + CY; xx = self.gx * F + CX
        for g in self.gals:
            u, v, z = self.project(cam, g["c"][None])
            z = float(z[0])
            if z < 2:
                continue
            u, v = float(u[0]), float(v[0])
            sig = 0.1 * g["R"] * F / z
            if u < -sig * 4 or u > W + sig * 4 or v < -sig * 4 or v > H + sig * 4:
                continue
            d2 = ((xx - u) ** 2 + (yy - v) ** 2)
            core = np.exp(-d2 / (2 * sig * sig)) * 1.2 + np.exp(-d2 / (2 * (sig * 3.5) ** 2)) * 0.25
            out += core[..., None] * np.array([1.0, 0.86, 0.68], np.float32) * P["galcore"]

    def rays(self, layer, P):
        eu, ev, _, _ = P["earth"]
        cx, cy = eu * 0.5, ev * 0.5
        src = np.maximum(layer - 0.05, 0)
        acc = np.zeros_like(layer); n = 8
        for i in range(1, n + 1):
            sc = 1 + 0.045 * i
            M = np.array([[1 / sc, 0, cx * (1 - 1 / sc)], [0, 1 / sc, cy * (1 - 1 / sc)]], np.float32)
            acc += cv2.warpAffine(src, M, (HW, HH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) * (1 - i / (n + 1))
        return acc / n

    def zoom_blur(self, img, P):
        eu, ev, _, _ = P["earth"]
        s = P["streak"]
        acc = img.copy(); n = 7
        for i in range(1, n):
            sc = 1 + 0.025 * i * s
            M = np.array([[sc, 0, eu * (1 - sc)], [0, sc, ev * (1 - sc)]], np.float32)
            acc += cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return acc / n * (1 + 0.6 * s)


def splat(buf, u, v, w):
    h, wd = buf.shape[:2]
    x0 = np.floor(u).astype(np.int64); y0 = np.floor(v).astype(np.int64)
    fx = (u - x0).astype(np.float32); fy = (v - y0).astype(np.float32)
    m = (x0 >= 0) & (x0 < wd - 1) & (y0 >= 0) & (y0 < h - 1)
    x0, y0, fx, fy, w = x0[m], y0[m], fx[m], fy[m], w[m]
    idx = y0 * wd + x0
    ii = np.concatenate([idx, idx + 1, idx + wd, idx + wd + 1])
    ww = np.concatenate([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy])
    flat = buf.reshape(-1, 3)
    for c in range(3):
        wc = np.concatenate([w[:, c]] * 4) * ww
        flat[:, c] += np.bincount(ii, weights=wc, minlength=h * wd).astype(np.float32)


# ---------------------------------------------------------------- telop
ROWH = {"ja": 76, "en": 70, "bg": 56, "tr": 52}   # 行の高さ(px)
TELOP_BOTTOM = 996                                 # テロップ下端（下から約8%）


class Telop:
    def __init__(self):
        fd = os.path.join(ASSETS, "fonts")
        self.fonts = {
            "ja": ImageFont.truetype(os.path.join(fd, "ShipporiMincho-700.ttf"), 58),
            "en": ImageFont.truetype(os.path.join(fd, "Cinzel-600.ttf"), 54),
            "bg": ImageFont.truetype(os.path.join(fd, "Cinzel-400.ttf"), 38),
            "title": ImageFont.truetype(os.path.join(fd, "Cinzel-600.ttf"), 104),
            "sub": ImageFont.truetype(os.path.join(fd, "ShipporiMincho-500.ttf"), 36),
            "endt": ImageFont.truetype(os.path.join(fd, "Cinzel-600.ttf"), 64),
            "tr": ImageFont.truetype(os.path.join(fd, "CormorantGaramond-Italic-600.ttf"), 42),
        }
        self.cache = {}
        self.cards = []
        n = len(CARDS)
        for i, c in enumerate(CARDS):
            s = c[0][1]; e = c[-1][2]
            ts = s - 0.45
            nxt = CARDS[i + 1][0][1] - 0.45 if i + 1 < n else 1e9
            te = min(e + 0.9, nxt)
            self.cards.append((ts, te, c))

    def line_img(self, text, kind, tracking=0):
        key = (text, kind, tracking)
        if key in self.cache:
            return self.cache[key]
        font = self.fonts[kind]
        pad = 40
        chars = list(text)
        widths = [font.getlength(ch) + tracking for ch in chars]
        tw = int(sum(widths)) + 2 * pad
        asc, desc = font.getmetrics()
        th = asc + desc + 2 * pad
        im = Image.new("L", (tw, th), 0)
        dr = ImageDraw.Draw(im)
        x = pad
        for ch, wch in zip(chars, widths):
            dr.text((x, pad), ch, font=font, fill=255)
            x += wch
        mask = np.asarray(im, np.float32) / 255.0
        glow = cv2.GaussianBlur(mask, (0, 0), 9)
        glow = glow / max(glow.max(), 1e-3)
        shadow = cv2.GaussianBlur(cv2.dilate(mask, np.ones((5, 5), np.uint8)), (0, 0), 12)
        shadow = np.clip(shadow / max(shadow.max(), 1e-3) * 1.3, 0, 1)
        r = (mask, glow, shadow, pad, tw - 2 * pad)
        self.cache[key] = r
        return r

    def blit(self, img, text, kind, cx, cy, alpha, reveal, active, P, tracking=0, textcol=(1, 1, 1), glow_k=1.0):
        mask, glow, shadow, pad, wtext = self.line_img(text, kind, tracking)
        h, w = mask.shape
        x0 = int(round(cx - w / 2)); y0 = int(round(cy - h / 2))
        # 左から光が広がるように表示（波のワイプ）
        xs = np.arange(w, dtype=np.float32) - pad
        soft = 90.0
        front = reveal * (wtext + soft * 2) - soft
        colmask = np.clip((front - xs) / soft + 0.5, 0, 1)
        colmask = (colmask * colmask * (3 - 2 * colmask))[None, :]
        X0, Y0 = max(x0, 0), max(y0, 0); X1, Y1 = min(x0 + w, W), min(y0 + h, H)
        if X1 <= X0 or Y1 <= Y0:
            return
        sl = (slice(Y0 - y0, Y1 - y0), slice(X0 - x0, X1 - x0))
        cm = colmask[:, sl[1]]
        m = mask[sl] * cm * alpha; g = glow[sl] * cm * alpha; s = shadow[sl] * cm * alpha
        sub = img[Y0:Y1, X0:X1]
        coh, warm = P["coh"], P["warm"]
        gcol = np.array([0.45, 0.75, 1.0]) * (1 - coh) + np.array([1.0, 0.78, 0.45]) * coh
        gcol = gcol * (1 - warm) + np.array([1.0, 0.6, 0.55]) * warm
        gcol = gcol.astype(np.float32)
        act = 0.35 + 0.65 * active
        sub *= (1 - 0.6 * s)[..., None]
        sub += g[..., None] * gcol * (0.55 * act * glow_k)
        tc = np.array(textcol, np.float32) * (0.62 + 0.38 * active)
        sub[:] = sub * (1 - m[..., None]) + m[..., None] * tc

    def draw(self, img, t, P):
        # タイトル
        if 1.2 < t < 8.8:
            a = float(smooth((t - 1.4) / 1.6) * (1 - smooth((t - 7.4) / 1.2)))
            rv = float(smooth((t - 1.4) / 2.2))
            self.blit(img, "COHERENCE SYMPHONY", "title", W / 2, 300, a, rv, 1.0, P, tracking=14)
            a2 = float(smooth((t - 3.0) / 1.5) * (1 - smooth((t - 7.4) / 1.2)))
            self.blit(img, "波動による全宇宙の調和", "sub", W / 2, 395, a2 * 0.9, float(smooth((t - 3.0) / 1.8)), 0.6, P, tracking=10)
            a3 = float(smooth((t - 3.6) / 1.5) * (1 - smooth((t - 7.4) / 1.2)))
            self.blit(img, "Harmony of the Whole Universe through Vibration", "tr", W / 2, 447, a3 * 0.85,
                      float(smooth((t - 3.6) / 1.8)), 0.6, P, tracking=2, glow_k=0.6)
        if t > 261.5:
            a = float(smooth((t - 261.8) / 1.5))
            self.blit(img, "COHERENCE SYMPHONY", "endt", W / 2, 890, a, float(smooth((t - 261.8) / 2.0)), 1.0, P, tracking=10)
        if t > 263.0:
            # クレジット（最後の最後まで表示）
            a = float(smooth((t - 263.2) / 1.2))
            self.blit(img, "Lyric & Music by Shota Watanabe", "tr", W / 2, 962, a * 0.95, float(smooth((t - 263.2) / 1.6)),
                      1.0, P, tracking=2, textcol=(0.93, 0.95, 1.0), glow_k=0.6)
        for (ts, te, lines) in self.cards:
            if t < ts or t > te:
                continue
            fade = float(1 - smooth((t - (te - 0.35)) / 0.35))
            # 行を下から積む（日本語の行の下に英訳）
            rows = []
            for li, (text, s, e, kind, tr) in enumerate(lines):
                rows.append((li, text, kind, s, e))
                if tr:
                    rows.append((li, tr, "tr", s, e))
            y = TELOP_BOTTOM - sum(ROWH[r[2]] for r in rows)
            for (li, text, kind, s, e) in rows:
                yc = y + ROWH[kind] / 2; y += ROWH[kind]
                delay = li * 0.25 + (0.2 if kind == "tr" else 0)
                rv = float(smooth((t - (ts + delay)) / 0.75))
                if len(lines) == 1:
                    active = 1.0
                else:
                    active = float(smooth((t - (s - 0.3)) / 0.3) * (1 - smooth((t - (e + 0.05)) / 0.35)))
                    if li == 0 and t < s:
                        active = 1.0
                if kind == "tr":
                    self.blit(img, text, kind, W / 2, yc, fade * 0.92, rv, active, P, tracking=1,
                              textcol=(0.88, 0.92, 1.0), glow_k=0.6)
                    continue
                textcol = (0.92, 0.95, 1.0) if kind == "bg" else (1, 1, 1)
                self.blit(img, text, kind, W / 2, yc, fade * (0.85 if kind == "bg" else 1.0), rv, active, P,
                          tracking=4 if kind != "ja" else 2, textcol=textcol)


# ---------------------------------------------------------------- drivers
_scene = None

def get_scene():
    global _scene
    if _scene is None:
        _scene = Scene()
    return _scene

def duration():
    return float(np.load(os.path.join(ASSETS, "audio_feat.npz"))["dur"])

def render_chunk(args):
    i0, i1, out = args
    sc = get_scene()
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
           "-pix_fmt", "yuv420p", "-threads", "1", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(i0, i1):
        fr = sc.render(i / FPS)
        p.stdin.write(fr.tobytes())
        if (i - i0) % 150 == 0:
            el = time.time() - t0
            print(f"[{out}] {i - i0}/{i1 - i0} frames, {el:.0f}s", flush=True)
    p.stdin.close(); p.wait()
    return out

def main():
    mode = sys.argv[1]
    if mode == "still":
        sc = get_scene()
        os.makedirs(os.path.join(HERE, "preview"), exist_ok=True)
        for s in sys.argv[2:]:
            t = float(s); t0 = time.time()
            fr = sc.render(t)
            fn = os.path.join(HERE, "preview", f"f_{t:07.2f}.jpg")
            cv2.imwrite(fn, fr[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(fn, f"{time.time() - t0:.2f}s", flush=True)
    elif mode == "render":
        out = sys.argv[2]
        nproc = int(os.environ.get("MV_PROCS", "4"))
        start = float(os.environ.get("MV_START", "0")); end = float(os.environ.get("MV_END", str(duration())))
        n0 = int(round(start * FPS)); n1 = int(math.ceil(end * FPS))
        segdir = os.path.join(HERE, "segs"); os.makedirs(segdir, exist_ok=True)
        nseg = nproc * 3
        bounds = np.linspace(n0, n1, nseg + 1).astype(int)
        jobs = [(int(bounds[k]), int(bounds[k + 1]), os.path.join(segdir, f"seg{k:03d}.mp4")) for k in range(nseg)]
        from multiprocessing import Pool
        with Pool(nproc) as pool:
            segs = pool.map(render_chunk, jobs, chunksize=1)
        lst = os.path.join(segdir, "list.txt")
        with open(lst, "w") as fh:
            for s in segs:
                fh.write(f"file '{s}'\n")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst]
        if AUDIO and start == 0:
            cmd += ["-i", AUDIO, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy"]
        elif AUDIO:
            cmd += ["-ss", str(start), "-t", str(end - start), "-i", AUDIO, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest"]
        else:
            cmd += ["-c", "copy"]
        cmd += ["-movflags", "+faststart", out]
        subprocess.check_call(cmd)
        print("done", out)


if __name__ == "__main__":
    main()
