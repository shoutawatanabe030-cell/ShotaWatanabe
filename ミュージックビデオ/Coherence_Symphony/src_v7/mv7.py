# -*- coding: utf-8 -*-
"""
Coherence Symphony MV v2 — カット割りを細かくした版
宇宙（イントロ）→ 地球上の人々（世界中の人が手を取り合い波動を共鳴させる）→ 地球の内部 →
地球がひとつになり、宇宙全体が調和する（終盤）

使い方:
  python3 mv2.py still 40 90 ...      # 静止画
  python3 mv2.py shots                 # 各カットの代表フレームを一覧
  MV_AUDIO=... python3 mv2.py render out.mp4
"""
import os, sys, math, time, bisect, subprocess
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mv_core as mc
from mv_core import kf, smooth, W, H, FPS, F
import scenes as SC
import photo_anim as PH
import real as RL
from core import Core

AUDIO = os.environ.get("MV_AUDIO", "")
BEAT = 60 / 128.5714
JAPAN_LON = 139.7


# ---------------------------------------------------------------- camera helpers
def mkcam(C, T=(0, 0, 0), roll=0.0):
    C = np.asarray(C, np.float32); T = np.asarray(T, np.float32)
    fwd = T - C; fwd /= np.linalg.norm(fwd)
    up0 = np.array([0, 1, 0], np.float32)
    if abs(float(fwd @ up0)) > 0.98:
        up0 = np.array([0, 0, -1], np.float32)
    right = np.cross(fwd, up0); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    cr, sr = math.cos(roll), math.sin(roll)
    right, up = right * cr + up * sr, up * cr - right * sr
    return dict(C=C, fwd=fwd.astype(np.float32), right=right.astype(np.float32), up=up.astype(np.float32),
                D=float(np.linalg.norm(C)))


def sph(D, el, az):
    return (D * math.cos(el) * math.sin(az), D * math.sin(el), D * math.cos(el) * math.cos(az))


def orbit(t0, t1, D=(10, 10), el=(0.4, 0.4), az=(0.0, 0.3), T=((0, 0, 0), (0, 0, 0)), roll=(0, 0), ease=True, logD=True):
    def f(t):
        u = (t - t0) / (t1 - t0)
        u = float(smooth(u)) * 0.6 + u * 0.4 if ease else u
        d = math.exp(math.log(D[0]) * (1 - u) + math.log(D[1]) * u) if logD else D[0] * (1 - u) + D[1] * u
        e = el[0] * (1 - u) + el[1] * u; a = az[0] * (1 - u) + az[1] * u
        tg = np.asarray(T[0]) * (1 - u) + np.asarray(T[1]) * u
        c = np.asarray(sph(d, e, a)) + tg
        return mkcam(c, tg, roll[0] * (1 - u) + roll[1] * u)
    return f


def path(t0, t1, C0, C1, T0=(0, 0, 0), T1=(0, 0, 0), roll=(0, 0)):
    def f(t):
        u = (t - t0) / (t1 - t0); u = float(smooth(u)) * 0.5 + u * 0.5
        c = np.asarray(C0) * (1 - u) + np.asarray(C1) * u
        tg = np.asarray(T0) * (1 - u) + np.asarray(T1) * u
        return mkcam(c, tg, roll[0] * (1 - u) + roll[1] * u)
    return f


def spin_for(lon_deg, t, az):
    return az + 0.045 * t + 2.2 - math.radians(lon_deg)


# ---------------------------------------------------------------- shot list
NIGHT_L = (0.75, 0.3, -0.6)     # 夜側が見える光の向き
DAY_L = (-0.45, 0.38, 0.81)
NO_FX = dict(spec=0, wave=0, plan=0, pvis=0)

S = []  # (start, end, kind, args)


def shot(t0, t1, kind, xin=0.0, **kw):
    S.append(dict(t0=t0, t1=t1, kind=kind, xin=xin, kw=kw))


# ---- INTRO（宇宙）
shot(0.0, 8.93, "space", cam=path(0, 8.93, (52, 32, 12), (45, 22, -4), (38, 10, -30), (38, 10, -30)),
     over=dict(coh=0, ring=0, wave=0, plan=0, spec=0, halo=0.6, galb=3.0, galg=4.0, galcore=0.9))
shot(8.93, 16.17, "space", xin=1.0, cam=orbit(8.93, 16.17, D=(26, 19), el=(0.75, 0.6), az=(0.4, 1.1), T=((-46, -6, -18), (-46, -6, -18))),
     over=dict(coh=0, ring=0, wave=0, plan=0, spec=0, galb=2.6, galg=4.0, galcore=0.8))
shot(16.17, 23.77, "space", xin=0.6, cam=orbit(16.17, 23.77, D=(70, 30), el=(0.35, 0.3), az=(0.9, 1.3)),
     over=dict(coh=0, ring=0, wave=0.3, plan=0.6, spec=0, galb=2.0, galg=2.5, galcore=0.6))
shot(23.77, 27.67, "planet", name="saturn", off=((3.8, 1.0, 3.2), (2.6, 0.7, 2.0)), over=dict(ring=1, coh=0, wave=0.4, plan=1, spec=0))
shot(27.67, 31.33, "planet", name="jupiter", off=((-4.2, 1.3, 2.8), (-3.0, 0.6, 3.2)), over=dict(ring=1, coh=0, wave=0.5, plan=1, spec=0))
shot(31.33, 38.67, "space", cam=orbit(31.33, 38.67, D=(13, 1.9), el=(0.35, 0.18), az=(1.6, 2.1)),
     over=dict(ring=0.2, coh=0, wave=0.4, plan=1, spec=0, night=0.6, light=(0.2, 0.4, 0.9), pvis=lambda t: 1 - smooth((t - 34) / 3)))
# ---- VERSE 1（バラバラの人々）
shot(38.67, 42.30, "people", fn="sc_alone", env_name="city_night", seed=1, x=600, hue=0.0, height=330)
shot(42.30, 45.97, "space", cam=orbit(42.30, 45.97, D=(2.25, 2.1), el=(0.25, 0.3), az=(0.3, 0.38)),
     over=dict(NO_FX, night=1.0, cities=1.0, netsync=0, light=NIGHT_L, spin=spin_for(80, 44, 0.34), coh=0, halo=0.5))
shot(45.97, 49.63, "people", fn="sc_walkers", env_name="city_night", seed=3, n=10)
shot(49.63, 53.30, "people", fn="sc_alone", env_name="desert_dusk", seed=4, x=1250, hue=0.12, height=300, pose="stand",
     others=((420, 150, 0.8),))
shot(53.30, 56.97, "people", fn="sc_alone", env_name="beach_dawn", seed=6, x=720, hue=0.55, height=360, pose="stand")
shot(56.97, 60.63, "people", fn="sc_pullback", env_name="night_hill", seed=5)
shot(60.63, 64.30, "people", fn="sc_chest", env_name="window_night", seed=7, dim=True)
shot(64.30, 67.97, "space", cam=orbit(64.30, 67.97, D=(3.6, 4.2), el=(0.3, 0.35), az=(0.5, 0.7)),
     over=dict(NO_FX, night=0.8, cities=0.7, netsync=0, light=NIGHT_L, spin=spin_for(-60, 66, 0.6), coh=0, halo=0.5))
# ---- PRE-CHORUS（手を差し出し、握り返す）
shot(67.97, 71.63, "people", fn="sc_hands", seed=9, mode="reach")
shot(71.63, 75.30, "people", fn="sc_hands", seed=10, mode="clasp")
shot(75.30, 78.97, "space", cam=orbit(75.30, 78.97, D=(2.4, 2.9), el=(0.45, 0.4), az=(0.2, 0.35)),
     over=dict(NO_FX, night=1.0, cities=1.0, netsync=lambda t: smooth((t - 75.3) / 3.6), light=NIGHT_L,
               net=lambda t: smooth((t - 75.6) / 3.3), netstart=0, spin=spin_for(JAPAN_LON - 10, 77, 0.28), coh=0.5, halo=0.6))
shot(78.97, 82.60, "core", R=(300, 420), pulse=lambda u: 0.4 + 1.2 * u, waves=lambda u: u, rise=0.0)
# ---- CHORUS 1
shot(82.60, 86.30, "space", cam=orbit(82.60, 86.30, D=(6.5, 9.0), el=(0.45, 0.55), az=(0.0, 0.25)), over=dict(halo=1.0))
shot(86.30, 89.97, "people", fn="sc_line", env_name="sunrise_ridge", seed=11, n=9)
shot(89.97, 93.63, "space", cam=orbit(89.97, 93.63, D=(2.7, 3.0), el=(0.5, 0.42), az=(0.1, 0.3)),
     over=dict(NO_FX, night=1.0, cities=1.0, netsync=1.0, net=1.0, light=NIGHT_L, spin=spin_for(30, 92, 0.2), halo=0.8))
shot(93.63, 97.30, "globe", seed=23)
shot(97.30, 100.97, "people", fn="sc_circle", env_name="night_hill", seed=13)
shot(100.97, 104.63, "core", R=(380, 360), pulse=lambda u: 1.0, waves=lambda u: 1.0, rise=0.6, yaw=-0.2)
shot(104.63, 108.27, "people", fn="sc_line", env_name="dusk_hill", seed=12, n=8, raise_keys=((0, 0), (0.55, 1), (1, 1)))
shot(108.27, 113.77, "space", cam=orbit(108.27, 113.77, D=(4.5, 12), el=(0.35, 0.5), az=(0.4, 0.9)),
     over=dict(night=0.6, net=1.0, cities=0.8, netsync=1, light=(0.3, 0.3, 0.9), spin=spin_for(100, 110, 0.6)))
# ---- VERSE 2
shot(113.77, 117.43, "people", fn="sc_two", env_name="dusk_hill", seed=15, mode="touch")
shot(117.43, 121.10, "space", cam=orbit(117.43, 121.10, D=(2.3, 2.5), el=(0.45, 0.4), az=(0.6, 0.7)),
     over=dict(NO_FX, night=1.0, cities=1.0, netsync=0.4, light=NIGHT_L, net=lambda t: 0.18 * smooth((t - 117.6) / 2.5),
               netstart=13, spin=spin_for(10, 119, 0.65), coh=0.4, halo=0.5))
shot(121.10, 124.77, "people", fn="sc_alone", env_name="city_night", seed=21, x=1320, hue=0.62, height=260, pose="stand")
shot(124.77, 128.43, "people", fn="sc_alone", env_name="night_hill", seed=22, x=520, hue=0.08, height=280, pose="stand",
     others=((1420, 270, 0.6),))
# ---- PRE-CHORUS 2
shot(128.43, 132.10, "core", R=(340, 380), pulse=lambda u: 1.4, waves=lambda u: 0.7, rise=0.0, yaw=-0.55)
shot(132.10, 135.77, "people", fn="sc_chest", env_name="window_night", seed=8, dim=False)
shot(135.77, 141.27, "space", cam=orbit(135.77, 141.27, D=(13, 9.5), el=(1.05, 0.9), az=(1.8, 2.0)),
     over=dict(spec=0, plan=0.6))
# ---- CHORUS 2
shot(141.27, 144.93, "space", cam=orbit(141.27, 144.93, D=(9, 13), el=(0.5, 0.55), az=(2.0, 2.3)))
shot(144.93, 148.60, "people", fn="sc_line", env_name="beach_dawn", seed=16, n=8, height=300)
shot(148.60, 152.30, "space", cam=orbit(148.60, 152.30, D=(2.8, 3.1), el=(0.3, 0.38), az=(1.0, 1.2)),
     over=dict(NO_FX, night=1.0, cities=1.0, netsync=1.0, net=1.0, light=NIGHT_L, spin=spin_for(-75, 150, 1.1), halo=0.8))
shot(152.30, 155.93, "people", fn="sc_circle", env_name="sunrise_ridge", seed=14, n=12, column=0.7)
shot(155.93, 159.60, "space", cam=orbit(155.93, 159.60, D=(26, 34), el=(0.55, 0.65), az=(2.4, 2.7)))
shot(159.60, 163.30, "core", R=(360, 300), pulse=lambda u: 1.2, waves=lambda u: 1.0, rise=1.0, yaw=-0.35)
shot(163.30, 166.93, "people", fn="sc_line", env_name="night_hill", seed=18, n=9, raise_keys=((0, 1), (1, 1)))
shot(166.93, 168.77, "globe", seed=24)
# ---- BRIDGE（暖色）
shot(168.77, 172.47, "space", cam=orbit(168.77, 172.47, D=(7, 6), el=(0.3, 0.35), az=(2.6, 2.8)),
     over=dict(night=0.4, light=(0.6, 0.4, 0.7), spin=spin_for(JAPAN_LON, 170, 2.7)))
shot(172.47, 176.13, "people", fn="sc_two", env_name="sunrise_ridge", seed=19, mode="embrace")
shot(176.13, 179.80, "people", fn="sc_share", env_name="desert_dusk", seed=17)
shot(179.80, 183.47, "people", fn="sc_lanterns", env_name="night_hill", seed=20)
shot(183.47, 187.13, "space", cam=orbit(183.47, 187.13, D=(2.6, 3.3), el=(0.4, 0.5), az=(2.9, 3.0)),
     over=dict(NO_FX, night=1.0, cities=0.6, netsync=0.6, kind=1.0, light=NIGHT_L, spin=spin_for(60, 185, 2.95), halo=0.7))
shot(187.13, 190.90, "people", fn="sc_reach_sky", env_name="night_hill", seed=21)
shot(190.90, 194.57, "people", fn="sc_lookup", env_name="aurora_snow", seed=25)
shot(194.57, 200.10, "space", cam=orbit(194.57, 200.10, D=(1.6, 17), el=(0.2, 1.1), az=(3.2, 3.4)),
     over=dict(night=0.5, light=(0.4, 0.5, 0.8), spin=spin_for(JAPAN_LON, 196, 3.2)))
# ---- FINAL CHORUS
shot(200.10, 203.77, "space", cam=orbit(200.10, 203.77, D=(15, 19), el=(1.15, 1.1), az=(3.4, 3.6)))
shot(203.77, 207.43, "people", fn="sc_circle", env_name="night_hill", seed=26, raise_keys=((0, 0), (0.5, 1), (1, 1)))
shot(207.43, 211.10, "space", cam=orbit(207.43, 211.10, D=(2.9, 3.4), el=(0.35, 0.45), az=(3.6, 3.8)),
     over=dict(NO_FX, night=1.0, cities=1.0, netsync=1.0, net=1.0, light=NIGHT_L, spin=spin_for(JAPAN_LON, 209, 3.7), halo=0.9))
shot(211.10, 214.80, "globe", seed=27, raise_keys=((0, 0), (0.6, 1), (1, 1)))
shot(214.80, 218.47, "people", fn="sc_line", env_name="sunrise_ridge", seed=28, n=11, raise_keys=((0, 1), (1, 1)), height=300, z=(1.0, 1.12))
shot(218.47, 222.13, "space", cam=orbit(218.47, 222.13, D=(16, 20), el=(0.55, 0.65), az=(3.8, 4.1)))
shot(222.13, 225.80, "people", fn="sc_lookup", env_name="aurora_snow", seed=29, n=7)
shot(225.80, 229.50, "space", cam=orbit(225.80, 229.50, D=(22, 26), el=(1.2, 1.25), az=(4.1, 4.3)))
# ---- OUTRO
shot(229.50, 236.80, "space", xin=0.8, cam=orbit(229.50, 236.80, D=(4.5, 6.5), el=(0.3, 0.4), az=(4.3, 4.6)),
     over=dict(night=0.7, net=1.0, cities=0.8, netsync=1, light=(0.5, 0.3, 0.8), spin=spin_for(JAPAN_LON, 233, 4.45)))
shot(236.80, 244.30, "globe", xin=0.8, seed=30, raise_keys=((0, 0), (0.7, 1), (1, 1)))
# ---- FINALE（地球がひとつになり、宇宙全体が調和する）
shot(244.30, 248.00, "space", cam=orbit(244.30, 248.00, D=(10, 20), el=(0.6, 0.7), az=(4.6, 4.9)))
shot(248.00, 251.70, "people", fn="sc_circle", env_name="night_hill", seed=31, n=16, raise_keys=((0, 1), (1, 1)), column=1.4)
shot(251.70, 255.37, "space", cam=orbit(251.70, 255.37, D=(3.0, 26), el=(0.4, 0.9), az=(5.0, 5.2)),
     over=dict(night=0.8, net=1.0, cities=1.0, netsync=1, light=(0.5, 0.3, 0.8), spin=spin_for(JAPAN_LON, 252, 5.0)))
shot(255.37, 259.07, "space", cam=orbit(255.37, 259.07, D=(28, 34), el=(1.25, 1.2), az=(5.2, 5.5)))
shot(259.07, 267.60, "space", cam=orbit(259.07, 267.60, D=(28, 95), el=(0.9, 1.0), az=(5.5, 5.9)))


# ---- v3: 人物のカットを実写風の画像（Canva AI生成）に差し替え
GOLD_C = (1.0, 0.82, 0.5); WARM_C = (1.0, 0.65, 0.45); CYAN_C = (0.5, 0.85, 1.0)
PHOTO = {
    38.67: dict(img="p01.png", z=(1.05, 1.15), c0=(0.3, 0.5), c1=(0.5, 0.5),
                fx=[("rings", 0.12, 0.5, (1.0, 0.35, 0.4), 260), ("rings", 0.47, 0.45, (0.35, 1.0, 0.6), 260), ("rings", 0.85, 0.42, (0.4, 0.6, 1.0), 220)]),
    45.97: dict(img="p02.png", z=(1.05, 1.12), c0=(0.42, 0.5), c1=(0.58, 0.5),
                fx=[("rings", 0.25, 0.38, (0.45, 0.6, 1.0), 320), ("rings", 0.76, 0.4, (1.0, 0.6, 0.3), 320)]),
    49.63: dict(img="p02.png", z=(1.4, 1.28), c0=(0.74, 0.45), c1=(0.7, 0.45), fx=[("rings", 0.76, 0.4, (1.0, 0.6, 0.3), 300)]),
    53.30: dict(img="p03.png", z=(1.32, 1.2), c0=(0.24, 0.42), c1=(0.3, 0.45), fx=[("rings", 0.22, 0.42, (0.7, 0.45, 1.0), 260)]),
    56.97: dict(img="p03.png", z=(1.18, 1.0), c0=(0.42, 0.5), c1=(0.5, 0.5), fx=[]),
    60.63: dict(img="p04.png", z=(1.0, 1.1), c0=(0.42, 0.45), c1=(0.36, 0.45), fx=[("glow", 0.27, 0.73, (0.6, 0.4, 0.25), 20)]),
    67.97: dict(img="p05.png", z=(1.0, 1.12), c0=(0.45, 0.5), c1=(0.38, 0.5),
                fx=[("glow", 0.33, 0.55, (0.9, 0.7, 0.45), 30), ("particles", 0.1, 0.7, 0.2, 0.9, (1.0, 0.8, 0.5), 50)]),
    71.63: dict(img="p06.png", z=(1.05, 1.15), c0=(0.55, 0.55), c1=(0.55, 0.52),
                fx=[("glow", 0.56, 0.55, (1.2, 0.95, 0.6), 45), ("rings", 0.56, 0.55, GOLD_C, 900), ("particles", 0.2, 0.9, 0.1, 0.9, (1.0, 0.85, 0.55), 80)]),
    86.30: dict(img="p07.png", z=(1.0, 1.08), c0=(0.5, 0.5), c1=(0.5, 0.47),
                fx=[("rings", 0.5, 0.78, GOLD_C, 1200), ("particles", 0.0, 1.0, 0.1, 0.9, (1.0, 0.85, 0.55), 90)]),
    93.63: dict(img="p21.png", z=(1.0, 1.1), c0=(0.5, 0.45), c1=(0.5, 0.45), fx=[("rings", 0.5, 0.45, GOLD_C, 1400)]),
    97.30: dict(img="p08.png", z=(1.1, 1.0), c0=(0.5, 0.4), c1=(0.5, 0.45),
                fx=[("rings", 0.5, 0.2, CYAN_C, 1100), ("particles", 0.0, 1.0, 0.2, 1.0, (0.7, 0.85, 1.0), 80)]),
    104.63: dict(img="p09.png", z=(1.0, 1.08), c0=(0.36, 0.55), c1=(0.3, 0.55),
                 fx=[("glow", 0.25, 0.62, (0.8, 0.5, 0.35), 28), ("particles", 0.0, 0.6, 0.2, 1.0, (1.0, 0.75, 0.5), 50)]),
    113.77: dict(img="p10.png", z=(1.0, 1.1), c0=(0.45, 0.62), c1=(0.47, 0.66),
                 fx=[("glow", 0.47, 0.72, (1.0, 0.8, 0.5), 26), ("rings", 0.47, 0.72, GOLD_C, 600)]),
    121.10: dict(img="p11.png", z=(1.0, 1.1), c0=(0.55, 0.45), c1=(0.6, 0.45), grade=(0.88, 0.92, 1.05), fx=[]),
    124.77: dict(img="p11.png", z=(1.38, 1.26), c0=(0.62, 0.42), c1=(0.6, 0.4), grade=(0.88, 0.92, 1.05),
                 fx=[("glow", 0.6, 0.58, (0.6, 0.45, 0.3), 18)]),
    132.10: dict(img="p12.png", z=(1.05, 1.15), c0=(0.6, 0.5), c1=(0.62, 0.52),
                 fx=[("glow", 0.73, 0.8, (1.1, 0.8, 0.5), 34), ("rings", 0.73, 0.8, WARM_C, 800)]),
    144.93: dict(img="p13.png", z=(1.0, 1.1), c0=(0.5, 0.5), c1=(0.5, 0.47),
                 fx=[("rings", 0.48, 0.47, GOLD_C, 1300), ("particles", 0.0, 1.0, 0.1, 0.9, (1.0, 0.85, 0.6), 80)]),
    152.30: dict(img="p20.png", z=(1.0, 1.08), c0=(0.5, 0.5), c1=(0.5, 0.45), fx=[("rings", 0.5, 0.4, GOLD_C, 1400)]),
    163.30: dict(img="p09.png", z=(1.28, 1.16), c0=(0.3, 0.5), c1=(0.32, 0.52),
                 fx=[("glow", 0.25, 0.62, (0.9, 0.55, 0.35), 30), ("particles", 0.0, 0.7, 0.1, 1.0, (1.0, 0.75, 0.5), 60)]),
    166.93: dict(img="p07.png", z=(1.25, 1.32), c0=(0.5, 0.5), c1=(0.5, 0.5), fx=[("rings", 0.5, 0.78, GOLD_C, 1200)]),
    172.47: dict(img="p14.png", z=(1.0, 1.08), c0=(0.5, 0.45), c1=(0.5, 0.42),
                 fx=[("glow", 0.42, 0.55, (0.7, 0.45, 0.3), 40), ("particles", 0.0, 1.0, 0.2, 1.0, (1.0, 0.7, 0.45), 60)]),
    176.13: dict(img="p15.png", z=(1.0, 1.1), c0=(0.55, 0.58), c1=(0.58, 0.62),
                 fx=[("glow", 0.6, 0.62, (1.1, 0.8, 0.5), 34), ("particles", 0.3, 0.9, 0.3, 1.0, (1.0, 0.8, 0.5), 50)]),
    179.80: dict(img="p16.png", z=(1.0, 1.1), c0=(0.42, 0.6), c1=(0.3, 0.65),
                 fx=[("glow", 0.22, 0.68, (1.0, 0.75, 0.45), 26), ("rings", 0.22, 0.68, WARM_C, 520), ("particles", 0.0, 1.0, 0.2, 1.0, (1.0, 0.75, 0.45), 60)]),
    187.13: dict(img="p17.png", z=(1.0, 1.12), c0=(0.5, 0.58), c1=(0.5, 0.6),
                 fx=[("glow", 0.5, 0.62, (1.2, 0.95, 0.6), 36), ("rings", 0.5, 0.62, GOLD_C, 800), ("particles", 0.1, 0.9, 0.2, 1.0, (1.0, 0.85, 0.55), 60)]),
    190.90: dict(img="p13.png", z=(1.32, 1.15), c0=(0.5, 0.72), c1=(0.5, 0.58), fx=[("rings", 0.48, 0.47, GOLD_C, 1100)]),
    203.77: dict(img="p08.png", z=(1.15, 1.05), c0=(0.5, 0.45), c1=(0.5, 0.42),
                 fx=[("rings", 0.5, 0.2, GOLD_C, 1200), ("particles", 0.0, 1.0, 0.1, 1.0, (1.0, 0.9, 0.65), 90)]),
    211.10: dict(img="p20.png", z=(1.15, 1.0), c0=(0.5, 0.45), c1=(0.5, 0.5), fx=[("rings", 0.5, 0.4, GOLD_C, 1500)]),
    214.80: dict(img="p07.png", z=(1.0, 1.1), c0=(0.5, 0.45), c1=(0.5, 0.48),
                 fx=[("rings", 0.5, 0.78, GOLD_C, 1300), ("particles", 0.0, 1.0, 0.1, 0.9, (1.0, 0.85, 0.55), 100), ("sweep", (1.0, 0.85, 0.6))]),
    222.13: dict(img="p12.png", z=(1.25, 1.12), c0=(0.5, 0.42), c1=(0.52, 0.45), fx=[("glow", 0.73, 0.8, (1.1, 0.8, 0.5), 34)]),
    236.80: dict(img="p19.png", z=(1.0, 1.1), c0=(0.4, 0.5), c1=(0.38, 0.52), fx=[("particles", 0.0, 1.0, 0.0, 0.7, (0.9, 0.85, 0.7), 40)]),
    248.00: dict(img="p18.png", z=(1.0, 1.1), c0=(0.5, 0.5), c1=(0.5, 0.45),
                 fx=[("rings", 0.5, 0.8, GOLD_C, 1500), ("particles", 0.0, 1.0, 0.0, 1.0, (1.0, 0.9, 0.65), 100), ("sweep", (1.0, 0.85, 0.6))]),
}
# ---- v4: 静止画をすべて動かす（奥行きの動き・呼吸・空/水/光の流れ・粒子）
ANIM = {
    "p01.png": dict(sky=0.42, skyspeed=10, bokeh=0.6, sway=1.0),
    "p02.png": dict(rain=(0.0, 1.0, 0.0, 1.0), bokeh=0.5, sway=1.0),
    "p03.png": dict(sky=0.3, water=(0.3, 0.52, 0.3, 1.0), sway=1.0),
    "p04.png": dict(depth="fg", rain=(0.72, 1.0, 0.0, 1.0), bokeh=0.8, sway=0.8),
    "p05.png": dict(depth="fg", bokeh=1.0, sway=1.5),
    "p06.png": dict(depth="fg", bokeh=1.0, sway=0.8, pulse=0.5),
    "p07.png": dict(depth="y", sky=0.33, flow=3.0, pulse=0.5, sway=0.8),
    "p08.png": dict(depth="y", sky=0.2, flow=3.0, pulse=0.6, sway=0.6),
    "p09.png": dict(depth="fg", sky=0.35, bokeh=0.7, sway=0.8),
    "p10.png": dict(depth="fg", steam=[(0.13, 0.61), (0.73, 0.51)], bokeh=0.4, sway=0.4),
    "p11.png": dict(bokeh=0.8, sway=0.6),
    "p12.png": dict(depth="fg", bokeh=0.8, sway=0.7, pulse=0.3),
    "p13.png": dict(depth="y", water=(0.55, 1.0), sky=0.25, flow=2.5, pulse=0.6),
    "p14.png": dict(depth="fg", sky=0.3, bokeh=0.5, sway=0.9),
    "p15.png": dict(depth="fg", bokeh=0.3, sway=0.6, pulse=0.3),
    "p16.png": dict(leaves=True, water=(0.75, 1.0), wamp=1.2, bokeh=0.5, sway=0.6),
    "p17.png": dict(depth="fg", sky=0.35, bokeh=0.6, sway=0.8, pulse=0.5),
    "p18.png": dict(depth="y", flow=3.0, pulse=0.6, sway=0.3),
    "p19.png": dict(sky=0.33, water=(0.35, 0.62, 0.45, 1.0), mist=0.6, sway=0.5),
    "p20.png": dict(depth="y", flow=3.0, pulse=0.6, sway=0.4),
    "p21.png": dict(depth="y", flow=2.5, pulse=0.6, sway=0.3),
}
for k_, v_ in PHOTO.items():
    v_["anim"] = ANIM[v_["img"]]
for s_ in S:
    key = round(s_["t0"], 2)
    if s_["kind"] in ("people", "globe"):
        assert key in PHOTO, key
        s_["kind"] = "photo"; s_["kw"] = PHOTO.pop(key)
assert not PHOTO, PHOTO.keys()


# ---- v5: 要所でデバイスを登場させる（人が持つ・手渡しする）
DEV_C = (0.7, 0.88, 1.15)
DEVICE = {
    67.97: dict(img="d1.png", z=(1.0, 1.1), c0=(0.5, 0.5), c1=(0.55, 0.48),
                fx=[("halo", 0.58, 0.47, DEV_C, 230), ("rings", 0.58, 0.47, DEV_C, 800), ("particles", 0.1, 0.9, 0.2, 0.9, (1.0, 0.85, 0.6), 50)],
                anim=dict(depth="fg", bokeh=0.8, sway=1.2, pulse=0.4)),
    71.63: dict(img="d2.png", z=(1.05, 1.15), c0=(0.5, 0.5), c1=(0.51, 0.5),
                fx=[("halo", 0.51, 0.5, DEV_C, 200), ("rings", 0.51, 0.5, GOLD_C, 900), ("particles", 0.1, 0.9, 0.1, 0.9, (1.0, 0.85, 0.55), 70)],
                anim=dict(depth="fg", bokeh=0.8, sway=0.8, pulse=0.4)),
    113.77: dict(img="d3.png", z=(1.0, 1.1), c0=(0.5, 0.5), c1=(0.49, 0.52),
                 fx=[("halo", 0.48, 0.53, DEV_C, 120), ("rings", 0.48, 0.53, DEV_C, 650)],
                 anim=dict(depth="fg", sky=0.18, bokeh=0.5, sway=0.7, pulse=0.3)),
    176.13: dict(img="d4.png", z=(1.0, 1.08), c0=(0.5, 0.55), c1=(0.5, 0.58),
                 fx=[("halo", 0.5, 0.7, DEV_C, 110), ("rings", 0.5, 0.7, WARM_C, 700), ("particles", 0.2, 0.8, 0.3, 1.0, (1.0, 0.8, 0.5), 50)],
                 anim=dict(depth="fg", sky=0.42, water=(0.5, 0.62, 0.15, 0.85), sway=0.9, pulse=0.3)),
    187.13: dict(img="d5.png", z=(1.0, 1.12), c0=(0.5, 0.5), c1=(0.52, 0.47),
                 fx=[("halo", 0.53, 0.43, DEV_C, 200), ("rings", 0.53, 0.43, GOLD_C, 800), ("particles", 0.1, 0.9, 0.2, 1.0, (1.0, 0.85, 0.55), 60)],
                 anim=dict(depth="fg", sky=0.3, bokeh=0.4, sway=1.0, pulse=0.4)),
    214.80: dict(img="d6.png", z=(1.0, 1.1), c0=(0.5, 0.5), c1=(0.52, 0.52),
                 fx=[("rings", 0.56, 0.69, DEV_C, 1300), ("particles", 0.0, 1.0, 0.1, 0.9, (0.8, 0.9, 1.0), 90), ("sweep", (0.8, 0.9, 1.0))],
                 anim=dict(depth="fg", flow=2.5, pulse=0.6, sky=0.15, sway=0.6)),
}
for s_ in S:
    key = round(s_["t0"], 2)
    if key in DEVICE:
        assert s_["kind"] == "photo", key
        s_["kw"] = DEVICE.pop(key)
assert not DEVICE


# ---- v6: いただいた実際の写真を、歌詞に合う場所に入れる（写真そのものは変形させない）
def R(img, **kw):
    d = dict(img=img); d.update(kw); d["_kind"] = "real"; return d

REAL = {
    86.30: [(86.30, 89.97, R("r09.png", z=(1.0, 1.06)))],                                   # 手を取り合っていこう: デバイスを持つ女性（v9で入れ替え）
    93.63: [(93.63, 97.30, R("r05.png", z=(1.0, 1.08), c0=(0.5, 0.52), c1=(0.5, 0.46)))],     # ひとつになっていこう: 集合写真
    104.63: [(104.63, 108.27, R("r15.png", z=(1.0, 1.08), c0=(0.5, 0.42), c1=(0.52, 0.4), bottom=0.95))],  # We are love もうひとりじゃない: 笑い合う二人
    144.93: [(144.93, 148.60, R("r01.png", z=(1.0, 1.06)))],                                   # 手を取り合っていこう: 大きなデバイスの前で
    152.30: [(152.30, 155.93, R("r07.png", z=(1.0, 1.08), c0=(0.5, 0.5), c1=(0.5, 0.46)))],   # ひとつになっていこう: デバイスを掲げる三人
    163.30: [(163.30, 165.10, R("r02.png", z=(1.0, 1.05), seed=2)),                           # We are love: 自撮り
             (165.10, 166.93, R("r03.png", z=(1.0, 1.05), seed=3))],                          # もうひとりじゃない: 自撮り
    166.93: [(166.93, 168.77, R("r12.png", z=(1.0, 1.08), c0=(0.5, 0.5), c1=(0.5, 0.47)))],   # 両手首のデバイス
    179.80: [(179.80, 181.63, R("r11.png", z=(1.0, 1.08), c0=(0.5, 0.5), c1=(0.53, 0.47))),  # 小さな優しさが: デバイスと休む人
             (181.63, 183.47, R("r06.png", z=(1.0, 1.05), c0=(0.5, 0.5), c1=(0.55, 0.55)))],  # 波になって 世界を変えていく: 表彰
    190.90: [(190.90, 194.57, R("r04.png", z=(1.0, 1.06)))],                                   # もうひとりじゃない: バナーの前の三人
    203.77: [(203.77, 207.43, R("r08.png", z=(1.0, 1.1), c0=(0.55, 0.55), c1=(0.5, 0.5)))],   # Coherence Symphony: デバイスを掲げて乾杯
    211.10: [(211.10, 212.93, R("r13.png", layout="card", card_h=0.8, card_y=0.42, z=(1.0, 1.05))),                                    # ひとつになっていこう: カンファレンス
             (212.93, 214.80, R("r14.png", z=(1.0, 1.05), seed=14))],                         # みんなで自撮り
    222.13: [(222.13, 225.80, R("r16.png", z=(1.0, 1.06)))],                                   # We are love もうひとりじゃない: 夕日の海辺で手をつなぐ二人（v9で入れ替え）
    # ---- v7: 追加でいただいた写真3枚と動画
    97.30: [(97.30, 100.97, R("r17.png", z=(1.0, 1.1), c0=(0.5, 0.55), c1=(0.42, 0.6)))],      # 同じ波動で: 講演に集まった会場
    183.47: [(183.47, 187.13, R("clip/f_%03d.jpg", seq=112, card_h=0.86, card_y=0.45, z=(1.0, 1.04)))],  # 世界を変えていく: 講演の動画
    218.47: [(218.47, 220.30, R("r19.png", z=(1.0, 1.05), seed=19)),                            # 同じ波動で: 大きなデバイスの前の集合写真
             (220.30, 222.13, R("r18.png", z=(1.0, 1.05), seed=18))],                           # （Hey!）: みんなで乾杯
    236.80: [(236.80, 240.60, None),                                                          # それはきっと: 海辺の二人（v5のまま）
             (240.60, 244.30, R("r10.png", z=(1.0, 1.08)))],                                   # 僕らの証: サイン入りのデバイス
}
_new = []
for s_ in S:
    key = round(s_["t0"], 2)
    if key in REAL:
        parts = REAL.pop(key)
        assert abs(parts[0][0] - s_["t0"]) < 1e-6 and abs(parts[-1][1] - s_["t1"]) < 1e-6, key
        for k, (a, b, spec) in enumerate(parts):
            if spec is None:
                _new.append(dict(t0=a, t1=b, kind=s_["kind"], xin=s_["xin"] if k == 0 else 0.0, kw=s_["kw"]))
            else:
                kw_ = {kk: vv for kk, vv in spec.items() if kk != "_kind"}
                _new.append(dict(t0=a, t1=b, kind="real", xin=s_["xin"] if k == 0 else 0.0, kw=kw_))
    else:
        _new.append(s_)
assert not REAL, REAL.keys()
S[:] = _new


# ---- 縦動画（MV_VERTICAL=1）: 横長の画像は縦に切り抜き、人物から人物へゆっくり横に流す
VERTICAL = bool(os.environ.get("MV_VERTICAL"))
if VERTICAL:
    VOVR = {
        38.67: dict(c0=(0.17, 0.5), c1=(0.5, 0.5), z=(1.0, 1.04)),
        45.97: dict(c0=(0.3, 0.5), c1=(0.72, 0.5), z=(1.0, 1.0)),
        49.63: dict(c0=(0.74, 0.45), c1=(0.71, 0.45), z=(1.1, 1.02)),
        53.30: dict(c0=(0.22, 0.45), c1=(0.26, 0.45), z=(1.06, 1.0)),
        56.97: dict(c0=(0.28, 0.5), c1=(0.78, 0.5), z=(1.0, 1.0)),
        60.63: dict(c0=(0.42, 0.45), c1=(0.36, 0.45), z=(1.0, 1.06)),
        67.97: dict(c0=(0.54, 0.52), c1=(0.6, 0.5), z=(1.0, 1.04)),
        71.63: dict(c0=(0.5, 0.5), c1=(0.51, 0.5), z=(1.0, 1.06)),
        113.77: dict(c0=(0.48, 0.52), c1=(0.48, 0.53), z=(1.0, 1.06)),
        121.10: dict(c0=(0.55, 0.45), c1=(0.58, 0.45), z=(1.0, 1.05)),
        124.77: dict(c0=(0.62, 0.42), c1=(0.6, 0.4), z=(1.15, 1.05)),
        132.10: dict(c0=(0.48, 0.45), c1=(0.62, 0.48), z=(1.0, 1.04)),
        166.93: dict(layout="cover", c0=(0.5, 0.5), c1=(0.5, 0.48)),
        172.47: dict(c0=(0.42, 0.45), c1=(0.55, 0.42), z=(1.0, 1.04)),
        176.13: dict(c0=(0.22, 0.5), c1=(0.78, 0.5), z=(1.0, 1.0)),
        179.80: dict(layout="cover", c0=(0.48, 0.5), c1=(0.52, 0.47)),
        187.13: dict(c0=(0.48, 0.5), c1=(0.53, 0.47), z=(1.0, 1.06)),
        214.80: dict(c0=(0.5, 0.55), c1=(0.56, 0.55), z=(1.0, 1.05)),
        236.80: dict(c0=(0.36, 0.5), c1=(0.44, 0.52), z=(1.0, 1.05)),
        248.00: dict(c0=(0.45, 0.5), c1=(0.55, 0.45), z=(1.0, 1.05)),
    }
    for s_ in S:
        key = round(s_["t0"], 2)
        if key in VOVR:
            s_["kw"] = dict(s_["kw"], **VOVR[key])

for i in range(len(S) - 1):
    assert abs(S[i]["t1"] - S[i + 1]["t0"]) < 1e-6, (i, S[i]["t1"], S[i + 1]["t0"])


# ---------------------------------------------------------------- renderer
class MV2:
    def __init__(self):
        self.sc = mc.Scene()
        self.core = Core(self.sc)
        self.feat = self.sc.feat
        self.starts = [s["t0"] for s in S]
        self.telop = self.sc.telop

    def gparams(self, t):
        f = self.feat
        P = dict(coh=kf(t, mc.COH), warm=kf(t, mc.WARM), gold=kf(t, mc.GOLD), fade=kf(t, mc.FADE))
        P["beat"] = f.beat_env(t); P["bass"] = f.at(f.bass_s, t)
        imp = sum(math.exp(-(t - ti) / 0.45) for ti in mc.IMPACTS if t >= ti)
        hey = sum(math.exp(-(t - ti) / 0.3) for ti in mc.HEYS if t >= ti)
        P["flash"] = min(imp + hey * 0.8, 1.5)
        j0 = bisect.bisect_left(f.beats, t - 4); j1 = bisect.bisect_right(f.beats, t)
        P["beats_near"] = list(f.beats[j0:j1])
        return P

    def earth_disc(self, t, cx, cy, R, P):
        img = np.zeros((H, W, 3), np.float32); a = np.zeros((H, W, 1), np.float32)
        cam = mkcam((0, 0.4, 3.0))
        PP = dict(coh=P.get("coh", 1), beat=P.get("beat", 0), bass=P.get("bass", 0), light=(-0.3, 0.4, 0.85),
                  night=0.6, netsync=1.0, spin=spin_for(JAPAN_LON, t, 0.0), plan=1)
        self.sc.draw_sphere(img, a, cam, dict(kind="earth", u=cx, v=cy, r=R, z=3.0), PP, t)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / R
        halo = np.exp(-np.maximum(d - 1, 0) / 0.06) * (d > 0.98) + 0.25 * np.exp(-np.maximum(d - 1, 0) / 0.5) * (d > 1)
        img += halo[..., None] * np.array([0.35, 0.6, 1.0], np.float32) * (0.8 + 0.4 * PP["beat"])
        return img

    def planet_cam(self, s, t):
        name = s["kw"]["name"]
        p = [q for q in self.sc.planets if q["name"] == name][0]
        ang = p["ph"] + p["om"] * t
        pos = np.array([math.cos(ang) * p["a"], 0.06 * p["a"] * math.sin(ang * 0.7), math.sin(ang) * p["a"]], np.float32)
        u = (t - s["t0"]) / (s["t1"] - s["t0"])
        o0, o1 = [np.array(o, np.float32) for o in s["kw"]["off"]]
        off = o0 * (1 - u) + o1 * u
        tgt = pos * (1 - 0.35 * u)
        return mkcam(pos + off, tgt)

    def render_shot(self, s, t):
        kind = s["kind"]; kw = s["kw"]
        u = (t - s["t0"]) / (s["t1"] - s["t0"])
        if kind == "space":
            img, P = self.sc.render_linear(t, cam_fn=kw.get("cam"), over=kw.get("over"))
            return img, P
        if kind == "planet":
            img, P = self.sc.render_linear(t, cam_fn=lambda tt: self.planet_cam(s, tt), over=kw.get("over"))
            return img, P
        P = self.gparams(t)
        P["shot_len"] = s["t1"] - s["t0"]
        tl = t - s["t0"]
        if kind == "core":
            R0, R1 = kw.get("R", (360, 360))
            R = R0 + (R1 - R0) * float(smooth(u))
            bg = SC.env("space", 77)["bg"]
            img = bg.copy() + self.core.render(t, P, R=R, yaw=kw.get("yaw", -0.42), pulse=kw["pulse"](u),
                                               waves=kw["waves"](u), rise=kw.get("rise", 0.0))
            return img, P
        if kind == "globe":
            img = SC.sc_globe_ring(tl, t, P, scene3d=self, seed=kw.get("seed", 23),
                                   raise_keys=kw.get("raise_keys", ((0, 0), (1, 0))))
            return img, P
        if kind == "real":
            img = RL.shot(tl, t, P, shot_len=P["shot_len"], **kw)
            P["bloomk"] = 0.2   # 実際の写真は光のにじみを弱くして、白く霞まないようにする
            return img, P
        if kind == "photo":
            args = {k: v for k, v in kw.items()}
            img = PH.shot(tl, t, P, shot_len=P["shot_len"], **args)
            return img, P
        if kind == "people":
            fn = getattr(SC, kw["fn"])
            args = {k: v for k, v in kw.items() if k != "fn"}
            img = fn(tl, t, P, **args)
            return img, P
        raise ValueError(kind)

    def post(self, img, P, t):
        small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        bright = np.maximum(small - 0.55, 0)
        b1 = cv2.GaussianBlur(bright, (0, 0), 6); b2 = cv2.GaussianBlur(bright, (0, 0), 22)
        img = img + cv2.resize(b1 * 0.55 + b2 * 0.5, (W, H), interpolation=cv2.INTER_LINEAR) * P.get("bloomk", 1.0)
        if P["flash"] > 0.01:
            img += np.array([1.0, 0.92, 0.8], np.float32) * (0.22 * P["flash"])
        img = 1 - np.exp(-img * 1.15)
        img = img * self.sc.vignette
        ef = kf(t, [(264.9, 0), (265.55, 1), (266.4, 0)])
        if ef > 0:
            img = img + (1 - img) * ef * 0.85
        return img

    def render(self, t):
        i = max(bisect.bisect_right(self.starts, t) - 1, 0)
        s = S[i]
        img, P = self.render_shot(s, t)
        P.setdefault("fade", kf(t, mc.FADE))
        out = self.post(img, P, t)
        if s["xin"] > 0 and t - s["t0"] < s["xin"] and i > 0:
            a = float(smooth((t - s["t0"]) / s["xin"]))
            img2, P2 = self.render_shot(S[i - 1], t)
            out = out * a + self.post(img2, P2, t) * (1 - a)
        out *= kf(t, mc.FADE)
        PT = dict(coh=P.get("coh", 0), warm=P.get("warm", 0))
        self.telop.draw(out, t, PT)
        return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


_mv = None


def get():
    global _mv
    if _mv is None:
        _mv = MV2()
    return _mv


def render_chunk(args):
    i0, i1, out = args
    m = get()
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
           "-pix_fmt", "yuv420p", "-threads", "1", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(i0, i1):
        p.stdin.write(m.render(i / FPS).tobytes())
        if (i - i0) % 150 == 0:
            print(f"[{os.path.basename(out)}] {i - i0}/{i1 - i0} {time.time() - t0:.0f}s", flush=True)
    p.stdin.close(); p.wait()
    return out


def main():
    mode = sys.argv[1]
    pv = os.path.join(HERE, "preview"); os.makedirs(pv, exist_ok=True)
    if mode == "still":
        m = get()
        for a in sys.argv[2:]:
            t = float(a); t0 = time.time()
            fr = m.render(t)
            fn = os.path.join(pv, f"{'vv' if VERTICAL else 'v7'}_{t:07.2f}.jpg")
            cv2.imwrite(fn, fr[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(fn, f"{time.time() - t0:.2f}s", flush=True)
    elif mode == "shots":
        m = get()
        sel = sys.argv[2:] if len(sys.argv) > 2 else None
        for k, s in enumerate(S):
            if sel and str(k) not in sel:
                continue
            t = s["t0"] + (s["t1"] - s["t0"]) * 0.6
            fr = m.render(t)
            small = cv2.resize(fr, (480, 270), interpolation=cv2.INTER_AREA)
            cv2.putText(small, f"{k} {s['t0']:.1f} {s['kind']} {s['kw'].get('fn', '')}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imwrite(os.path.join(pv, f"s7_{k:02d}.jpg"), small[:, :, ::-1])
            print(k, flush=True)
    elif mode == "render":
        out = sys.argv[2]
        nproc = int(os.environ.get("MV_PROCS", "4"))
        dur = float(np.load(os.path.join(mc.ASSETS, "audio_feat.npz"))["dur"])
        n1 = int(math.ceil(dur * FPS))
        segdir = os.path.join(HERE, "segs7v" if VERTICAL else "segs7"); os.makedirs(segdir, exist_ok=True)
        nseg = nproc * 4
        bounds = np.linspace(0, n1, nseg + 1).astype(int)
        jobs = [(int(bounds[k]), int(bounds[k + 1]), os.path.join(segdir, f"seg{k:03d}.mp4")) for k in range(nseg)]
        from multiprocessing import Pool
        with Pool(nproc) as pool:
            segs = pool.map(render_chunk, jobs, chunksize=1)
        lst = os.path.join(segdir, "list.txt")
        with open(lst, "w") as fh:
            for s_ in segs:
                fh.write(f"file '{s_}'\n")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst]
        if AUDIO:
            cmd += ["-i", AUDIO, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy"]
        else:
            cmd += ["-c", "copy"]
        cmd += ["-movflags", "+faststart", out]
        subprocess.check_call(cmd)
        print("done", out)


if __name__ == "__main__":
    main()
