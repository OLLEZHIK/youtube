#!/usr/bin/env python3
"""Бесшовный зацикленный фон: светящиеся плавные линии на тёмно-синем фоне.

Зацикливание точное: каждая гармоника кривых, фон и пульсации яркости
вращаются с ЦЕЛЫМ числом оборотов за период T, поэтому кадр N == кадр 0.

Запуск:
    python3 glow_loop.py                      # 1920x1080, 20 c, 30 fps -> glow_loop.mp4
    python3 glow_loop.py --preview            # быстрый просмотр 960x540, 6 c
    python3 glow_loop.py --seed 7             # другая «хаотичная» раскладка линий
    python3 glow_loop.py --duration 30 --fps 60 --out big.mp4
"""
import argparse
import subprocess
from multiprocessing import Pool

import cv2
import numpy as np

TAU = 2 * np.pi


# ----------------------------------------------------------------- палитры
PALETTES = {
    "blue": dict(
        base=(0.004, 0.012, 0.035), tint=(0.05, 0.22, 0.46), blob=1.0, line=1.0, glow=1.0,
        glows=((0.55, 0.78, 1.0), (0.30, 0.60, 1.0), (0.14, 0.42, 0.95), (0.08, 0.30, 0.85)),
        core=(0.9, 0.97, 1.0), curve_cols=((1, 1, 1),), flare_thr=1.2,
        beam_core=(0.80, 0.93, 1.0), beam_mid=(0.10, 0.40, 1.0)),
    # тёмное золото на почти чёрном, вторая линия — приглушённая серо-голубая
    "gold": dict(
        base=(0.006, 0.005, 0.008), tint=(0.30, 0.19, 0.02), blob=0.55, line=0.8, glow=0.8,
        glows=((1.0, 0.72, 0.20), (0.95, 0.60, 0.10), (0.85, 0.48, 0.06), (0.60, 0.33, 0.03)),
        core=(1.0, 0.86, 0.48), curve_cols=((1, 1, 1), (0.55, 0.62, 0.78)), flare_thr=0.7,
        beam_core=(1.0, 0.88, 0.45), beam_mid=(1.0, 0.50, 0.04)),
}

# ----------------------------------------------------------------- сцена
def make_scene(seed, n_curves):
    """Случайные, но детерминированные параметры (по seed)."""
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n_curves):
        c = {"cx": rng.uniform(-0.12, 0.12), "cy": rng.uniform(-0.2, 0.2), "x": [], "y": []}
        c["idx"] = len(curves)
        for axis in ("x", "y"):
            for k in range(1, 5):
                amp = (0.95 if axis == "x" else 0.75) * rng.uniform(0.6, 1.0) / (k ** 1.9)
                # целые скорости вращения фазы => идеальный цикл; разные => «хаос»
                rate = int(rng.choice([-1, -1, 0, 1, 1]))
                c[axis].append((k, amp, rng.uniform(0, TAU), rate))
        # бегущие яркие «кометы» вдоль линии (целые числа => цикл)
        c["pulse_a"] = int(rng.integers(1, 3))      # сколько пятен на кривой
        c["pulse_rate"] = int(rng.choice([-1, 1]))
        c["pulse_phase"] = rng.uniform(0, TAU)
        c["gain"] = rng.uniform(0.75, 1.1)
        curves.append(c)
    blobs = []
    for _ in range(4):  # мягкие световые пятна фона
        blobs.append({
            "cx": rng.uniform(-0.6, 0.6), "cy": rng.uniform(-0.4, 0.4),
            "rx": rng.uniform(0.25, 0.5), "ry": rng.uniform(0.3, 0.6),
            "ax": rng.uniform(0.1, 0.3), "ay": rng.uniform(0.1, 0.25),
            "px": rng.uniform(0, TAU), "py": rng.uniform(0, TAU),
            "nx": int(rng.choice([-1, 1])), "ny": int(rng.choice([-1, 1])),
            "br": rng.uniform(0.6, 1.0), "bn": int(rng.choice([1, 2])), "bp": rng.uniform(0, TAU),
        })
    return {"curves": curves, "blobs": blobs}


def curve_points(c, u, W, H, S=2400):
    """Замкнутая кривая в пикселях в момент u in [0,1) (u = t/T)."""
    s = np.linspace(0, TAU, S, endpoint=False)
    x = np.zeros(S)
    y = np.zeros(S)
    for k, a, ph, r in c["x"]:
        x += a * np.cos(k * s + ph + TAU * r * u)
    for k, a, ph, r in c["y"]:
        y += a * np.sin(k * s + ph + TAU * r * u)
    px = (c["cx"] + x) * W * 0.5 + W * 0.5
    py = (c["cy"] + y) * H * 0.5 + H * 0.5
    # комета: яркость и толщина вместе
    p = 0.5 + 0.5 * np.cos(c["pulse_a"] * s + c["pulse_phase"] + TAU * c["pulse_rate"] * u)
    return px, py, s, p ** 4


def splat(buf, x, y, w):
    """Билинейный «разбрызг» точек в буфер (векторно)."""
    H, W = buf.shape
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    fx = x - x0; fy = y - y0
    flat = buf.ravel()
    for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                       (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi = x0 + dx; yi = y0 + dy
        m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        flat += np.bincount(yi[m] * W + xi[m], weights=(ww * w)[m], minlength=flat.size)


def blur(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)


def blur_big(img, sigma, f):
    """Большое размытие через уменьшение — быстро и без потери качества."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // f, h // f), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


# ----------------------------------------------------------------- кадр
SCENE = None
CFG = None


def init(scene, cfg):
    global SCENE, CFG
    SCENE, CFG = scene, cfg


def background(u, W, H):
    gw, gh = 480, 270
    X, Y = np.meshgrid(np.linspace(-1, 1, gw), np.linspace(-1, 1, gh))
    X = X * (W / H)
    P = CFG["pal"]
    base = np.stack([np.full_like(X, v) for v in P["base"]], -1)
    tint = np.array(P["tint"])
    for b in SCENE["blobs"]:
        cx = b["cx"] * (W / H) + b["ax"] * np.cos(b["px"] + TAU * b["nx"] * u)
        cy = b["cy"] + b["ay"] * np.sin(b["py"] + TAU * b["ny"] * u)
        k = b["br"] * (0.75 + 0.25 * np.cos(b["bp"] + TAU * b["bn"] * u))
        g = np.exp(-(((X - cx) / b["rx"]) ** 2 + ((Y - cy) / b["ry"]) ** 2))
        base += (k * P["blob"] * g)[..., None] * tint
    vig = 1.0 - 0.45 * np.clip((X / (W / H)) ** 2 * 0.6 + Y ** 2 * 0.6, 0, 1)
    base *= vig[..., None]
    return cv2.resize(base.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)  # RGB


def beams(u, W, H, P):
    """Два «живых» сгустка света по углам (без явных лучей): контур плавно меняет форму.
    Деформация — гармоники угла с целыми скоростями => идеальный цикл, движение очень медленное."""
    gw, gh = 480, 270
    X, Y = np.meshgrid(np.linspace(0, W / H, gw), np.linspace(0, 1, gh))
    out = np.zeros((gh, gw, 3), np.float32)
    mid = np.array(P["beam_mid"]); core = np.array(P["beam_core"])
    rng = np.random.default_rng(CFG["seed"] + 101)
    for sx, sy in ((-0.03 * W / H, -0.07), (0.96 * W / H, 0.98)):
        # центр сгустка медленно блуждает
        cx = sx + 0.05 * np.sin(rng.uniform(0, TAU) + TAU * u * int(rng.choice([-1, 1])))
        cy = sy + 0.05 * np.sin(rng.uniform(0, TAU) + TAU * u * int(rng.choice([-1, 1])))
        dx, dy = X - cx, Y - cy
        r_ = np.hypot(dx, dy) + 1e-4
        th = np.arctan2(dy, dx)
        deform = np.ones_like(r_)
        for k, amp in ((1, 0.22), (2, 0.18), (3, 0.10), (4, 0.05)):
            deform += amp * np.cos(k * th + rng.uniform(0, TAU) + TAU * int(rng.choice([-1, 1])) * u)
        deform = np.clip(deform, 0.55, None)
        rr = r_ / (0.30 * deform)
        breath = 0.85 + 0.15 * np.cos(rng.uniform(0, TAU) + TAU * int(rng.choice([1, 2])) * u)
        body = np.exp(-rr ** 1.3 * 1.8) * 0.9 + 0.35 / (1 + (rr * 1.4) ** 2)
        hot = 1.6 * np.exp(-((r_ / (0.10 * deform)) ** 1.4))
        out += ((body)[..., None] * mid + hot[..., None] * core) * breath
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC)


def render(i):
    W, H, N = CFG["W"], CFG["H"], CFG["N"]
    P = CFG["pal"]
    sc = W / 1920.0                    # масштаб толщин под разрешение
    u = i / N
    bg = background(u, W, H)

    # три канала ядра (хроматическая окантовка) x три класса толщины
    sig = np.array([0.8, 1.9, 4.2]) * sc
    core = np.zeros((3, 3, H, W), np.float32)      # [канал][класс]
    off = (-1.3 * sc, 0.0, 1.3 * sc)               # R / G / B смещение вдоль нормали
    for c in SCENE["curves"]:
        px, py, s, pulse = curve_points(c, u, W, H)
        # нормаль
        dx = np.gradient(px); dy = np.gradient(py)
        nl = np.hypot(dx, dy) + 1e-6
        nx, ny = -dy / nl, dx / nl
        # равномерная выборка по дуге
        px2 = np.append(px, px[0]); py2 = np.append(py, py[0])
        L = np.concatenate([[0], np.cumsum(np.hypot(np.diff(px2), np.diff(py2)))])
        n = max(int(L[-1] / (0.6 * sc)), 400)
        t = np.linspace(0, L[-1], n, endpoint=False)
        ii = np.arange(len(px2))
        fi = np.interp(t, L, ii)
        xs = np.interp(fi, ii, px2); ys = np.interp(fi, ii, py2)
        nxs = np.interp(fi, ii, np.append(nx, nx[0])); nys = np.interp(fi, ii, np.append(ny, ny[0]))
        pl = np.interp(fi, ii, np.append(pulse, pulse[0]))
        step = L[-1] / n
        inten = (0.28 + 0.9 * pl) * c["gain"] * P["line"] * step / sc  # яркость вдоль кривой
        wid = np.clip(pl * 1.6, 0, 1) * 2.0               # 0..2 -> между классами
        wts = (np.clip(1 - wid, 0, 1), np.clip(1 - np.abs(wid - 1), 0, 1), np.clip(wid - 1, 0, 1))
        for ch in range(3):
            X = xs + nxs * off[ch]; Y = ys + nys * off[ch]
            for k in range(3):
                splat(core[ch, k], X, Y, inten * wts[k] * P["curve_cols"][c["idx"] % len(P["curve_cols"])][ch])
    img = np.zeros((H, W, 3), np.float32)
    gain_k = (1.0, 1.35, 2.2)  # толстые классы шире => компенсируем яркость
    for ch in range(3):
        acc = np.zeros((H, W), np.float32)
        for k in range(3):
            acc += blur(core[ch, k], sig[k]) * gain_k[k] * (1.0 / (1.0 + 0.25 * k)) * 2.0
        img[..., ch] = acc
    lum = img.mean(-1)

    # свечение: несколько радиусов, синий оттенок; пересечения вспыхивают сильнее (lum**1.4)
    lg = lum ** 1.4
    g1 = blur(lg, 6 * sc)
    g2 = blur_big(lg, 22 * sc, 2)
    g3 = blur_big(lg, 70 * sc, 4)
    g4 = blur_big(lg, 180 * sc, 8)
    glow = (g1[..., None] * np.array(P["glows"][0]) * 1.6 * P["glow"] +
            g2[..., None] * np.array(P["glows"][1]) * 2.2 * P["glow"] +
            g3[..., None] * np.array(P["glows"][2]) * 3.2 * P["glow"] +
            g4[..., None] * np.array(P["glows"][3]) * 3.5 * P["glow"])

    # отдельные яркие вспышки в самых ярких точках (пересечения/«кометы»)
    hot = np.clip(lum - P["flare_thr"], 0, None)
    flare = blur_big(hot, 10 * sc, 1)[..., None] * np.array([0.8, 0.9, 1.0]) * 3.0

    # core слегка уходит в белый
    corec = img * np.array(P["core"])
    out = bg + glow + corec + flare
    if CFG["style"] == "beams":
        out = out + beams(u, W, H, P)
    out = 1.0 - np.exp(-out * 1.15)            # мягкий tone-mapping без клиппинга
    out = out ** (1 / 1.05)
    # зерно + дизеринг (убирает бандинг на тёмных градиентах, на цикл не влияет)
    rng = np.random.default_rng(i)
    out += rng.normal(0, 0.006, out.shape).astype(np.float32)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="glow_loop.mp4")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--duration", type=float, default=20.0, help="длина цикла, сек")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--curves", type=int, default=2)
    ap.add_argument("--style", choices=["lines", "beams"], default="lines",
                    help="beams = два пучка света по углам + еле заметные медленные линии")
    ap.add_argument("--palette", choices=sorted(PALETTES), default="blue")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--frames", type=int, default=0, help="отрендерить только N кадров (тест)")
    a = ap.parse_args()
    if a.preview:
        a.width, a.duration, a.fps = 960, 6.0, 24
    W = a.width - a.width % 2
    H = (W * 9 // 16) // 2 * 2
    N = int(round(a.duration * a.fps))
    pal = dict(PALETTES[a.palette])
    if a.style == "beams":  # линии почти невидимы, фон тёмный: пучки — главные
        pal["line"] *= 0.14; pal["glow"] *= 0.5; pal["blob"] *= 0.2
    cfg = {"W": W, "H": H, "N": N, "pal": pal, "style": a.style, "seed": a.seed}
    scene = make_scene(a.seed, a.curves)
    total = a.frames or N
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "14",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", a.out],
        stdin=subprocess.PIPE)
    with Pool(4, initializer=init, initargs=(scene, cfg)) as pool:
        for n, fr in enumerate(pool.imap(render, range(total), chunksize=2)):
            ff.stdin.write(fr.tobytes())
            if n % 30 == 0:
                print(f"{n}/{total}", flush=True)
    ff.stdin.close(); ff.wait()
    print("готово:", a.out, f"{W}x{H}, {N} кадров, {a.fps} fps")


if __name__ == "__main__":
    main()
