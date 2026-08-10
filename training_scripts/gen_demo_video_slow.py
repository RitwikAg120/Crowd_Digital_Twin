"""
Slow-motion variant of the synthetic crowd video, tuned so that ByteTrack's
IoU-based association still works correctly even at the ~1-2 effective FPS
achievable on a single CPU core in this sandbox (vs. 70.8 FPS on the Tesla T4
GPU described in the report). Movement-per-processed-frame is kept small.
"""
import cv2
import numpy as np
import random

random.seed(7)
np.random.seed(7)

W, H = 1280, 720
N_FRAMES = 1400
FPS = 30
N_PEOPLE = 22
SPEED_SCALE = 0.18   # much slower drift so low-FPS CPU tracking stays coherent

class Walker:
    def __init__(self, idx):
        self.idx = idx
        edge = random.choice(["left", "right", "top", "bottom"])
        if edge == "left":
            self.x, self.y = random.uniform(60, 300), random.uniform(100, H - 100)
            self.vx, self.vy = random.uniform(0.4, 0.8), random.uniform(-0.15, 0.15)
        elif edge == "right":
            self.x, self.y = random.uniform(W-300, W-60), random.uniform(100, H - 100)
            self.vx, self.vy = -random.uniform(0.4, 0.8), random.uniform(-0.15, 0.15)
        elif edge == "top":
            self.x, self.y = random.uniform(100, W - 100), random.uniform(60, 300)
            self.vx, self.vy = random.uniform(-0.15, 0.15), random.uniform(0.3, 0.7)
        else:
            self.x, self.y = random.uniform(100, W - 100), random.uniform(H-300, H-60)
            self.vx, self.vy = random.uniform(-0.15, 0.15), -random.uniform(0.3, 0.7)
        self.color = tuple(int(c) for c in np.random.randint(60, 230, size=3))
        self.body_w = random.randint(10, 16)
        self.body_h = random.randint(34, 50)
        self.cluster_bias = random.random() < 0.35
        self.cluster_pt = (random.uniform(W*0.35, W*0.65), random.uniform(H*0.35, H*0.65))

    def step(self, t):
        if self.cluster_bias and 500 < t < 1000:
            dx = self.cluster_pt[0] - self.x
            dy = self.cluster_pt[1] - self.y
            dist = max(1.0, (dx**2 + dy**2) ** 0.5)
            self.vx += (dx / dist) * 0.015
            self.vy += (dy / dist) * 0.015
            self.vx = np.clip(self.vx, -1.0, 1.0)
            self.vy = np.clip(self.vy, -1.0, 1.0)
        self.x += (self.vx + np.random.normal(0, 0.04)) * 1.0
        self.y += (self.vy + np.random.normal(0, 0.04)) * 1.0
        if self.x < 20: self.x = 20; self.vx = abs(self.vx)
        if self.x > W - 20: self.x = W - 20; self.vx = -abs(self.vx)
        if self.y < 20: self.y = 20; self.vy = abs(self.vy)
        if self.y > H - 20: self.y = H - 20; self.vy = -abs(self.vy)

    def draw(self, img):
        cx, cy = int(self.x), int(self.y)
        head_r = self.body_w // 2 + 2
        head_cy = cy - self.body_h // 2 - head_r
        cv2.circle(img, (cx, head_cy), head_r, self.color, -1)
        cv2.ellipse(img, (cx, cy), (self.body_w, self.body_h // 2), 0, 0, 360, self.color, -1)
        cv2.circle(img, (cx, head_cy), head_r, (10,10,10), 1)
        cv2.ellipse(img, (cx, cy), (self.body_w, self.body_h // 2), 0, 0, 360, (10,10,10), 1)

walkers = [Walker(i) for i in range(N_PEOPLE)]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("videos/synthetic_crowd_demo_slow.mp4", fourcc, FPS, (W, H))

for t in range(N_FRAMES):
    frame = np.full((H, W, 3), (35, 38, 42), dtype=np.uint8)
    for gx in range(0, W, 80):
        cv2.line(frame, (gx, 0), (gx, H), (45, 48, 52), 1)
    for gy in range(0, H, 80):
        cv2.line(frame, (0, gy), (W, gy), (45, 48, 52), 1)
    for w in walkers:
        w.step(t)
        w.draw(frame)
    out.write(frame)

out.release()
print(f"Wrote videos/synthetic_crowd_demo_slow.mp4 ({N_FRAMES} frames @ {FPS}fps, {N_PEOPLE} agents, all start pre-populated in-frame)")
