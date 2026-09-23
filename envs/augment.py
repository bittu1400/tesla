"""Image-space domain randomization (training) and held-out perturbations (evaluation).

Every augmenter has the same interface:
    aug = Augmenter(rng)       # rng: np.random.Generator
    aug.new_episode()          # sample per-episode settings (also done once by the constructor)
    aug(frame, t) -> frame     # uint8 120x160x3 RGB in and out; t = step within the episode

Training ranges and evaluation strengths never overlap, so a model trained
with DR can't pass an evaluation condition just by having seen it:
  low light  eval brightness 0.35   < training minimum 0.6
  shadows    eval: moving band, darkness 0.3; training: static polygons, darkness >= 0.5
  distractors eval only
"""
import cv2
import numpy as np


def _polygon_mask(shape, points) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 1)
    return mask.astype(bool)


class DomainRandomizer:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.new_episode()

    def new_episode(self) -> None:
        r = self.rng
        self.brightness = r.uniform(0.6, 1.4)
        self.contrast = r.uniform(0.7, 1.3)
        self.saturation = r.uniform(0.7, 1.3)
        self.hue_shift = r.uniform(-8.0, 8.0)  # OpenCV hue units (0-180)
        self.shadow_polygons = [
            r.uniform((0, 0), (160, 120), size=(int(r.integers(3, 6)), 2)) for _ in range(int(r.integers(0, 3)))
        ]
        self.shadow_darkness = r.uniform(0.5, 0.8)
        self.noise_std = r.uniform(0.0, 8.0)

    def __call__(self, frame, t: int = 0) -> np.ndarray:
        hsv = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + self.hue_shift) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * self.saturation, 0, 255)
        x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        mean = x.mean()
        x = ((x - mean) * self.contrast + mean) * self.brightness
        for polygon in self.shadow_polygons:
            x[_polygon_mask(x.shape, polygon)] *= self.shadow_darkness
        x += self.rng.normal(0.0, self.noise_std, x.shape)
        out = np.clip(x, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(out, (3, 3), 0) if self.rng.random() < 0.3 else out


class LowLight:
    """Dim scene: darker image with more sensor noise."""

    BRIGHTNESS = 0.35
    NOISE_STD = 6.0

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def new_episode(self) -> None:
        pass

    def __call__(self, frame, t: int = 0) -> np.ndarray:
        x = np.asarray(frame, dtype=np.float32) * self.BRIGHTNESS
        x += self.rng.normal(0.0, self.NOISE_STD, x.shape)
        return np.clip(x, 0, 255).astype(np.uint8)


class MovingShadow:
    """A dark diagonal band sweeping sideways across the image, like the
    shadow of something moving past the track."""

    DARKNESS = 0.3
    WIDTH = 30  # pixels
    SPEED = 4  # pixels per step

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.new_episode()

    def new_episode(self) -> None:
        self.offset = int(self.rng.integers(0, 1000))

    def __call__(self, frame, t: int = 0) -> np.ndarray:
        x = np.asarray(frame, dtype=np.float32)
        h, w = x.shape[:2]
        period = w + h + 2 * self.WIDTH
        left = (self.offset + self.SPEED * t) % period - self.WIDTH - h
        ys, xs = np.mgrid[0:h, 0:w]
        d = xs - ys - left
        x[(d >= 0) & (d < self.WIDTH)] *= self.DARKNESS
        return x.astype(np.uint8)


class Distractors:
    """Three saturated rectangles (objects beside the track), fixed per episode."""

    COLOURS = ((220, 30, 30), (30, 60, 220), (240, 140, 0))

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.new_episode()

    def new_episode(self) -> None:
        self.rects = []
        for colour in self.COLOURS:
            w, h = (int(v) for v in self.rng.integers(12, 26, size=2))
            x = int(self.rng.integers(0, 160 - w))
            y = int(self.rng.integers(50, 120 - h))  # below the cropped top rows
            self.rects.append((x, y, w, h, colour))

    def __call__(self, frame, t: int = 0) -> np.ndarray:
        out = np.array(frame, dtype=np.uint8, copy=True)
        for x, y, w, h, colour in self.rects:
            out[y : y + h, x : x + w] = colour
        return out


AUGMENTERS = {"dr": DomainRandomizer, "low_light": LowLight, "shadows": MovingShadow, "distractors": Distractors}


def make_augmenter(name: str | None, rng: np.random.Generator):
    return None if name is None else AUGMENTERS[name](rng)
