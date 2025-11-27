import imageio.v2 as imageio
import pygame
import os

class VideoExporter:
    def __init__(self, path, fps=15):
        path = self._unique_path(path)
        self.path = path
        self.fps = fps
        self.frames = []

    def _unique_path(self, path):
        base, ext = os.path.splitext(path)
        if not os.path.exists(path):
            return path
        idx = 1
        while True:
            candidate = f"{base}_{idx}{ext}"
            if not os.path.exists(candidate):
                return candidate
            idx += 1

    def add_frame(self, surface):
        frame_str = pygame.image.tostring(surface, "RGB")
        w, h = surface.get_size()
        image = pygame.image.fromstring(frame_str, (w, h), "RGB")
        raw = pygame.surfarray.array3d(image)
        frame = raw.swapaxes(0, 1)  # transpose to match imageio
        self.frames.append(frame)

    def export(self):
        imageio.mimsave(self.path, self.frames, fps=self.fps)