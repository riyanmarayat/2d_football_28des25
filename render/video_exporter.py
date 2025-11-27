import imageio.v2 as imageio
import pygame

class VideoExporter:
    def __init__(self, filename="simulation.mp4", fps=15):
        self.filename = filename
        self.fps = fps
        self.frames = []

    def add_frame(self, surface):
        frame_str = pygame.image.tostring(surface, "RGB")
        w, h = surface.get_size()
        image = pygame.image.fromstring(frame_str, (w, h), "RGB")
        raw = pygame.surfarray.array3d(image)
        frame = raw.swapaxes(0, 1)  # transpose to match imageio
        self.frames.append(frame)

    def export(self):
        imageio.mimsave(self.filename, self.frames, fps=self.fps)