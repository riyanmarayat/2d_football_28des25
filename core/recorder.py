import imageio

class Recorder:
    def __init__(self):
        self.frames = []

    def log(self, state):
        self.frames.append(state.copy())

    def export(self, path):
        imageio.mimsave(path, self.frames, fps=15)