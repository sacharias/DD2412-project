import numpy as np

class CutoutTransform:
    """Implementation of Cutout.

    For CIFAR-10, use size 16.
    For CIFAR-100, use size 8.

    See: https://arxiv.org/abs/1708.04552
    """

    def __init__(self, size=16):
        self.size = size

    def __call__(self, img):
        w, h = img.size

        # Pick random center point
        y = np.random.randint(h)
        x = np.random.randint(w)

        # Calculate corners of cutout mask
        offset = self.size // 2
        y1 = np.clip(y - offset, 0, h)
        y2 = np.clip(y + offset, 0, h)
        x1 = np.clip(x - offset, 0, w)
        x2 = np.clip(x + offset, 0, w)

        # Apply cutout mask
        pixels = img.load()
        for i in range(x1, x2):
            for j in range(y1, y2):
                pixels[i, j] = (127, 127, 127)

        return img
