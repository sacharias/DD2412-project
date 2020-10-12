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
        _, h, w = img.shape

        # Draw random center point
        y = np.random.randint(h)
        x = np.random.randint(w)

        # Set corners of the cutout mask
        offset = self.size // 2
        y1 = np.clip(y - offset, 0, h)
        y2 = np.clip(y + offset, 0, h)
        x1 = np.clip(x - offset, 0, w)
        x2 = np.clip(x + offset, 0, w)

        # Apply cutout mask
        img[:, y1:y2, x1:x2] = 0

        return img