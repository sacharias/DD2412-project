from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import torchvision.transforms.functional as TF
import random


def auto_contrast(x):
    return ImageOps.autocontrast(x)


def brightness(x, min=0.05, max=0.95):
    param = random.uniform(min, max)
    return ImageEnhance.Brightness(x).enhance(param)


def color(x, min=0.05, max=0.95):
    param = random.uniform(min, max)
    return ImageEnhance.Color(x).enhance(param)


def contrast(x, min=0.05, max=0.95):
    param = random.uniform(min, max)
    return ImageEnhance.Contrast(x).enhance(param)


def equalize(x):
    return ImageOps.equalize(x)


def identity(x):
    return x


def posterize(x, min=4, max=8):
    param = random.randint(min, max)
    return ImageOps.posterize(x, bits=param)


def rotate(x, min=-30, max=30):
    param = random.uniform(min, max)
    return TF.rotate(x, param)


def sharpness(x, min=0.05, max=0.95):
    param = random.uniform(min, max)
    return ImageEnhance.Sharpness(x).enhance(param)


def shear_x(x, min=-0.3, max=0.3):
    param = random.uniform(min, max)
    return x.transform(x.size, Image.AFFINE, (1, param, 0, 0, 1, 0))


def shear_y(x, min=-0.3, max=0.3):
    param = random.uniform(min, max)
    return x.transform(x.size, Image.AFFINE, (1, 0, 0, param, 1, 0))


def solarize(x, min=0, max=1):
    param = random.uniform(min, max) * 255.999
    return ImageOps.solarize(x, param)


def translate_x(x, min=-0.3, max=0.3):
    param = random.uniform(min, max) * x.size[0]
    return x.transform(x.size, Image.AFFINE, (1, 0, param, 0, 1, 0))


def translate_y(x, min=-0.3, max=0.3):
    param = random.uniform(min, max) * x.size[1]
    return x.transform(x.size, Image.AFFINE, (1, 0, 0, 0, 1, param))


transformations = [
    auto_contrast,
    brightness,
    color,
    contrast,
    equalize,
    identity,
    posterize,
    rotate,
    sharpness,
    shear_x,
    shear_y,
    solarize,
    translate_x,
    translate_y,
]


class RandAugmentTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        no_of_transformations = random.randint(1, len(transformations))
        operations = random.choices(transformations, k=no_of_transformations)

        for op in operations:
            x = op(x)

        return x