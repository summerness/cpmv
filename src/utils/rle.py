import numpy as np


def rle_encode(mask: np.ndarray) -> str:
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask = mask.astype(np.uint8)
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if len(runs) == 0:
        return ""
    return " ".join(str(x) for x in runs)


def rle_decode(rle: str, shape) -> np.ndarray:
    s = list(map(int, rle.split())) if rle else []
    starts, lengths = s[0::2], s[1::2]
    starts = np.array(starts) - 1
    ends = starts + np.array(lengths)
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")
