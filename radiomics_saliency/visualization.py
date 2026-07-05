import numpy as np
import skimage.io as skio
from skimage.util import img_as_ubyte
from skimage import img_as_float
from skimage.color import gray2rgba


def blend_image_with_mask(filename: str, image: np.ndarray, mask: np.ndarray):
    # Ensure image is RGB (drop alpha if present)
    img = image.copy()
    if img.ndim == 2:
        # grayscale -> convert to RGB
        img = gray2rgba(img)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    # convert to float [0,1]
    img_f = img_as_float(img)
    # treat positive values as red, negative as blue; zeros transparent
    m = mask.astype('float32')
    eps = np.finfo(float).eps
    pos = np.clip(m, 0.0, None)
    neg = np.clip(-m, 0.0, None)
    max_pos = pos.max() if pos.size else 0.0
    max_neg = neg.max() if neg.size else 0.0
    pos_n = pos / (max_pos + eps)
    neg_n = neg / (max_neg + eps)

    # per-pixel magnitude (0 for zeros -> fully transparent)
    mask_mag = np.clip(pos_n + neg_n, 0.0, 1.0)

    # overlay colors
    overlay_pos = np.stack([pos_n, np.zeros_like(pos_n), np.zeros_like(pos_n)], axis=-1)
    overlay_neg = np.stack([np.zeros_like(neg_n), np.zeros_like(neg_n), neg_n], axis=-1)
    overlay = overlay_pos + overlay_neg

    alpha = 0.7
    # alpha blending where alpha is scaled by mask magnitude; zeros remain original
    blended = (1.0 - (alpha * mask_mag)[..., None]) * img_f + (alpha * mask_mag)[..., None] * overlay

    # save as uint8 image
    skio.imsave(filename, img_as_ubyte(np.clip(blended, 0.0, 1.0)))


def prepare_image_for_serialization(image):
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        if image.size == 0:
            return image.astype('uint8')
        if image.min() >= 0.0 and image.max() <= 1.0:
            return img_as_ubyte(image)
    return np.clip(image, 0, 255).astype('uint8')


def prepare_images_for_serialization(images):
    return np.asarray([prepare_image_for_serialization(image) for image in images], dtype='uint8')
