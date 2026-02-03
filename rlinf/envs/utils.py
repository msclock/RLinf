# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, Optional, Union

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

def to_tensor(
    array: Union[dict, torch.Tensor, np.ndarray, list, Any], device: str = "cpu"
) -> Union[dict, torch.Tensor]:
    """
    Copied from ManiSkill!
    Maps any given sequence to a torch tensor on the CPU/GPU. If physx gpu is not enabled then we use CPU, otherwise GPU, unless specified
    by the device argument

    Args:
        array: The data to map to a tensor
        device: The device to put the tensor on. By default this is None and to_tensor will put the device on the GPU if physx is enabled
            and CPU otherwise

    """
    if isinstance(array, (dict)):
        return {k: to_tensor(v, device=device) for k, v in array.items()}
    elif isinstance(array, torch.Tensor):
        ret = array.to(device)
    elif isinstance(array, np.ndarray):
        if array.dtype == np.uint16:
            array = array.astype(np.int32)
        elif array.dtype == np.uint32:
            array = array.astype(np.int64)
        ret = torch.tensor(array).to(device)
    else:
        if isinstance(array, list) and isinstance(array[0], np.ndarray):
            ret = torch.tensor(np.array(array), device=device)
        elif isinstance(array, list) and isinstance(array[0], torch.Tensor):
            ret = torch.stack(array).to(device)
        else:
            ret = torch.tensor(array, device=device)
    if ret.dtype == torch.float64:
        ret = ret.to(torch.float32)
    return ret

def list_of_dict_to_dict_of_list(
    list_of_dict: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    """
    Convert a list of dictionaries to a dictionary of lists.

    Args:
        list_of_dict: List of dictionaries with same keys

    Returns:
        Dictionary where each key maps to a list of values
    """
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()
    output = {key: [] for key in keys}
    for data in list_of_dict:
        for key, item in data.items():
            assert key in output
            output[key].append(item)
    return output

def save_rollout_video(
    rollout_images: list[np.ndarray], output_dir: str, video_name: str, fps: int = 30
) -> None:
    """
    Saves an MP4 replay of an episode.

    Args:
        rollout_images: List of images from the episode
        output_dir: Directory to save the video
        video_name: Name of the output video file
        fps: Frames per second for the video
    """
    os.makedirs(output_dir, exist_ok=True)
    mp4_path = os.path.join(output_dir, f"{video_name}.mp4")
    video_writer = imageio.get_writer(mp4_path, fps=fps)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()

def tile_images(
    images: list[Union[np.ndarray, torch.Tensor]], nrows: int = 1
) -> Union[np.ndarray, torch.Tensor]:
    """
    Copied from maniskill https://github.com/haosulab/ManiSkill
    Tile multiple images to a single image comprised of nrows and an appropriate number of columns to fit all the images.
    The images can also be batched (e.g. of shape (B, H, W, C)), but give images must all have the same batch size.

    if nrows is 1, images can be of different sizes. If nrows > 1, they must all be the same size.
    """
    # Sort images in descending order of vertical height
    batched = False
    if len(images[0].shape) == 4:
        batched = True
    if nrows == 1:
        images = sorted(images, key=lambda x: x.shape[0 + batched], reverse=True)

    columns: list[list[Union[np.ndarray, torch.Tensor]]] = []
    if batched:
        max_h = images[0].shape[1] * nrows
        cur_h = 0
        cur_w = images[0].shape[2]
    else:
        max_h = images[0].shape[0] * nrows
        cur_h = 0
        cur_w = images[0].shape[1]

    # Arrange images in columns from left to right
    column = []
    for im in images:
        if cur_h + im.shape[0 + batched] <= max_h and cur_w == im.shape[1 + batched]:
            column.append(im)
            cur_h += im.shape[0 + batched]
        else:
            columns.append(column)
            column = [im]
            cur_h, cur_w = im.shape[0 + batched : 2 + batched]
    columns.append(column)

    # Tile columns
    total_width = sum(x[0].shape[1 + batched] for x in columns)

    is_torch = False
    if torch is not None:
        is_torch = isinstance(images[0], torch.Tensor)

    output_shape = (max_h, total_width, 3)
    if batched:
        output_shape = (images[0].shape[0], max_h, total_width, 3)
    if is_torch:
        output_image = torch.zeros(output_shape, dtype=images[0].dtype)
    else:
        output_image = np.zeros(output_shape, dtype=images[0].dtype)
    cur_x = 0
    for column in columns:
        cur_w = column[0].shape[1 + batched]
        next_x = cur_x + cur_w
        if is_torch:
            column_image = torch.concatenate(column, dim=0 + batched)
        else:
            column_image = np.concatenate(column, axis=0 + batched)
        cur_h = column_image.shape[0 + batched]
        output_image[..., :cur_h, cur_x:next_x, :] = column_image
        cur_x = next_x
    return output_image

def put_text_on_image(
    image: np.ndarray, lines: list[str], max_width: int = 200
) -> np.ndarray:
    """
    Put text lines on an image with automatic line wrapping.

    Args:
        image: Input image as numpy array
        lines: List of text lines to add
        max_width: Maximum width for text wrapping
    """
    assert image.dtype == np.uint8, image.dtype
    image = image.copy()
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)

    new_lines = []
    for line in lines:
        words = line.split()
        current_line = []

        for word in words:
            test_line = " ".join(current_line + [word])
            test_width = font.getlength(test_line)

            if test_width <= max_width:
                current_line.append(word)
            else:
                new_lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            new_lines.append(" ".join(current_line))

    y = -10
    for line in new_lines:
        bbox = draw.textbbox((0, 0), text=line)
        textheight = bbox[3] - bbox[1]
        y += textheight + 10
        x = 10
        draw.text((x, y), text=line, fill=(0, 0, 0))
    return np.array(image)

def put_info_on_image(
    image: np.ndarray,
    info: dict[str, float],
    extras: Optional[list[str]] = None,
    overlay: bool = True,
) -> np.ndarray:
    """
    Put information dictionary and extra lines on an image.

    Args:
        image: Input image
        info: Dictionary of key-value pairs to display
        extras: Additional text lines to display
        overlay: Whether to overlay text on image
    """
    lines = [
        f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
        for k, v in info.items()
    ]
    if extras is not None:
        lines.extend(extras)
    return put_text_on_image(image, lines)

def crop_and_resize(image, crop_scale, batch_size, out_size=(224, 224)):
    """
    Center-crops an image (or batch of images) to have area `crop_scale` * (original image area),
    and then resizes each crop back to `out_size`.

    Args:
        image: numpy array or PIL Image. Supported shapes: (H, W, C) or (B, H, W, C).
        crop_scale: scalar or array-like of length batch_size. Interpreted as area scale; we use sqrt(crop_scale)
            to get per-side scale like the original TF implementation.
        batch_size: number of images in the batch (if image is single image, batch_size should be 1).
        out_size: (height, width) tuple for the output size.

    Returns:
        numpy array of shape (B, out_h, out_w, C) if input was batched, otherwise (out_h, out_w, C).
    """
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        arr = np.array(image)
    else:
        arr = np.asarray(image)

    single = False
    if arr.ndim == 3:
        # (H, W, C) -> make batch dim
        arr = np.expand_dims(arr, 0)
        single = True
    elif arr.ndim != 4:
        raise ValueError("Unsupported image array shape: %r" % (arr.shape,))

    B, H, W, C = arr.shape
    if B != batch_size:
        # allow batch_size mismatch by using actual batch size
        batch_size = B

    # Normalize crop_scale to array of length batch_size
    if np.isscalar(crop_scale):
        scales = np.full(batch_size, float(crop_scale), dtype=float)
    else:
        scales = np.asarray(crop_scale, dtype=float)
        if scales.shape[0] != batch_size:
            raise ValueError("crop_scale must be scalar or have length batch_size")

    scales = np.clip(np.sqrt(scales), 0.0, 1.0)

    outputs = []
    for i in range(batch_size):
        s = scales[i]
        new_h = max(1, int(round(H * s)))
        new_w = max(1, int(round(W * s)))
        top = (H - new_h) // 2
        left = (W - new_w) // 2
        bottom = top + new_h
        right = left + new_w

        crop = arr[i, top:bottom, left:right]
        pil = Image.fromarray(crop)
        resized = pil.resize((out_size[1], out_size[0]), resample=Image.BILINEAR)
        out_arr = np.asarray(resized)
        # If original had single channel, ensure shape consistency
        if out_arr.ndim == 2:
            out_arr = out_arr[:, :, None]
        outputs.append(out_arr)

    out = np.stack(outputs, axis=0)
    if single:
        return out[0]
    return out

def center_crop_image(image):
    batch_size = 1
    crop_scale = 0.9

    # Accept PIL.Image or numpy array
    if isinstance(image, Image.Image):
        pil = image
    else:
        arr = np.asarray(image)
        # If grayscale array (H, W) or (H, W, 1), convert to RGB for consistent processing
        if arr.ndim == 2:
            pil = Image.fromarray(arr).convert("RGB")
        elif arr.ndim == 3 and arr.shape[2] == 1:
            pil = Image.fromarray(arr.squeeze(-1)).convert("RGB")
        else:
            pil = Image.fromarray(arr)

    orig_mode = pil.mode

    # Convert to uint8 RGB processing to match previous behavior
    pil = pil.convert("RGB")

    # Perform crop and resize using the numpy/Pillow implementation
    cropped_resized = crop_and_resize(np.array(pil), crop_scale, batch_size, out_size=(224, 224))

    # Convert back to PIL Image and to original mode if appropriate
    image_out = Image.fromarray(cropped_resized)
    image_out = image_out.convert("RGB")
    return image_out
