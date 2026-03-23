"""
This file contains utility functions for visualizing image observations in the training pipeline.
These functions can be a useful debugging tool.
"""
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend: save-only, no display
import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils


def _chw_to_hwc_uint8(x: np.ndarray) -> np.ndarray:
    """Single frame (C, H, W) -> (H, W, 3) uint8 for PNG."""
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError("expected CHW with ndim=3, got shape {}".format(x.shape))
    c, h, w = x.shape
    if c == 1:
        x = np.repeat(x, 3, axis=0)
    elif c == 4:
        x = x[:3]
    elif c != 3:
        raise ValueError("expected C in {{1,3,4}}, got C={}".format(c))
    x = np.transpose(x, (1, 2, 0))
    if x.dtype in (np.float32, np.float64):
        if np.nanmax(x) <= 1.0 + 1e-3:
            x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        else:
            x = np.clip(x, 0, 255).astype(np.uint8)
    elif x.dtype != np.uint8:
        x = np.clip(x.astype(np.float64), 0, 255).astype(np.uint8)
    return x

def save_bchw_images_to_tmp(images, prefix="obs", out_dir="/tmp"):
    """
    Save a batch of channel-first images ``(B, C, H, W)`` as PNGs under ``out_dir`` (default ``/tmp``).

    Typical use: ``C=3`` (RGB). Also supports ``C in {1, 4}`` (grayscale repeated to RGB, or RGBA -> RGB).
    Accepts ``torch.Tensor`` or ``numpy.ndarray``. Float in ~[0, 1] is scaled to uint8; else clipped to [0, 255].

    Args:
        images: shape (B, C, H, W)
        prefix: filename prefix
        out_dir: output directory (created if missing)

    Returns:
        list of str: absolute paths ``{prefix}_b{i}.png`` for ``i = 0 .. B-1``.
    """
    x = TensorUtils.to_numpy(images)
    if x.ndim != 4:
        raise ValueError("expected shape (B, C, H, W), got {}".format(x.shape))
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    b = x.shape[0]
    for bi in range(b):
        hwc = _chw_to_hwc_uint8(x[bi])
        fname = os.path.join(out_dir, "{}_b{}.png".format(prefix, bi))
        Image.fromarray(hwc).save(fname)
        saved.append(os.path.abspath(fname))
    return saved


def save_btchw_images_to_tmp(images, prefix="obs", out_dir="/tmp"):
    """
    Save a batch of image sequences (B, T, C, H, W) as PNGs under ``out_dir`` (default ``/tmp``).

    Accepts ``torch.Tensor`` or ``numpy.ndarray``. Assumes channel-first (C, H, W) per frame.
    Float in ~[0, 1] is scaled to uint8; else clipped to [0, 255].

    Args:
        images: shape (B, T, C, H, W)
        prefix: filename prefix
        out_dir: directory to write into (created if missing)

    Returns:
        list of str: absolute paths ``{prefix}_b{bi}_t{ti}.png`` for each (bi, ti).
    """
    x = TensorUtils.to_numpy(images)
    if x.ndim != 5:
        raise ValueError("expected shape (B, T, C, H, W), got {}".format(x.shape))
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    b, t = x.shape[0], x.shape[1]
    for bi in range(b):
        for ti in range(t):
            hwc = _chw_to_hwc_uint8(x[bi, ti])
            fname = os.path.join(out_dir, "{}_b{}_t{}.png".format(prefix, bi, ti))
            Image.fromarray(hwc).save(fname)
            saved.append(os.path.abspath(fname))
    return saved


def image_tensor_to_numpy(image):
    """
    Converts processed image tensors to numpy so that they can be saved to disk or video.
    A useful utility function for visualizing images in the middle of training.

    Args:
        image (torch.Tensor): images of shape [..., C, H, W]

    Returns:
        image (np.array): converted images of shape [..., H, W, C] and type uint8
    """
    return TensorUtils.to_numpy(
            ObsUtils.unprocess_image(image)
        ).astype(np.uint8)


def image_to_disk(image, fname):
    """
    Writes an image to disk.

    Args:
        image (np.array): image of shape [H, W, 3]
        fname (str): path to save image to
    """
    image = Image.fromarray(image)
    image.save(fname)


def image_tensor_to_disk(image, fname):
    """
    Writes an image tensor to disk. Any leading batch dimensions are indexed out
    with the first element.

    Args:
        image (torch.Tensor): image of shape [..., C, H, W]. All leading dimensions
            will be indexed out with the first element
        fname (str): path to save image to
    """
    # index out all leading dimensions before [C, H, W]
    num_leading_dims = len(image.shape[:-3])
    for _ in range(num_leading_dims):
        image = image[0]
    image = image_tensor_to_numpy(image)
    image_to_disk(image, fname)


def visualize_image_randomizer(original_image, randomized_image, randomizer_name=None):
    """
    A function that visualizes the before and after of an image-based input randomizer
    Args:
        original_image: batch of original image shaped [B, H, W, 3]
        randomized_image: randomized image shaped [B, N, H, W, 3]. N is the number of randomization per input sample
        randomizer_name: (Optional) name of the randomizer
    Returns:
        None
    """

    B, N, H, W, C = randomized_image.shape

    # Create a grid of subplots with B rows and N+1 columns (1 for the original image, N for the randomized images)
    fig, axes = plt.subplots(B, N + 1, figsize=(4 * (N + 1), 4 * B))

    for i in range(B):
        # Display the original image in the first column of each row
        axes[i, 0].imshow(original_image[i])
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        # Display the randomized images in the remaining columns of each row
        for j in range(N):
            axes[i, j + 1].imshow(randomized_image[i, j])
            axes[i, j + 1].axis("off")

    title = randomizer_name if randomizer_name is not None else "Randomized"
    fig.suptitle(title, fontsize=16)

    # Adjust the space between subplots for better visualization
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # Show the entire grid of subplots
    plt.show()


def make_model_prediction_plot(
    hdf5_path,
    save_path,
    images,
    action_names,
    actual_actions,
    predicted_actions,
):
    """
    TODO: documentation
    actual_actions: (T, D)
    predicted_actions: (T, D)
    """
    image_keys = sorted(list(images.keys()))
    action_dim = actual_actions.shape[1]
    traj_length = len(actual_actions)
    if traj_length < 15: 
        return

    # Plot
    fig, axs = plt.subplots(len(images) + action_dim, 1, figsize=(30, (len(images) + action_dim) * 3))
    for i, image_key in enumerate(image_keys):
        interval = int(traj_length/15) # plot `5` images
        images[image_key] = images[image_key][::interval]
        combined_images = np.concatenate(images[image_key], axis=1)

        ## If doing goal image conditioniing visualize goal ims below
        if combined_images.shape[-1] == 6:
            c1 = combined_images[:, :, :3]
            c2 = combined_images[:, :, 3:]
            combined_images = np.concatenate([c1, c2], 0)
            
        axs[i].imshow(combined_images)
        if i == 0:
            axs[i].set_title(hdf5_path + '\n' + image_key, fontsize=30)
        else:
            axs[i].set_title(image_key, fontsize=30)
        axs[i].axis("off")
    for dim in range(action_dim):
        ax = axs[len(images)+dim]
        ax.plot(range(traj_length), actual_actions[:, dim], label='Actual Action', color='blue')
        ax.plot(range(traj_length), predicted_actions[:, dim], label='Predicted Action', color='red')
        # ax.set_xlabel('Timestep')
        # ax.set_ylabel('Action Dimension {}'.format(dim + 1))
        ax.set_title(action_names[dim], fontsize=30)
        ax.xaxis.set_tick_params(labelsize=24)
        ax.yaxis.set_tick_params(labelsize=24)
        ax.legend(fontsize=20)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.6)
    
    # Save the figure with the specified path and filename
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_path) 

    fig.clear()
    plt.close()
    plt.cla()
    plt.clf()


def make_batch_vis_plot(
    save_path,
    images,
):
    image_keys = sorted(list(images.keys()))

    # Plot
    fig, axs = plt.subplots(len(images), 1, figsize=(30, (len(images)) * 3))
    for i, image_key in enumerate(image_keys):
        combined_images = images[image_key].cpu().detach().numpy()
        combined_images = [combined_images[i] for i in range(combined_images.shape[0])]
        combined_images = np.concatenate(combined_images, axis=1)
        axs[i].imshow(combined_images)
        axs[i].set_title(image_key, fontsize=30)
        axs[i].axis("off")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.6)

    # Save the figure with the specified path and filename
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_path)

    fig.clear()
    plt.close()
    plt.cla()
    plt.clf()