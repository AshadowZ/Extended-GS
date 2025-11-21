# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
import cv2
import sys
import numpy as np
import subprocess
import time
import threading
from queue import Queue

cmd = "nvidia-smi -q -d Memory |grep -A4 GPU|grep Used"
result = (
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split("\n")
)
os.environ["CUDA_VISIBLE_DEVICES"] = str(
    np.argmin([int(x.split()[2]) for x in result[:-1]])
)

os.system("echo $CUDA_VISIBLE_DEVICES")

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import warnings
from tqdm import tqdm
import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
import warnings

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore", category=UserWarning)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_boundary(mask, kernel_size_erode=5):
    kernel_erode = np.ones(
        (kernel_size_erode, kernel_size_erode), np.uint8
    )  # rubbish -> 15
    mask = np.float32(mask)  # 255
    mask = mask - cv2.erode(mask, kernel_erode, iterations=1)

    return mask > 0


def vis_mask(image, masks):
    colors = np.random.rand(
        len(masks), 3
    )  # plt.cm.hsv(np.linspace(0, 1, masks.max() + 1))[:, :3]
    image_copy = image.copy()
    for mask_id, mask in enumerate(masks):
        # ensure mask is boolean for safe array indexing
        mask = mask.astype(bool)
        color = colors[mask_id] * 255
        image_copy[mask] = np.uint8(image[mask] * 0.3 + 255.0 * 0.7 * colors[mask_id])
        boundary = get_boundary(np.uint8(mask * 255.0), kernel_size_erode=3)
        image_copy[boundary] = np.uint8(255.0 * colors[mask_id] * 0.75)
    if False:
        for mask_id, mask in enumerate(masks):
            color = colors[mask_id] * 255
            mask_coords = np.argwhere(mask)
            y_min, x_min = mask_coords.min(axis=0)
            y_max, x_max = mask_coords.max(axis=0)

            # Draw the bounding box on the overlay image
            cv2.rectangle(
                image_copy,
                (x_min, y_min),
                (x_max, y_max),
                (int(color[0]), int(color[1]), int(color[2])),
                thickness=1,
            )

            # Annotate the mask ID in the bounding box
            cv2.putText(
                image_copy,
                f"ID: {mask_id}",
                (x_min + 5, y_min + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (int(color[0]), int(color[1]), int(color[2])),
                max(1, 2),
            )

    return image_copy


def vis_mask_fast(image, mask_id_map):
    """
    image: RGB image (H, W, 3)
    mask_id_map: uint8/uint16 map with 0 as background and positive ids as instances
    """
    num_instances = mask_id_map.max()
    if num_instances == 0:
        return image

    # 1) random color table (id 0 stays black)
    colors = np.random.randint(0, 255, (num_instances + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]

    # 2) lookup colors for whole image
    mask_colors = colors[mask_id_map]

    # 3) blend once using broadcasting
    foreground = (mask_id_map > 0)[..., None]
    vis_image = np.where(
        foreground,
        cv2.addWeighted(image, 0.4, mask_colors, 0.6, 0),
        image,
    )

    # 4) boundaries via morphological gradient
    kernel = np.ones((3, 3), np.uint8)
    gradient = cv2.morphologyEx(mask_id_map, cv2.MORPH_GRADIENT, kernel)
    vis_image[gradient > 0] = [255, 255, 255]

    return vis_image


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--scene_dir", type=str)
    parser.add_argument("--image_path_pattern", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    seq_dir = args.scene_dir
    image_list = sorted(glob.glob(os.path.join(seq_dir, args.image_path_pattern)))
    print(os.path.join(seq_dir, args.image_path_pattern))

    output_vis_dir = os.path.join(seq_dir, "sam_vis/mask")
    output_seg_dir = os.path.join(seq_dir, "sam/mask")
    os.makedirs(output_vis_dir, exist_ok=True)
    os.makedirs(output_seg_dir, exist_ok=True)

    read_queue: Queue = Queue(maxsize=8)
    # unbounded write queue so writer threads handle backpressure, not the infer loop
    write_queue: Queue = Queue()
    stop_token = object()

    def reader_worker():
        for path in image_list:
            t_read_start = time.perf_counter()
            img_bgr = read_image(path, format="BGR")
            t_read = time.perf_counter() - t_read_start
            read_queue.put((path, img_bgr, t_read))
        read_queue.put(stop_token)

    def writer_worker(worker_id: int):
        while True:
            item = write_queue.get()
            if item is stop_token:
                write_queue.task_done()
                break

            path, img_bgr, pred_masks, pred_scores, t_read = item

            # move masks/scores to CPU and apply threshold filter here
            pred_masks_cpu = pred_masks.cpu()
            pred_scores_cpu = pred_scores.cpu()
            selected_indexes = pred_scores_cpu >= args.confidence_threshold
            selected_scores = pred_scores_cpu[selected_indexes]
            selected_masks = pred_masks_cpu[selected_indexes]

            # t_post_start = time.perf_counter()
            if selected_masks.numel() > 0:
                _, m_H, m_W = selected_masks.shape
            else:
                m_H, m_W = pred_masks_cpu.shape[1], pred_masks_cpu.shape[2]
            mask_image = np.zeros((m_H, m_W), dtype=np.uint8)

            if selected_scores.numel() > 0:
                selected_scores_sorted, ranks = torch.sort(selected_scores)
                mask_id = 1
                for index in ranks:
                    num_pixels = torch.sum(selected_masks[index])
                    if num_pixels < 400:
                        # ignore small masks
                        continue
                    mask_image[(selected_masks[index] == 1).cpu().numpy()] = mask_id
                    mask_id += 1
            # t_post = time.perf_counter() - t_post_start

            # t_save_seg_start = time.perf_counter()
            cv2.imwrite(
                os.path.join(
                    output_seg_dir, os.path.basename(path).split(".")[0] + ".png"
                ),
                mask_image,
                [cv2.IMWRITE_PNG_COMPRESSION, 1],  # lower compression for faster write
            )
            # t_save_seg = time.perf_counter() - t_save_seg_start

            # t_vis_start = time.perf_counter()
            save_vis_file = os.path.join(
                output_vis_dir, os.path.basename(path).split(".")[0] + ".png"
            )
            img_rgb = img_bgr[:, :, ::-1]  # convert in writer
            vis_img = vis_mask_fast(img_rgb, mask_image)  # RGB
            cv2.imwrite(
                save_vis_file,
                vis_img[:, :, ::-1],  # convert to BGR for OpenCV
                [cv2.IMWRITE_PNG_COMPRESSION, 1],
            )
            # t_vis = time.perf_counter() - t_vis_start

            # print timing logs if needed
            write_queue.task_done()

    reader_thread = threading.Thread(target=reader_worker, daemon=True)
    reader_thread.start()

    num_writer_threads = min(2, os.cpu_count() or 1)
    print(f"[INFO] writer threads: {num_writer_threads}")
    writer_threads = []
    for wid in range(num_writer_threads):
        t = threading.Thread(target=writer_worker, args=(wid,), daemon=True)
        t.start()
        writer_threads.append(t)

    pbar = tqdm(total=len(image_list))

    while True:
        item = read_queue.get()
        if item is stop_token:
            read_queue.task_done()
            break

        path, img_bgr, t_read = item
        predictions = demo.run_on_image(img_bgr)

        pred_masks = predictions["instances"].pred_masks
        pred_scores = predictions["instances"].scores

        write_queue.put((path, img_bgr, pred_masks, pred_scores, t_read))
        read_queue.task_done()
        pbar.update(1)

    write_queue.join()
    for _ in writer_threads:
        write_queue.put(stop_token)
    for t in writer_threads:
        t.join()
    pbar.close()
