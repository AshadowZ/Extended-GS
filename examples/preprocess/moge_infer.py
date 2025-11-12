import argparse
import os
import threading
import time
from queue import Queue

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel  # 使用 MoGe-2
from tqdm import tqdm


def run_inference(
    data_dir,
    image_dir_name="images",
    read_threads=6,
    write_threads=3,
    queue_size=8,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("正在加载模型...")
    model_load_start = time.time()
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    model.eval()
    model_load_time = time.time() - model_load_start
    print(f"模型加载完成，耗时: {model_load_time:.2f}秒")

    normal_output_dir = os.path.join(data_dir, "moge_normal")
    os.makedirs(normal_output_dir, exist_ok=True)

    image_dir = os.path.join(data_dir, image_dir_name)
    image_files = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not image_files:
        print(f"未在 {image_dir} 目录下找到可处理的图像。")
        return

    print(f"找到 {len(image_files)} 张图像")
    queue_size = max(1, queue_size)

    filename_queue = Queue()
    data_queue = Queue(maxsize=queue_size)
    save_queue = Queue(maxsize=queue_size)

    total_read_time = 0.0
    total_inference_time = 0.0
    stats_lock = threading.Lock()
    save_stats = {"time": 0.0, "count": 0}

    for fname in image_files:
        filename_queue.put(fname)
    for _ in range(read_threads):
        filename_queue.put(None)

    def reader_worker():
        while True:
            fname = filename_queue.get()
            if fname is None:
                filename_queue.task_done()
                break

            img_path = os.path.join(image_dir, fname)
            read_start = time.time()
            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                read_time = time.time() - read_start
                data_queue.put((fname, None, read_time))
                print(f"[WARN] 无法读取图像 {img_path}")
                filename_queue.task_done()
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_rgb = np.ascontiguousarray(img_rgb)
            read_time = time.time() - read_start
            data_queue.put((fname, img_rgb, read_time))
            filename_queue.task_done()

    def writer_worker():
        while True:
            item = save_queue.get()
            if item is None:
                save_queue.task_done()
                break

            base_name, normal_tensor, mask_tensor = item
            save_start = time.time()
            normal_np = normal_tensor.numpy()
            normal_np = normal_np * [-0.5, -0.5, -0.5] + 0.5
            normal_uint8 = np.clip(normal_np * 255.0, 0, 255).astype(np.uint8)

            if mask_tensor is not None:
                mask_np = mask_tensor.numpy()
                mask_bool = mask_np.astype(bool)
                if mask_bool.ndim == 2:
                    mask_bool = mask_bool[:, :, None]
                mask_bool = np.broadcast_to(mask_bool, normal_uint8.shape)
                normal_uint8 = normal_uint8.copy()
                normal_uint8[~mask_bool] = 0

            output_path = os.path.join(normal_output_dir, f"{base_name}.png")
            success = cv2.imwrite(output_path, cv2.cvtColor(normal_uint8, cv2.COLOR_RGB2BGR))
            if not success:
                print(f"[WARN] 无法写入法向图 {output_path}")

            save_time = time.time() - save_start
            with stats_lock:
                save_stats["time"] += save_time
                save_stats["count"] += 1
            save_queue.task_done()

    reader_threads = [
        threading.Thread(target=reader_worker, name=f"reader-{idx}", daemon=True)
        for idx in range(read_threads)
    ]
    writer_threads = [
        threading.Thread(target=writer_worker, name=f"writer-{idx}", daemon=True)
        for idx in range(write_threads)
    ]

    for t in reader_threads + writer_threads:
        t.start()

    processed = 0
    pbar = tqdm(total=len(image_files), desc="MoGe 推理", unit="img")

    try:
        while processed < len(image_files):
            fname, img_rgb, read_time = data_queue.get()
            total_read_time += read_time

            if img_rgb is None:
                tqdm.write(f"跳过图像 {fname}（读取失败）")
                processed += 1
                pbar.update(1)
                continue

            tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).to(device)

            inference_start = time.time()
            with torch.no_grad():
                output = model.infer(tensor)
            inference_time = time.time() - inference_start
            total_inference_time += inference_time

            """
            `output` has keys "points", "depth", "mask", "normal" (optional) and "intrinsics",
            The maps are in the same size as the input image. 
            {
                "points": (H, W, 3),    # point map in OpenCV camera coordinate system (x right, y down, z forward). For MoGe-2, the point map is in metric scale.
                "depth": (H, W),        # depth map
                "normal": (H, W, 3)     # normal map in OpenCV camera coordinate system. (available for MoGe-2-normal)
                "mask": (H, W),         # a binary mask for valid pixels. 
                "intrinsics": (3, 3),   # normalized camera intrinsics
            }
            """

            normal = output.get("normal")
            if normal is not None:
                normal_cpu = normal.detach().cpu()
                mask = output.get("mask")
                mask_for_writer = None
                if mask is not None:
                    mask_cpu = mask.detach().cpu()
                    mask_bool = mask_cpu > 0.5
                    mask_for_zero = mask_bool
                    if mask_for_zero.ndim == 2:
                        mask_for_zero = mask_for_zero.unsqueeze(-1)
                    mask_for_zero = mask_for_zero.expand_as(normal_cpu)
                    normal_cpu = normal_cpu.clone()
                    normal_cpu[~mask_for_zero] = 0.0
                    mask_for_writer = mask_bool

                base_name = os.path.splitext(fname)[0]
                save_queue.put((base_name, normal_cpu, mask_for_writer))

            processed += 1
            pbar.update(1)

            if processed % 10 == 0 or processed == len(image_files):
                with stats_lock:
                    avg_save = (
                        save_stats["time"] / save_stats["count"]
                        if save_stats["count"] > 0 else 0.0
                    )
                tqdm.write(
                    f"图像 {fname}: 读取 {read_time:.3f}s, 推理 {inference_time:.3f}s, "
                    f"平均保存 {avg_save:.3f}s"
                )

    finally:
        pbar.close()

    filename_queue.join()

    save_queue.join()
    for _ in writer_threads:
        save_queue.put(None)
    for t in writer_threads:
        t.join()

    for t in reader_threads:
        t.join()

    total_save_time = save_stats["time"]

    print("\n" + "=" * 50)
    print("处理完成！详细耗时统计:")
    print(f"模型加载耗时: {model_load_time:.2f}秒")
    print(f"总图像读取耗时: {total_read_time:.2f}秒")
    print(f"总推理耗时: {total_inference_time:.2f}秒")
    print(f"总保存耗时: {total_save_time:.2f}秒")
    print(f"处理图像数量: {len(image_files)}张")
    if len(image_files) > 0:
        total_avg = (total_read_time + total_inference_time + total_save_time) / len(image_files)
        print(f"平均每张图像耗时: {total_avg:.2f}秒")
        print(f"平均推理耗时: {total_inference_time / len(image_files):.2f}秒")
        if save_stats["count"]:
            print(f"平均保存耗时: {total_save_time / save_stats['count']:.2f}秒")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多线程运行 MoGe-2 法线推理")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="包含图像子目录的数据根目录，例如：/path/to/data",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="images",
        help="data_dir 下实际存放图像的子目录名称，默认 images",
    )
    parser.add_argument(
        "--read_threads",
        type=int,
        default=4,
        help="并行读取图像的线程数，默认 4",
    )
    parser.add_argument(
        "--write_threads",
        type=int,
        default=2,
        help="并行写入法向图的线程数，默认 2",
    )
    parser.add_argument(
        "--queue_size",
        type=int,
        default=8,
        help="读写缓冲队列长度，默认 8",
    )
    args = parser.parse_args()

    run_inference(
        data_dir=args.data_dir,
        image_dir_name=args.image_dir,
        read_threads=args.read_threads,
        write_threads=args.write_threads,
        queue_size=args.queue_size,
    )
