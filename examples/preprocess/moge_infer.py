import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import time
import argparse
from moge.model.v2 import MoGeModel  # 使用 MoGe-2


def run_inference(data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载模型
    print("正在加载模型...")
    model_load_start = time.time()
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    model.eval()
    model_load_time = time.time() - model_load_start
    print(f"模型加载完成，耗时: {model_load_time:.2f}秒")

    # 2. 创建保存目录（仅保留 normal 输出目录）
    normal_output_dir = os.path.join(data_dir, "moge_normal")
    os.makedirs(normal_output_dir, exist_ok=True)

    # 3. 遍历 images 目录下所有图像文件
    image_dir = os.path.join(data_dir, "images")
    image_files = sorted(
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    print(f"找到 {len(image_files)} 张图像")

    total_inference_time = 0
    total_save_time = 0
    total_read_time = 0

    for fname in tqdm(image_files, desc="Running MoGe inference"):
        # 读取图像
        read_start = time.time()
        img_path = os.path.join(image_dir, fname)

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(
            img / 255.0, dtype=torch.float32, device=device
        ).permute(2, 0, 1)
        read_time = time.time() - read_start
        total_read_time += read_time

        # 推理
        inference_start = time.time()
        with torch.no_grad():
            output = model.infer(img_tensor)
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

        # 保存 normal 为 PNG
        save_start = time.time()
        if normal is not None:
            normal_np = normal.cpu().numpy()
            normal_np = normal_np * [-0.5, -0.5, -0.5] + 0.5  # 翻转逻辑

            normal_uint8 = (normal_np * 255).astype(np.uint8)

            base_name = os.path.splitext(fname)[0]
            output_path = os.path.join(normal_output_dir, f"{base_name}.png")
            cv2.imwrite(output_path, cv2.cvtColor(normal_uint8, cv2.COLOR_RGB2BGR))

        save_time = time.time() - save_start
        total_save_time += save_time

        if image_files.index(fname) % 10 == 0:
            tqdm.write(
                f"图像 {fname}: 读取 {read_time:.3f}s, 推理 {inference_time:.3f}s, 保存 {save_time:.3f}s"
            )

    # 总结统计
    print("\n" + "=" * 50)
    print("处理完成！详细耗时统计:")
    print(f"模型加载耗时: {model_load_time:.2f}秒")
    print(f"总图像读取耗时: {total_read_time:.2f}秒")
    print(f"总推理耗时: {total_inference_time:.2f}秒")
    print(f"总保存耗时: {total_save_time:.2f}秒")
    print(f"处理图像数量: {len(image_files)}张")
    if len(image_files) > 0:
        print(
            f"平均每张图像耗时: {(total_read_time + total_inference_time + total_save_time) / len(image_files):.2f}秒"
        )
        print(f"平均推理耗时: {total_inference_time / len(image_files):.2f}秒")
        print(f"平均保存耗时: {total_save_time / len(image_files):.2f}秒")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MoGe-2 normal inference on images"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="包含 images 文件夹的数据根目录，例如：/path/to/data",
    )
    args = parser.parse_args()

    run_inference(args.data_dir)
