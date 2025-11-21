import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import open3d as o3d
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from tqdm import tqdm
from scipy.sparse import csr_matrix, vstack
from scipy.spatial import cKDTree
from scipy.stats import mode

try:
    from cuml.cluster import DBSCAN as cuDBSCAN
    import cupy as cp

    HAS_CUML = True
except ImportError:
    cuDBSCAN = None
    cp = None
    HAS_CUML = False

# -------------------------------------------------------------------------
# Ensure the examples directory (parent of this file) is on sys.path so the
# dataset utilities can be imported just like other example scripts do.
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from datasets.colmap import Dataset, Parser  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402


class Node:
    def __init__(
        self,
        mask_list,
        visible_frame,
        contained_mask,
        point_ids: np.ndarray,
        node_info,
        son_node_info=None,
    ):
        self.mask_list = mask_list
        self.visible_frame = visible_frame
        self.contained_mask = contained_mask
        self.point_ids = point_ids
        self.node_info = node_info
        self.son_node_info = son_node_info

    @staticmethod
    def create_node_from_list(node_list, node_info):
        mask_list = []
        point_ids_list = []
        son_node_info = set()
        visible_stack = []
        contained_stack = []
        for node in node_list:
            mask_list += node.mask_list
            point_ids_list.append(node.point_ids)
            son_node_info.add(node.node_info)
            visible_stack.append(node.visible_frame)
            contained_stack.append(node.contained_mask)

        if len(point_ids_list) == 1:
            point_ids = point_ids_list[0].copy()
        else:
            point_ids = point_ids_list[0]
            for arr in point_ids_list[1:]:
                point_ids = np.union1d(point_ids, arr)

        if len(visible_stack) == 1:
            visible_frame = visible_stack[0].copy()
        else:
            visible_frame = vstack(visible_stack).sum(axis=0)
            visible_frame[visible_frame > 1] = 1
            visible_frame = csr_matrix(visible_frame)

        if len(contained_stack) == 1:
            contained_mask = contained_stack[0].copy()
        else:
            contained_mask = vstack(contained_stack).sum(axis=0)
            contained_mask[contained_mask > 1] = 1
            contained_mask = csr_matrix(contained_mask)

        return Node(mask_list, visible_frame, contained_mask, point_ids, node_info, son_node_info)


def compute_mask_visible_frame(global_gaussian_in_mask_matrix, gaussian_in_frame_matrix, threshold=0.0):
    """Use sparse matrix multiplication to determine mask visibility per frame."""
    print("[Mask Visibility] 使用稀疏矩阵计算 mask 可见帧 ...")
    A = global_gaussian_in_mask_matrix.astype(np.float32)
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A, dtype=np.float32)
    B = csr_matrix(gaussian_in_frame_matrix, dtype=np.float32)

    intersection_counts = A.T @ B  # [N_masks, N_frames]
    mask_point_counts = np.array(A.sum(axis=0)).ravel() + 1e-6

    intersection_counts = intersection_counts.tocoo()
    visibility = (intersection_counts.data / mask_point_counts[intersection_counts.row]) > threshold

    result = csr_matrix(
        (
            np.ones(np.count_nonzero(visibility), dtype=bool),
            (
                intersection_counts.row[visibility],
                intersection_counts.col[visibility],
            ),
        ),
        shape=(A.shape[1], B.shape[1]),
    )
    print("[Mask Visibility] 计算完成。")
    return result.toarray()


def judge_single_mask(
    gaussian_in_mask_matrix,
    mask_gaussian_pclds,
    frame_mask_id,
    mask_visible_frame,
    frame_mask_to_index,
    num_global_masks,
    mask_visible_threshold=0.7,
    contained_threshold=0.8,
    undersegment_filter_threshold=0.3,
):
    mask_gaussian_pcld = mask_gaussian_pclds[frame_mask_id]
    visible_frame = np.zeros(gaussian_in_mask_matrix.shape[1], dtype=bool)
    contained_mask = np.zeros(num_global_masks, dtype=bool)

    mask_gaussians_info = gaussian_in_mask_matrix[list(mask_gaussian_pcld), :]
    split_num = 0
    visible_num = 0

    for frame_id in np.where(mask_visible_frame)[0]:
        frame_info = mask_gaussians_info[:, frame_id]
        if frame_info.size == 0:
            continue
        mask_counts = np.bincount(frame_info)
        if mask_counts.size == 0:
            continue
        invalid_cnt = mask_counts[0] if mask_counts.shape[0] > 0 else 0
        total_cnt = frame_info.size
        if total_cnt == 0:
            continue
        if invalid_cnt / total_cnt > mask_visible_threshold:
            continue
        mask_counts[0] = 0
        if mask_counts.sum() == 0:
            continue
        best_mask_id = int(mask_counts.argmax())
        best_cnt = mask_counts[best_mask_id]
        valid_cnt = total_cnt - invalid_cnt
        if valid_cnt <= 0:
            continue
        visible_num += 1
        contained_ratio = best_cnt / valid_cnt
        if contained_ratio > contained_threshold:
            frame_mask_idx = frame_mask_to_index.get((int(frame_id), best_mask_id))
            if frame_mask_idx is None:
                continue
            contained_mask[frame_mask_idx] = True
            visible_frame[frame_id] = True
        else:
            split_num += 1

    if visible_num == 0 or split_num / max(visible_num, 1) > undersegment_filter_threshold:
        return False, contained_mask, visible_frame
    else:
        return True, contained_mask, visible_frame


def get_observer_num_thresholds(visible_frames_sparse: csr_matrix):
    observer_num_matrix = visible_frames_sparse @ visible_frames_sparse.T
    observer_num_list = observer_num_matrix.data
    observer_num_list = observer_num_list[observer_num_list > 0]
    if observer_num_list.size == 0:
        return [1]

    percentiles = np.arange(95, -5, -5)
    percentile_values = np.percentile(observer_num_list, percentiles)
    observer_num_thresholds: List[float] = []
    for percentile, observer_num in zip(percentiles, percentile_values):
        if observer_num <= 1:
            if percentile < 50:
                break
            observer_num = 1
        observer_num_thresholds.append(observer_num)
    return observer_num_thresholds


def update_graph(nodes, observer_num_threshold, connect_threshold):
    node_visible_frames = vstack([node.visible_frame for node in nodes])
    node_contained_masks = vstack([node.contained_mask for node in nodes])
    observer_nums = node_visible_frames @ node_visible_frames.T
    supporter_nums = node_contained_masks @ node_contained_masks.T
    observer_nums_dense = observer_nums.toarray()
    supporter_nums_dense = supporter_nums.toarray()
    view_concensus_rate = supporter_nums_dense / (observer_nums_dense + 1e-7)
    num_nodes = len(nodes)
    disconnect = np.eye(num_nodes, dtype=bool)
    disconnect = disconnect | (observer_nums_dense < observer_num_threshold)
    A = view_concensus_rate >= connect_threshold
    A = A & ~disconnect
    G = nx.from_numpy_array(A)
    return G


def cluster_into_new_nodes(iteration, old_nodes, graph):
    new_nodes = []
    for component in nx.connected_components(graph):
        node_info = (iteration, len(new_nodes))
        new_nodes.append(Node.create_node_from_list([old_nodes[node] for node in component], node_info))
    return new_nodes


def iterative_clustering(nodes, observer_num_thresholds, connect_threshold):
    iterator = tqdm(enumerate(observer_num_thresholds), total=len(observer_num_thresholds), desc="迭代聚类")
    for iterate_id, observer_num_threshold in iterator:
        graph = update_graph(nodes, observer_num_threshold, connect_threshold)
        nodes = cluster_into_new_nodes(iterate_id + 1, nodes, graph)
    return nodes


def iterative_cluster_masks(tracker: Dict) -> Dict:
    gaussian_in_frame_matrix = tracker["gaussian_in_frame_matrix"]
    mask_gaussian_pclds = tracker["mask_gaussian_pclds"]
    global_frame_mask_list = tracker["global_frame_mask_list"]

    frame_mask_to_index = {(int(frame_id), int(mask_id)): idx for idx, (frame_id, mask_id) in enumerate(global_frame_mask_list)}

    num_points = gaussian_in_frame_matrix.shape[0]
    gaussian_in_frame_maskid_matrix = np.zeros((num_points, gaussian_in_frame_matrix.shape[1]), dtype=np.uint16)

    mask_rows: List[np.ndarray] = []
    mask_cols: List[np.ndarray] = []

    for mask_idx, (frame_id, mask_id) in enumerate(global_frame_mask_list):
        ids = mask_gaussian_pclds[f"{frame_id}_{mask_id}"]
        gaussian_in_frame_maskid_matrix[ids, frame_id] = mask_id
        if len(ids) == 0:
            continue
        ids_arr = np.asarray(ids, dtype=np.int64)
        mask_rows.append(ids_arr)
        mask_cols.append(np.full(ids_arr.shape[0], mask_idx, dtype=np.int64))

    if len(mask_rows) > 0:
        mask_row_idx = np.concatenate(mask_rows)
        mask_col_idx = np.concatenate(mask_cols)
    else:
        mask_row_idx = np.empty(0, dtype=np.int64)
        mask_col_idx = np.empty(0, dtype=np.int64)

    data = np.ones(mask_row_idx.shape[0], dtype=bool)
    global_gaussian_in_mask_matrix = csr_matrix(
        (data, (mask_row_idx, mask_col_idx)),
        shape=(num_points, len(global_frame_mask_list)),
    )

    mask_visible_frames = compute_mask_visible_frame(
        global_gaussian_in_mask_matrix, gaussian_in_frame_matrix
    )

    contained_masks = []
    visible_frames = []
    undersegment_mask_ids = []

    for mask_cnts, (frame_id, mask_id) in enumerate(tqdm(global_frame_mask_list, desc="过滤欠分割 Mask")):
        valid, contained_mask, visible_frame = judge_single_mask(
            gaussian_in_frame_maskid_matrix,
            mask_gaussian_pclds,
            f"{frame_id}_{mask_id}",
            mask_visible_frames[mask_cnts],
            frame_mask_to_index,
            len(global_frame_mask_list),
        )
        contained_masks.append(contained_mask)
        visible_frames.append(visible_frame)
        if not valid:
            undersegment_mask_ids.append(mask_cnts)

    contained_masks = np.stack(contained_masks, axis=0)
    visible_frames = np.stack(visible_frames, axis=0)

    for global_mask_id in undersegment_mask_ids:
        frame_id, _ = global_frame_mask_list[global_mask_id]
        mask_projected_idx = np.where(contained_masks[:, global_mask_id])[0]
        contained_masks[:, global_mask_id] = False
        visible_frames[mask_projected_idx, frame_id] = False

    contained_masks_sparse = csr_matrix(contained_masks, dtype=np.int32)
    visible_frames_sparse = csr_matrix(visible_frames, dtype=np.int32)
    contained_masks_sparse.sort_indices()
    visible_frames_sparse.sort_indices()

    print("[Perf] 开始计算共现矩阵与观察者阈值 ...")
    threshold_t0 = time.perf_counter()
    observer_num_thresholds = get_observer_num_thresholds(visible_frames_sparse)
    print(f"[Perf] 共现矩阵/阈值耗时 {time.perf_counter() - threshold_t0:.2f}s")

    print("[Perf] 开始构建节点列表 ...")
    node_t0 = time.perf_counter()
    nodes = []
    for global_mask_id, (frame_id, mask_id) in enumerate(global_frame_mask_list):
        if global_mask_id in undersegment_mask_ids:
            continue
        mask_list = [(frame_id, mask_id)]
        frame = visible_frames_sparse.getrow(global_mask_id)
        frame_mask = contained_masks_sparse.getrow(global_mask_id)
        point_ids = np.unique(np.asarray(mask_gaussian_pclds[f"{frame_id}_{mask_id}"], dtype=np.int64))
        node_info = (0, len(nodes))
        node = Node(mask_list, frame, frame_mask, point_ids, node_info, None)
        nodes.append(node)
    print(f"[Perf] 构建节点耗时 {time.perf_counter() - node_t0:.2f}s，节点数 {len(nodes)}")

    nodes = iterative_clustering(nodes, observer_num_thresholds, connect_threshold=0.9)

    tracker.update(
        {
            "nodes": nodes,
            "observer_num_thresholds": observer_num_thresholds,
            "undersegment_mask_ids": undersegment_mask_ids,
        }
    )
    return tracker


def _gpu_dbscan(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    points_cp = cp.asarray(points.astype(np.float32))
    model = cuDBSCAN(eps=eps, min_samples=min_points)
    labels = model.fit_predict(points_cp)
    if hasattr(labels, "values"):
        labels = labels.values
    if hasattr(labels, "__cuda_array_interface__"):
        labels = cp.asnumpy(labels)
    labels = np.asarray(labels, dtype=np.int32)
    return labels + 1  # shift noise (-1) to 0


def dbscan_process(
    pcld: o3d.geometry.PointCloud,
    point_ids: List[int],
    eps: float = 0.1,
    min_points: int = 4,
    use_gpu: bool = False,
):
    points_np = np.asarray(pcld.points)
    if use_gpu and not HAS_CUML:
        raise RuntimeError("cuML 未安装，无法使用 GPU DBSCAN。")
    if use_gpu and points_np.shape[0] >= min_points and points_np.size > 0:
        labels = _gpu_dbscan(points_np, eps, min_points)
    else:
        labels = np.array(pcld.cluster_dbscan(eps=eps, min_points=min_points)) + 1
    count = np.bincount(labels)
    pcld_list, point_ids_list = [], []
    pcld_ids_list = np.array(point_ids)
    for i in range(len(count)):
        remain_index = np.where(labels == i)[0]
        if len(remain_index) == 0:
            continue
        new_pcld = pcld.select_by_index(remain_index)
        pts = pcld_ids_list[remain_index]
        pcld_list.append(new_pcld)
        point_ids_list.append(pts)
    return pcld_list, point_ids_list


def merge_overlapping_objects(total_point_ids_list, total_bbox_list, total_mask_list, overlapping_ratio=0.8):
    total_object_num = len(total_point_ids_list)
    invalid_object = np.zeros(total_object_num, dtype=bool)
    for i in range(total_object_num):
        if invalid_object[i]:
            continue
        point_ids_i = set(total_point_ids_list[i])
        bbox_i = total_bbox_list[i]
        for j in range(i + 1, total_object_num):
            if invalid_object[j]:
                continue
            point_ids_j = set(total_point_ids_list[j])
            bbox_j = total_bbox_list[j]
            # bbox overlap check
            overlap_bbox = True
            for k in range(3):
                if bbox_i[0][k] > bbox_j[1][k] or bbox_j[0][k] > bbox_i[1][k]:
                    overlap_bbox = False
                    break
            if not overlap_bbox:
                continue
            intersect = len(point_ids_i.intersection(point_ids_j))
            if intersect / len(point_ids_i) > overlapping_ratio:
                invalid_object[i] = True
            elif intersect / len(point_ids_j) > overlapping_ratio:
                invalid_object[j] = True

    valid_point_ids_list = []
    valid_mask_list = []
    for i in range(total_object_num):
        if not invalid_object[i]:
            valid_point_ids_list.append(total_point_ids_list[i])
            valid_mask_list.append(total_mask_list[i])
    return valid_point_ids_list, valid_mask_list


def filter_point(
    point_frame_matrix: np.ndarray,
    node: Node,
    pcld_list: List[o3d.geometry.PointCloud],
    point_ids_list: List[np.ndarray],
    mask_point_clouds: Dict,
    point_filter_threshold: float,
):
    node_global_frame_id_list = np.where(node.visible_frame.toarray().ravel() > 0)[0]
    mask_list = node.mask_list

    point_appear_in_video_nums = []
    point_appear_in_node_matrixs = []
    for point_ids in point_ids_list:
        point_appear_in_video_matrix = point_frame_matrix[point_ids, :]
        point_appear_in_video_matrix = point_appear_in_video_matrix[:, node_global_frame_id_list]
        point_appear_in_video_nums.append(np.sum(point_appear_in_video_matrix, axis=1))
        point_appear_in_node_matrix = np.zeros_like(point_appear_in_video_matrix, dtype=bool)
        point_appear_in_node_matrixs.append(point_appear_in_node_matrix)

    object_mask_list = [[] for _ in range(len(point_ids_list))]
    for frame_id, mask_id in mask_list:
        if frame_id not in node_global_frame_id_list:
            continue
        frame_id_in_list = np.where(node_global_frame_id_list == frame_id)[0][0]
        mask_point_ids = list(mask_point_clouds[f"{frame_id}_{mask_id}"])
        for idx_obj, point_ids in enumerate(point_ids_list):
            point_ids_within_object = np.where(np.isin(point_ids, mask_point_ids))[0]
            point_appear_in_node_matrixs[idx_obj][point_ids_within_object, frame_id_in_list] = True
            if len(point_ids_within_object) > 0:
                object_mask_list[idx_obj].append((frame_id, mask_id, len(point_ids_within_object) / len(point_ids)))

    filtered_point_ids, filtered_bbox_list, filtered_mask_list = [], [], []
    for i, (point_appear_in_video_num, point_appear_in_node_matrix) in enumerate(
        zip(point_appear_in_video_nums, point_appear_in_node_matrixs)
    ):
        detection_ratio = np.sum(point_appear_in_node_matrix, axis=1) / (point_appear_in_video_num + 1e-6)
        valid_point_ids = np.where(detection_ratio > point_filter_threshold)[0]
        if len(valid_point_ids) == 0 or len(object_mask_list[i]) < 2:
            continue
        filtered_point_ids.append(point_ids_list[i][valid_point_ids])
        pcld = pcld_list[i]
        filtered_bbox_list.append([np.amin(np.asarray(pcld.points), axis=0), np.amax(np.asarray(pcld.points), axis=0)])
        filtered_mask_list.append(object_mask_list[i])
    return filtered_point_ids, filtered_bbox_list, filtered_mask_list


def post_process_clusters(
    tracker: Dict,
    point_positions: torch.Tensor,
    point_filter_threshold: float = 0.5,
    dbscan_eps: float = 0.1,
    dbscan_min_points: int = 4,
    overlap_ratio: float = 0.8,
    use_gpu_dbscan: bool = False,
) -> Dict:
    nodes = tracker["nodes"]
    mask_gaussian_pclds = tracker["mask_gaussian_pclds"]
    gaussian_in_frame_matrix = tracker["gaussian_in_frame_matrix"]

    total_point_ids_list, total_bbox_list, total_mask_list = [], [], []
    scene_points = point_positions.cpu().numpy()

    iterator = tqdm(nodes, total=len(nodes), desc="DBScan+点过滤")
    for node in iterator:
        if len(node.mask_list) < 2:
            continue
        node_point_ids = node.point_ids.tolist()
        pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_points[node_point_ids]))
        if True:
            pcld_list, point_ids_list = dbscan_process(
                pcld,
                node_point_ids,
                eps=dbscan_eps,
                min_points=dbscan_min_points,
                use_gpu=use_gpu_dbscan,
            )
        else:
            pcld_list, point_ids_list = [pcld], [np.array(node_point_ids)]
        point_ids_list, bbox_list, mask_list = filter_point(
            gaussian_in_frame_matrix,
            node,
            pcld_list,
            point_ids_list,
            mask_gaussian_pclds,
            point_filter_threshold,
        )
        total_point_ids_list.extend(point_ids_list)
        total_bbox_list.extend(bbox_list)
        total_mask_list.extend(mask_list)

    total_point_ids_list, total_mask_list = merge_overlapping_objects(
        total_point_ids_list, total_bbox_list, total_mask_list, overlapping_ratio=overlap_ratio
    )

    tracker.update(
        {
            "total_point_ids_list": total_point_ids_list,
            "total_mask_list": total_mask_list,
        }
    )
    return tracker


def remedy_undersegment(
    tracker: Dict,
    threshold: float = 0.8,
) -> Dict:
    """尝试将欠分割的 mask 分配到最匹配的实例中。"""
    undersegment_frame_masks = [tracker["global_frame_mask_list"][fid] for fid in tracker["undersegment_mask_ids"]]
    error_undersegment_frame_masks = {}
    remedy_undersegment_frame_masks = []

    total_instances = len(tracker["total_point_ids_list"])
    if total_instances == 0:
        tracker["undersegment_mask_ids"] = remedy_undersegment_frame_masks
        return tracker

    num_points = tracker["gaussian_in_frame_matrix"].shape[0]
    gaussian_to_instance = np.full(num_points, -1, dtype=np.int32)
    for inst_idx, point_ids in enumerate(tracker["total_point_ids_list"]):
        if len(point_ids) == 0:
            continue
        gaussian_to_instance[np.asarray(point_ids, dtype=np.int64)] = inst_idx

    frame_mask_to_index = {
        tuple(frame_mask): idx for idx, frame_mask in enumerate(tracker["global_frame_mask_list"])
    }

    for frame_mask in tqdm(undersegment_frame_masks, desc="修补欠分割 mask"):
        frame_id, mask_id = frame_mask
        frame_mask_gaussian = tracker["mask_gaussian_pclds"][f"{frame_id}_{mask_id}"]
        if len(frame_mask_gaussian) == 0:
            remedy_undersegment_frame_masks.append(frame_mask_to_index[tuple(frame_mask)])
            continue
        instance_ids = gaussian_to_instance[np.asarray(frame_mask_gaussian, dtype=np.int64)]
        instance_ids = instance_ids[instance_ids >= 0]
        if instance_ids.size == 0:
            remedy_undersegment_frame_masks.append(frame_mask_to_index[tuple(frame_mask)])
            continue
        counts = np.bincount(instance_ids, minlength=total_instances)
        best_match_instance_idx = int(counts.argmax())
        best_match_intersect = int(counts[best_match_instance_idx])
        if best_match_intersect / len(frame_mask_gaussian) > threshold:
            error_undersegment_frame_masks[frame_mask] = best_match_instance_idx
        else:
            remedy_undersegment_frame_masks.append(frame_mask_to_index[tuple(frame_mask)])

    tracker["undersegment_mask_ids"] = remedy_undersegment_frame_masks
    total_mask_list = tracker["total_mask_list"]
    for frame_mask in error_undersegment_frame_masks:
        total_mask_list[error_undersegment_frame_masks[frame_mask]].append(frame_mask)
    tracker["total_mask_list"] = total_mask_list
    return tracker


def export_color_cluster(
    tracker: Dict,
    point_positions: torch.Tensor,
    save_dir: Path,
    filename: str = "color_cluster.ply",
    assign_unlabeled_knn: bool = True,
    knn_k: int = 1,
    knn_filename: str = "color_cluster_knn.ply",
):
    """导出彩色实例点云，可选地通过 KNN 为未分类点赋予实例颜色。"""
    os.makedirs(save_dir, exist_ok=True)
    total_point_ids_list = tracker.get("total_point_ids_list", [])
    if len(total_point_ids_list) == 0:
        print("[Export] total_point_ids_list 为空，跳过导出。")
        return

    xyz = point_positions.cpu().numpy()
    num_points = xyz.shape[0]
    colors = np.zeros((num_points, 3), dtype=np.float32)
    inst_labels = np.full(num_points, -1, dtype=np.int32)
    rng = np.random.default_rng(0)
    inst_colors = rng.random((len(total_point_ids_list), 3)) * 0.7 + 0.3

    for idx, point_ids in enumerate(total_point_ids_list):
        pts = np.array(point_ids, dtype=int)
        colors[pts] = inst_colors[idx]
        inst_labels[pts] = idx

    pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcld.colors = o3d.utility.Vector3dVector(colors)
    save_path = save_dir / filename
    o3d.io.write_point_cloud(str(save_path), pcld)
    print(f"[Export] 已保存彩色实例点云到 {save_path}")

    if not assign_unlabeled_knn:
        return

    unlabeled_mask = inst_labels < 0
    assigned_mask = ~unlabeled_mask
    if not np.any(unlabeled_mask):
        print("[Export] 所有点已有实例颜色，跳过 KNN 重新着色。")
        return
    if not np.any(assigned_mask):
        print("[Export] 未找到可供 KNN 参考的实例点，跳过。")
        return

    k = max(1, min(knn_k, int(np.sum(assigned_mask))))
    assigned_xyz = xyz[assigned_mask]
    assigned_labels = inst_labels[assigned_mask]
    tree = cKDTree(assigned_xyz)
    _, nn_idx = tree.query(xyz[unlabeled_mask], k=k)
    if k == 1:
        inferred_labels = assigned_labels[nn_idx]
    else:
        neighbor_labels = assigned_labels[nn_idx]
        if neighbor_labels.ndim == 1:
            neighbor_labels = neighbor_labels[None, :]
        mode_result = mode(neighbor_labels, axis=1, keepdims=False)
        inferred_labels = (
            np.asarray(mode_result.mode)
            if hasattr(mode_result, "mode")
            else np.asarray(mode_result)
        )
        if inferred_labels.ndim > 1:
            inferred_labels = inferred_labels.squeeze(-1)
        inferred_labels = inferred_labels.astype(np.int32, copy=False)
    inst_labels[unlabeled_mask] = inferred_labels

    colors_knn = colors.copy()
    colors_knn[unlabeled_mask] = inst_colors[inst_labels[unlabeled_mask]]
    pcld_knn = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcld_knn.colors = o3d.utility.Vector3dVector(colors_knn)
    knn_path = save_dir / knn_filename
    o3d.io.write_point_cloud(str(knn_path), pcld_knn)
    print(f"[Export] 已保存 KNN 填充彩色点云到 {knn_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare camera, mask, and Gaussian parameters for GausCluster."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root COLMAP-style data directory (e.g. /path/to/scene).",
    )
    parser.add_argument(
        "--data_factor",
        type=int,
        default=1,
        help="Image downscale factor to match Parser behaviour (default: 1).",
    )
    parser.add_argument(
        "--normalize_world_space",
        action="store_true",
        help="Match Parser option to normalize world coordinates.",
    )
    parser.add_argument(
        "--test_every",
        type=int,
        default=8,
        help="Keep consistent with Parser/Dataset split (default: 8).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        nargs="+",
        required=True,
        help="One or more checkpoint paths containing Gaussian parameters.",
    )
    parser.add_argument(
        "--mask_subdir",
        type=str,
        default=None,
        help="Optional override for which subfolder inside data_dir/sam to use.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save the prepared payload (.pt). Defaults to <data_dir>/gauscluster_input.pt",
    )
    parser.add_argument(
        "--use_gpu_dbscan",
        action="store_true",
        help="Use cuML (GPU) DBSCAN for post-processing if available.",
    )
    return parser.parse_args()


def load_gaussians_from_ckpt(
    ckpt_paths: List[str], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Load and merge Gaussian parameters from checkpoint files."""
    means, quats, scales, opacities, sh0, shn = [], [], [], [], [], []

    print(f"[Gaussians] Loading {len(ckpt_paths)} checkpoint(s) on {device} ...")
    for ckpt_path in ckpt_paths:
        ckpt_path = os.path.expanduser(ckpt_path)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device)
        if "splats" not in payload:
            raise KeyError(f"Checkpoint missing 'splats' key: {ckpt_path}")
        splats = payload["splats"]
        means.append(splats["means"])
        quats.append(F.normalize(splats["quats"], dim=-1))
        scales.append(torch.exp(splats["scales"]))
        opacities.append(torch.sigmoid(splats["opacities"]))
        sh0.append(splats["sh0"])
        shn.append(splats["shN"])
        print(f"  • Loaded {ckpt_path}")

    def _cat(items: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(items, dim=0)

    means = _cat(means)
    quats = _cat(quats)
    scales = _cat(scales)
    opacities = _cat(opacities)
    colors = torch.cat([_cat(sh0), _cat(shn)], dim=-2)
    sh_degree = int(np.sqrt(colors.shape[-2]) - 1)

    print(
        f"[Gaussians] Total={means.shape[0]}, feature_dim={(sh_degree + 1) ** 2}, sh_degree={sh_degree}"
    )
    return means, quats, scales, opacities, colors, sh_degree


def discover_mask_dir(data_dir: str, override: Optional[str] = None) -> str:
    """Find the directory that stores SAM masks."""
    sam_root = os.path.join(data_dir, "sam")
    if not os.path.isdir(sam_root):
        raise FileNotFoundError(f"sam directory not found under {data_dir}")

    candidates = []
    if override is not None:
        candidates.append(os.path.join(sam_root, override))
    # Preferred fallbacks.
    candidates.extend(
        [
            os.path.join(sam_root, "mask_sorted"),
            os.path.join(sam_root, "mask"),
            os.path.join(sam_root, "mask_filtered"),
        ]
    )
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            print(f"[Masks] Using SAM masks from: {candidate}")
            return candidate

    raise FileNotFoundError(f"No mask directory found inside {sam_root}")


def read_mask(mask_path: str) -> np.ndarray:
    """Load a mask image/npy as integer labels."""
    if mask_path.endswith(".npy"):
        mask = np.load(mask_path)
    else:
        mask = imageio.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int32)


def collect_view_data(
    parser: Parser, dataset: Dataset, mask_dir: str
) -> List[Dict[str, torch.Tensor]]:
    """Assemble per-view camera parameters and SAM masks."""
    view_data = []
    image_hw_cache: Dict[str, Tuple[int, int]] = {}
    indices = dataset.indices

    pbar = tqdm(range(len(indices)), desc="[Cameras] 准备视角", total=len(indices))
    for local_idx in pbar:
        parser_idx = indices[local_idx]
        image_path = parser.image_paths[parser_idx]
        image_name = Path(image_path).stem

        camera_id = parser.camera_ids[parser_idx]
        width, height = parser.imsize_dict[camera_id]
        K = parser.Ks_dict[camera_id]
        camtoworld = parser.camtoworlds[parser_idx]

        mask_path = None
        for ext in (".png", ".jpg", ".jpeg", ".npy"):
            candidate = os.path.join(mask_dir, f"{image_name}{ext}")
            if os.path.exists(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            raise FileNotFoundError(f"No mask found for {image_name} in {mask_dir}")

        mask_np = read_mask(mask_path)
        mask_h, mask_w = mask_np.shape

        if mask_h != height or mask_w != width:
            if image_path not in image_hw_cache:
                rgb = imageio.imread(image_path)
                image_hw_cache[image_path] = rgb.shape[:2]
            actual_h, actual_w = image_hw_cache[image_path]
            if mask_h == actual_h and mask_w == actual_w:
                height, width = actual_h, actual_w
            else:
                raise ValueError(
                    f"Mask 尺寸 {mask_h}x{mask_w} 与相机解析到的尺寸 {height}x{width} "
                    f"以及原始图像尺寸 {actual_h}x{actual_w} 均不一致：{mask_path}"
                )
        view_entry = {
            "index": int(parser_idx),
            "image_name": image_name,
            "image_path": image_path,
            "width": int(width),
            "height": int(height),
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworld).float(),
            "mask_path": mask_path,
        }
        view_data.append(view_entry)
        pbar.set_postfix_str(f"{local_idx + 1}/{len(indices)} current: {image_name}")

    return view_data


def build_mask_gaussian_tracker(
    gaussians: Tuple[torch.Tensor, ...],
    view_data: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Dict:
    """构建 mask -> Gaussian 跟踪：为每帧的每个 mask 找到对应的高斯点 id 列表。"""
    means, quats, scales, opacities, colors, sh_degree = gaussians
    num_points = means.shape[0]
    num_frames = len(view_data)

    gaussian_in_frame_matrix = np.zeros((num_points, num_frames), dtype=bool)
    mask_gaussian_pclds: Dict[str, np.ndarray] = {}
    frame_gaussian_ids: List[set] = []
    global_frame_mask_list: List[Tuple[int, int]] = []

    means_d = means.to(device)
    quats_d = quats.to(device)
    scales_d = scales.to(device)
    opacities_d = opacities.to(device)
    colors_d = colors.to(device)

    pbar = tqdm(view_data, desc="构建 mask→Gaussian 跟踪", total=num_frames)
    for frame_idx, view in enumerate(pbar):
        width, height = view["width"], view["height"]
        viewmats = torch.linalg.inv(view["camtoworld"].to(device)).unsqueeze(0)
        Ks = view["K"].to(device).unsqueeze(0)

        with torch.no_grad():
            _, _, meta = rasterization(
                means=means_d,
                quats=quats_d,
                scales=scales_d,
                opacities=opacities_d,
                colors=colors_d,
                viewmats=viewmats,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree,
                render_mode="RGB",
                packed=False,
                radius_clip=3.0,
                distributed=False,
                with_ut=False,
                with_eval3d=False,
                track_pixel_gaussians=True,
                max_gaussians_per_pixel=100,
                pixel_gaussian_threshold=0.25,
            )

        pixel_gaussians = meta.get("pixel_gaussians", None)
        if pixel_gaussians is None or pixel_gaussians.numel() == 0:
            raise RuntimeError(f"未在视角 {view['image_name']} 获取到像素-高斯对应关系。")

        gaus_ids = pixel_gaussians[:, 0].long()
        pixel_ids = pixel_gaussians[:, 1].long()

        segmap_np = read_mask(view["mask_path"])
        segmap = torch.from_numpy(segmap_np).long().to(device)
        segmap_flat = segmap.view(-1)
        labels = segmap_flat[pixel_ids]
        valid_mask = labels > 0
        if valid_mask.sum() == 0:
            frame_gaussian_ids.append(set())
            continue

        gaus_ids = gaus_ids[valid_mask]
        labels = labels[valid_mask]

        unique_labels = labels.unique()
        frame_gauss_set: set = set()
        for lbl in unique_labels:
            lbl_int = int(lbl.item())
            lbl_mask = labels == lbl
            lbl_gauss = torch.unique(gaus_ids[lbl_mask]).cpu().numpy()
            mask_gaussian_pclds[f"{frame_idx}_{lbl_int}"] = lbl_gauss
            global_frame_mask_list.append((frame_idx, lbl_int))
            frame_gauss_set.update(lbl_gauss.tolist())

        frame_gaussian_ids.append(frame_gauss_set)
        if len(frame_gauss_set) > 0:
            gaussian_in_frame_matrix[list(frame_gauss_set), frame_idx] = True

    del quats_d, scales_d, opacities_d, colors_d
    if means_d.is_cuda:
        del means_d
        torch.cuda.empty_cache()

    return {
        "gaussian_in_frame_matrix": gaussian_in_frame_matrix,
        "mask_gaussian_pclds": mask_gaussian_pclds,
        "frame_gaussian_ids": frame_gaussian_ids,
        "global_frame_mask_list": global_frame_mask_list,
    }


def main():
    args = parse_args()

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else Path(args.data_dir) / "gauscluster_input.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gaussians = load_gaussians_from_ckpt(args.ckpt, device=device)

    print(f"[Dataset] Loading COLMAP data from {args.data_dir}")
    parser = Parser(
        data_dir=args.data_dir,
        factor=args.data_factor,
        normalize=args.normalize_world_space,
        test_every=args.test_every,
    )
    dataset = Dataset(parser, split="train")
    print(f"[Dataset] Train images: {len(dataset)}")

    mask_dir = discover_mask_dir(args.data_dir, args.mask_subdir)
    view_data = collect_view_data(parser, dataset, mask_dir)
    print(f"[Summary] Prepared {len(view_data)} camera views with SAM masks.")

    means, quats, scales, opacities, colors, sh_degree = gaussians
    print("\n================ 数据校验 ================")
    print(f"数据目录: {args.data_dir}")
    print(f"Mask 目录: {mask_dir}")
    print(f"相机数量: {len(view_data)}")
    print(f"高斯数量: {means.shape[0]}")
    print(f"SH 阶数: {sh_degree}")
    print("----------------------------------------")
    print("示例视角：")
    for sample in view_data[:3]:
        print(
            f"  • {sample['image_name']} | 分辨率 {sample['width']}x{sample['height']} | "
            f"mask: {sample['mask_path']}"
        )
    print("========================================\n")

    tracker = build_mask_gaussian_tracker(gaussians, view_data, device=device)
    total_masks = len(tracker["global_frame_mask_list"])
    total_points_hit = tracker["gaussian_in_frame_matrix"].sum()
    print("============== 跟踪统计（阶段一） ==============")
    print(f"有效 mask 数量: {total_masks}")
    print(f"有像素贡献的 Gaussian 总计: {total_points_hit}")
    for i, (frame_id, mask_id) in enumerate(tracker["global_frame_mask_list"][:5]):
        gs_ids = tracker["mask_gaussian_pclds"][f"{frame_id}_{mask_id}"]
        print(
            f"  • frame {frame_id:03d}, mask {mask_id}: 高斯数 {len(gs_ids)}"
        )
    print("=============================================")

    # 阶段二：迭代聚类
    print("\n开始迭代聚类（阶段二）...")
    clustering_result = iterative_cluster_masks(tracker)
    print("聚类完成。")
    print(f"实例数（聚类后节点数）: {len(clustering_result['nodes'])}")

    # 阶段三：DBSCAN + 点过滤
    print("\n开始后处理（阶段三：DBSCAN+点过滤）...")
    if args.use_gpu_dbscan and not HAS_CUML:
        raise RuntimeError("检测到 --use_gpu_dbscan 但未安装 cuML，请安装后重试或取消该选项。")

    clustering_result = post_process_clusters(
        clustering_result,
        point_positions=means,
        point_filter_threshold=0.5,
        dbscan_eps=0.1,
        dbscan_min_points=4,
        overlap_ratio=0.8,
        use_gpu_dbscan=args.use_gpu_dbscan,
    )
    print("后处理完成。")
    print(f"实例数（后处理）: {len(clustering_result['total_point_ids_list'])}")

    # 阶段四：修补欠分割 mask
    print("\n开始修补欠分割 Mask（阶段四）...")
    clustering_result = remedy_undersegment(clustering_result, threshold=0.8)
    print("修补完成。")
    print(f"最终实例数: {len(clustering_result['total_point_ids_list'])}")

    # 导出彩色实例点云
    save_dir = Path(args.data_dir) / "cluster_result"
    export_color_cluster(clustering_result, point_positions=means, save_dir=save_dir)


if __name__ == "__main__":
    main()
