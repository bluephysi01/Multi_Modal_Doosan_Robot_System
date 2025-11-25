"""
Vision Module for Object Recognition
object_recog.py의 로직과 구성을 그대로 사용
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import math
import hdbscan


##############################
# util
##############################
def angle_diff(a, b):
    d = a - b
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return d


##############################
# extract
##############################
def extract_instances_from_pcd(
        vtx,
        rgb_img,
        valid_mask,
        fx, fy, cx, cy,
        voxel_size=0.004,
        depth_axis='z',
        ground_ratio=0.8,
        dbscan_eps=0.03,
        dbscan_min_samples=30,
        global_min_pts=110
):
    """포인트 클라우드에서 물체 인스턴스 추출 (object_recog.py와 동일)"""

    if isinstance(vtx, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vtx)
    else:
        pcd = vtx

    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_denoised, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    points = np.asarray(pcd_denoised.points)

    if len(points) == 0:
        return np.empty((0, 3)), np.array([]), [], np.empty((0, 3))

    axis_idx = {'x': 0, 'y': 1, 'z': 2}[depth_axis]

    #########################################
    # Ground segmentation
    #########################################
    try:
        plane_model, inliers = pcd_denoised.segment_plane(
            distance_threshold=0.008,
            ransac_n=3,
            num_iterations=1000
        )
        plane_pts = points[inliers]
        d_cut = np.max(plane_pts[:, axis_idx]) * ground_ratio

        ground_mask = np.zeros(len(points), dtype=bool)
        ground_mask[inliers] = (plane_pts[:, axis_idx] >= d_cut)
    except:
        ground_mask = np.zeros(len(points), dtype=bool)

    #########################################
    # Global sparse removal
    #########################################
    labels_global = np.array(pcd_denoised.cluster_dbscan(eps=0.03, min_points=10))
    if np.any(labels_global >= 0):
        counts = np.bincount(labels_global[labels_global >= 0])
        keep = np.where(counts > global_min_pts)[0]
        sparse_mask = ~np.isin(labels_global, keep)
    else:
        sparse_mask = np.zeros(len(points), dtype=bool)

    #########################################
    # Local density filter
    #########################################
    tree = cKDTree(points)
    neighbor_counts = np.array([len(tree.query_ball_point(p, r=0.015))
                                for p in points])
    local_sparse_mask = neighbor_counts < 20

    depth_abnormal = (points[:, 2] > 0.378) | (points[:, 2] < 0.15)

    #########################################
    # FILTERED POINTS
    #########################################
    remove_mask = ground_mask | sparse_mask | local_sparse_mask | depth_abnormal
    filtered_points = points[~remove_mask]
    ground_points = points[ground_mask]

    if len(filtered_points) == 0:
        return np.empty((0,3)), np.array([]), [], ground_points

    #########################################
    # HDBSCAN 1차 수행
    #########################################
    scaled = filtered_points.copy()
    scaled[:, 2] *= 2.0

    clustering = hdbscan.HDBSCAN(
        min_cluster_size=dbscan_min_samples,
        min_samples=dbscan_min_samples,
        cluster_selection_epsilon=dbscan_eps * 1.2
    ).fit(scaled)

    labels = clustering.labels_
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    ###########################################################
    # RGB Gradient 분리 (클러스터 1개일 때만)
    ###########################################################
    if num_clusters == 1:

        # Step 1: Sobel Edge
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
        sobel = cv2.GaussianBlur(np.abs(sobel), (5,5), 0)
        edge = sobel > 30

        # Step 2: filtered_points → pixel 변환
        uv = []
        for p in filtered_points:
            X, Y, Z = p
            u = int(X * fx / Z + cx)
            v = int(Y * fy / Z + cy)
            uv.append((u, v))
        uv = np.array(uv)

        # Step 3: edge 픽셀에 해당하는 point 제거
        H, W = gray.shape
        keep = []
        for i, (u, v) in enumerate(uv):
            if u < 0 or u >= W or v < 0 or v >= H:
                keep.append(True)
            else:
                if edge[v, u]:
                    keep.append(False)
                else:
                    keep.append(True)

        filtered_points2 = filtered_points[np.array(keep)]

        if len(filtered_points2) > 20:
            filtered_points = filtered_points2

            # Step 4: HDBSCAN 재실행
            scaled = filtered_points.copy()
            scaled[:, 2] *= 2.0

            clustering = hdbscan.HDBSCAN(
                min_cluster_size=dbscan_min_samples,
                min_samples=dbscan_min_samples,
                cluster_selection_epsilon=dbscan_eps * 1.0
            ).fit(scaled)

            labels = clustering.labels_
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    ###########################################################
    # FINAL Bounding Box
    ###########################################################
    results = []
    for lbl in range(num_clusters):
        cluster_pts = filtered_points[labels == lbl]
        if len(cluster_pts) < 10:
            continue

        center = np.mean(cluster_pts, axis=0)
        center[2] = np.max(cluster_pts[:, 2])

        other_axes = [i for i in range(3) if i != axis_idx]
        plane_pts = cluster_pts[:, other_axes]

        pca = PCA(n_components=2).fit(plane_pts)
        eigvals = pca.explained_variance_
        minor_idx = np.argmax(eigvals)
        main_dir = pca.components_[minor_idx]

        yaw = math.atan2(main_dir[1], main_dir[0])

        R = np.array([
            [np.cos(-yaw), -np.sin(-yaw)],
            [np.sin(-yaw),  np.cos(-yaw)]
        ])
        rotated = (cluster_pts[:, :2] - np.mean(cluster_pts[:, :2], axis=0)) @ R.T

        lx = rotated[:,0].max() - rotated[:,0].min()
        ly = rotated[:,1].max() - rotated[:,1].min()

        results.append({
            "id": lbl,
            "x": float(center[0]),
            "y": float(center[1]),
            "z": float(center[2]),
            "yaw": float(yaw),
            "lx": float(lx),
            "ly": float(ly),
            "length": float(ly),
        })

    return filtered_points, labels, results, ground_points


##############################
# tracker
##############################
class Tracker:
    """물체 추적 클래스 (object_recog.py와 동일)"""

    def __init__(self, max_age=10, dist_thresh=0.05):
        self.objects = {}
        self.next_id = 0
        self.max_age = max_age
        self.dist_thresh = dist_thresh

    def update(self, dets):
        # 현재 감지 결과만을 기반으로 트랙을 재구성해 잔상이 남지 않도록 한다.
        if len(dets) == 0:
            self.objects.clear()
            return self.objects

        new_objects = {}

        for det in dets:
            dx, dy = det["x"], det["y"]

            assigned = None
            best = 999
            for oid, obj in self.objects.items():
                ox, oy = obj["x"], obj["y"]
                d = np.hypot(dx - ox, dy - oy)
                if d < best and d < self.dist_thresh:
                    best = d
                    assigned = oid

            if assigned is None:
                assigned = self.next_id
                self.next_id += 1
                det["age"] = 0
            else:
                prev = self.objects[assigned]
                yaw = det["yaw"]
                dyaw = angle_diff(yaw, prev["yaw"])
                yaw = prev["yaw"] + 0.3 * dyaw
                det["yaw"] = yaw
                det["age"] = 0

            new_objects[assigned] = det

        # 감지되지 않은 이전 트랙은 즉시 제거
        self.objects = new_objects
        return self.objects


##############################
# projection
##############################
def project_point(pt, fx, fy, cx, cy):
    """3D 포인트를 2D 픽셀로 투영"""
    X, Y, Z = pt
    if Z <= 0:
        return None
    u = int(X * fx / Z + cx)
    v = int(Y * fy / Z + cy)
    return (u, v)


def get_3d_bbox(x, y, z, yaw, lx, ly):
    """3D 바운딩 박스 생성"""
    dx = lx / 2
    dy = ly / 2

    base = np.array([
        [-dx, -dy, 0],
        [ dx, -dy, 0],
        [ dx,  dy, 0],
        [-dx,  dy, 0]
    ])

    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    pts = base @ R.T
    pts += np.array([x, y, z])
    return pts
