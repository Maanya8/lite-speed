import numpy as np


def compute_visibility(keypoints_3d, rot_matrix, tvec, faces, self_hit_eps=1e-4, eps=1e-7):

    #eps 1e-7 -> This one guards against dividing by zero when a = np.dot(e1, h) 
    # is near zero (ray parallel to the triangle's plane). 
    # It's not about your object's scale — it's about floating-point precision.

    # As a general heuristic: self_hit_eps ≈ 1/1000th to 1/10000th of your object's 
    # smallest relevant dimension is a safe starting range. For your satellite 
    # (smallest dimension ~0.26–0.32m), that puts you around 1e-4 to 3e-5, which 
    # is why 1e-4 is a sensible default
    #If real occlusions are being missed, decrease self_hit_eps.
    """
    Determine self-occlusion visibility for a set of 3D keypoints on a rigid body.

    Parameters
    ----------
    keypoints_3d : (N,3) array-like
        3D keypoints in the body frame.
    rot_matrix : (3,3) array-like
        Rotation matrix mapping body frame -> camera frame (X_cam = R @ X_body + t).
    tvec : (3,) or (3,1) array-like
        Translation vector, body frame -> camera frame.
    faces : list of (i,j,k) tuples
        Triangle faces as indices into keypoints_3d, describing a coarse mesh
        of the rigid body used to test occlusion.
    self_hit_eps : float
        Margin to avoid false positives when a ray's true target point sits
        exactly on a face.
    eps : float
        Numerical tolerance for the ray-triangle intersection test.

    Returns
    -------
    vis : (N,) np.ndarray of int
        Visibility flag per keypoint, COCO-style: 2 = visible, 1 = occluded.
    """
    keypoints_3d = np.asarray(keypoints_3d, dtype=np.float64)
    rot_matrix = np.asarray(rot_matrix, dtype=np.float64)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)

    def ray_triangle_intersect(orig, direction, v0, v1, v2):
        """Moller-Trumbore ray-triangle intersection. Returns t or None."""
        e1 = v1 - v0
        e2 = v2 - v0
        h = np.cross(direction, e2)
        a = np.dot(e1, h)
        if abs(a) < eps:
            return None
        f = 1.0 / a
        s = orig - v0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return None
        q = np.cross(s, e1)
        v = f * np.dot(direction, q)
        if v < 0.0 or u + v > 1.0:
            return None
        t = f * np.dot(e2, q)
        return t if t > eps else None
    #t means viewpoint is occluded, None means its not

    # Camera center in body frame: solve 0 = R @ C + t  ->  C = -R^T @ t
    cam_center_body = -rot_matrix.T @ tvec

    vis = []
    for i, kp in enumerate(keypoints_3d):
        direction = kp - cam_center_body
        dist_to_point = np.linalg.norm(direction)
        d_norm = direction / dist_to_point

        occluded = False
        for face in faces:
            if i in face:
                continue
            v0, v1, v2 = keypoints_3d[list(face)]
            hit_t = ray_triangle_intersect(cam_center_body, d_norm, v0, v1, v2)
            if hit_t is not None and hit_t < dist_to_point - self_hit_eps:
                occluded = True
                break

        vis.append(1 if occluded else 2)

    #print(np.array(vis))

    return np.array(vis)