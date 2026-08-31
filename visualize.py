import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from visibility import compute_visibility

# ---------------------------------------------------------
# STEP 1: Your fixed 3D keypoints (satellite body frame, meters)
# ---------------------------------------------------------
keypoints_3d = np.array([
[-0.37,	-0.385,	0.3215],
[-0.37,	0.385,	0.3215],
[0.37,	0.385,	0.3215],
[0.37,	-0.385,	0.3215],
[-0.37,	-0.264,	0],
[-0.37,	0.304,	0],
[0.37,	0.304,	0],
[0.37,	-0.264,	0],
[-0.5427,	0.4877,	0.2535],
[0.5427,	0.4877,	0.2591],
[0.305,	-0.579,	0.2515]

], dtype=np.float32)

# ---------------------------------------------------------
# STEP 2: Camera intrinsics (SPEED camera parameters)
# fx, fy = focal length in pixels | cx, cy = principal point (usually image center)
# ---------------------------------------------------------
fx, fy = 0.0176972364,  0.0176972364   # example values — replace with SPEED+'s actual ones
cx, cy = 960, 600

camera_matrix = np.array([
    [3020.006, 0, 960],
    [0, 3020.006, 600],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.array([-0.2125, 0.4444, -0.000387, -0.000449, 0.5684])  # lens distortion

# ---------------------------------------------------------
# STEP 3: Load one image's ground truth pose from the JSON labels
# ---------------------------------------------------------
# with open("train.json", "r") as f:
#     labels = json.load(f)

#{"filename": "img000240real.jpg", "q_vbs2tango": [0.29082, 0.911185, 0.0209, -0.291082], "r_Vo2To_vbs_true": [-0.372055, -0.078814, 4.203779]
#"img000450real.jpg", "q_vbs2tango": [0.256909, 0.954168, -0.128485, -0.08398], "r_Vo2To_vbs_true": [-0.071347, 0.119053, 3.957217]



# sample = labels[0]                     # pick the first image as a test
# filename = sample["filename"]
filename = "img000240real.jpg"
q =[0.29082, 0.911185, 0.0209, -0.291082]
# q = sample["q_vbs2tango_true"]         # [q0,q1,q2,q3] scalar-first
# t = sample["r_Vo2To_vbs_true"]         # [x,y,z] translation
t = [-0.372055, -0.078814, 4.203779]


# convert quaternion (scalar-first) -> rotation vector for OpenCV
q_scipy = [q[1], q[2], q[3], q[0]]     # reorder to scalar-last for scipy
rot_matrix = R.from_quat(q_scipy).as_matrix() #converts that quaternion into a 3×3 rotation matrix.
rvec, _ = cv2.Rodrigues(rot_matrix) #converts the rotation matrix into OpenCV’s rotation-vector format. OpenCV’s projectPoints expects this representation.
tvec = np.array(t, dtype=np.float32).reshape(3, 1) #converts the translation into a NumPy array of type float32 and reshapes it into a 3×1 column vector, which is the format OpenCV expects.

box_faces = [ #specific to our particular keypoints
    (0,1,2), (0,2,3),
    (4,5,6), (4,6,7),
    (0,1,5), (0,5,4),
    (1,2,6), (1,6,5),
    (2,3,7), (2,7,6),
    (3,0,4), (3,4,7),
]

visibility = compute_visibility(keypoints_3d, rot_matrix, t, box_faces)
#returns visibility array of all points, 2 if visible, 1 if occluded

# ---------------------------------------------------------
# STEP 4: Project the 3D keypoints onto the 2D image
# ---------------------------------------------------------
points_2d, _ = cv2.projectPoints(keypoints_3d, rvec, tvec, camera_matrix, dist_coeffs)
#This uses OpenCV to project each 3D point in keypoints_3d onto the 2D image plane. It applies:
# the rotation rvec
# the translation tvec
# the camera intrinsics camera_matrix
# the lens distortion dist_coeffs
# points_2d comes back as the projected pixel coordinates for each 3D point. 
# The _ is the second return value, which you are ignoring. 
# OpenCV returns it as a matrix related to Jacobians/derivatives.
points_2d = points_2d.reshape(-1, 2)   # shape (11, 2) -> pixel (x,y) per keypoint
print(points_2d)
points_2d = 0.2*points_2d #resize, divide by 5
#OpenCV usually returns the points in a shape like (N, 1, 2). 
# This line flattens it into (N, 2), so each row becomes one point in the form [x,y]

# ---------------------------------------------------------
# STEP 5: Load the image and draw the keypoints on it
# ---------------------------------------------------------
img = cv2.imread(f"./speed/speed/images/real/{filename}")   
img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA) #resize, divide by 5
#interpolation by INTER AREA produces best results compare to other techniques
# adjust path to where your images are stored
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 6))
plt.imshow(img_rgb)
for i, (x, y) in enumerate(points_2d):
    if visibility[i] == 2:
        plt.scatter(x, y, c="lime", s=50, marker="x")   # visible
    else:
        plt.scatter(x, y, c="red", s=50, marker="x")    # occluded/hidden
    plt.text(x + 5, y - 5, str(i), color="yellow", fontsize=9)


for i, (x, y) in enumerate(points_2d):
    plt.text(x + 5, y - 5, str(i), color="yellow", fontsize=9)
plt.title(f"Projected keypoints on {filename}")
plt.axis("off")
plt.show()