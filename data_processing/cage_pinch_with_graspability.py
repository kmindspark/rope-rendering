import enum
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from graspable import find_all_graspability_scores 

NUM_STEPS_MIN_FOR_CROSSING = 10
DIST_THRESH = 0.1
RADIUS = 40
THRESHOLD = 50

input_file_path = './examples'
out_file_path = './processed_sim_data/crop_cage_pinch_dataset'

if not os.path.exists(out_file_path):
    os.makedirs(out_file_path)

input_file = os.listdir(input_file_path)
np_data = np.load(os.path.join(input_file_path, 'test.npy'), allow_pickle=True).item()

def _and_matrices(mat1, mat2):
    if mat1.shape != mat2.shape:
        raise Exception("Dimensions don't match!")
    mat_and = np.copy(mat1)
    for i in range(mat1.shape[0]):
        for j in range(mat1.shape[1]):
            mat_and[i][j] = mat1[i][j] and mat2[i][j]
    return mat_and

# pixels are ordered in the list in the order they are present in the trace
pixel_dict = np_data['pixels']

# note: in a list
points_3d = np_data['points_3d']

pixel_idx_to_point = {}
pixel_point_to_idx = {}

for i in range(len(pixel_dict.keys())):
    pixel_idx_to_point[i] = pixel_dict[i][0]
    pixel_point_to_idx[pixel_dict[i][0]] = i 

graspable_dict = find_all_graspability_scores(radius=RADIUS, trace_threshold=THRESHOLD, pixel_idx_to_point=pixel_idx_to_point, 
    pixel_point_to_idx=pixel_point_to_idx, points_3d=points_3d)

 # identify all locations of crossings
pixels = np.zeros((len(pixel_dict), 2))
for i in range(len(pixel_dict)):
    pixels[i] = np.array(pixel_dict[i])

img = np_data['img']
img_dim_x, img_dim_y = img.shape[0], img.shape[1]

max_cable_height_map = np.ones((img_dim_x, img_dim_y)) * -1 * np.inf

spread_width = 5
for i, pixel in enumerate(pixels):
    point_height = points_3d[i][2]
    px, py = int(pixel[0]), int(pixel[1])
    # ignore off-frame pixels
    if px not in range(img_dim_x) or py not in range(img_dim_y):
        continue
    # constraining bounds for px +/- spread_width, py +/- spread_width to be within frame
    px_start, px_end = max(0, px - spread_width), min(max_cable_height_map.shape[0], px + spread_width)
    py_start, py_end = max(0, py - spread_width), min(max_cable_height_map.shape[1], py + spread_width)
    prev_max = max_cable_height_map[px_start:px_end, py_start:py_end].max()
    new_max = max(prev_max, point_height)
    max_cable_height_map[px_start:px_end, py_start:py_end] = new_max

graspability_map = np.ones((img_dim_x, img_dim_y))
cage_point_mask = np.zeros((img_dim_x, img_dim_y))

undercrossing_hit = False
for i, pixel in enumerate(pixels):
    px, py = int(pixel[0]), int(pixel[1])
    # ignore off-frame pixels
    if px not in range(img_dim_x) or py not in range(img_dim_y):
        continue
    undercrossing_hit = undercrossing_hit or (max_cable_height_map[px][py] > points_3d[i][2])
    if undercrossing_hit and not (max_cable_height_map[px][py] > points_3d[i][2]):
        cage_point_mask[px][py] = 1

ground_truth = _and_matrices(cage_point_mask, graspability_map)

x = []
y = []
color = []
for pixel in pixels:
    # ignore off-frame pixels
    px, py = int(pixel[0]), int(pixel[1])
    if px not in range(img_dim_x) or py not in range(img_dim_y):
        continue
    x.append(px)
    y.append(py)
    if ground_truth[px][py] != 0:
        color.append(1)
    else:
        color.append(0)
plt.scatter(x, y, c = color)
plt.show()
