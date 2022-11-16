import enum
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from graspable import find_all_graspability_scores 

NUM_STEPS_MIN_FOR_CROSSING = 10
DIST_THRESH = 0.1

input_file_path = './examples'
out_file_path = './processed_sim_data/crop_cage_pinch_dataset'

if not os.path.exists(out_file_path):
    os.makedirs(out_file_path)

input_file = os.listdir(input_file_path)
np_data = np.load(os.path.join(input_file_path, 'test.npy'), allow_pickle=True).item()

RADIUS = 40
THRESHOLD = 50

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

max_cable_height_map = np.zeros(img.shape)
spread_width = 5
for i, pixel in enumerate(pixels):
    point_height = points_3d[i][2]
    px, py = int(pixel[0]), int(pixel[1])
    max_cable_height_map[px-spread_width:px+spread_width, py-spread_width:py+spread_width] = max(
        max_cable_height_map[px-spread_width:px+spread_width, py-spread_width:py+spread_width].max(), 
        point_height)

graspability_map = np.zeros(img.shape)

undercrossing_hit = False

cage_point_mask = np.zeros(img.shape)
for pixel in pixels:
    px, py = int(pixel[0]), int(pixel[1])
    undercrossing_hit = undercrossing_hit or (max_cable_height_map[px, py] > points_3d[i, 2])

    cage_point_mask = np.zeros(img.shape)
    if undercrossing_hit and not (max_cable_height_map[px, py] > points_3d[i, 2]):
        cage_point_mask[px, py] = 1

print(graspable_dict)
print("---")
print(cage_point_mask)
ground_truth = cage_point_mask and graspability_map


