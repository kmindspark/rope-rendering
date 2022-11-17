import enum
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

NUM_STEPS_MIN_FOR_CROSSING = 10
DIST_THRESH = 0.1

input_file_path = './sim_data/annotated'
out_file_path = './processed_sim_data/crop_cage_pinch_dataset'

if not os.path.exists(out_file_path):
    os.makedirs(out_file_path)

files = os.listdir(input_file_path)

for file in files:
    if not file.endswith('.npy'):
        continue

    # we need to find the first graspable point after the first undercrossing
    np_data = np.load(os.path.join(input_file_path, file), allow_pickle=True).item()

    # identify all locations of crossings
    pixels_dict = np_data['pixels']
    pixels = np.zeros((len(pixels_dict), 2))
    for i in range(len(pixels_dict)):
        pixels[i] = np.array(pixels_dict[i])

    points_3d = np.array(np_data['points_3d'])
    crop_img = np_data['crop_img']

    max_cable_height_map = np.zeros(crop_img.shape)
    spread_width = 5
    for i, pixel in enumerate(pixels):
        point_height = points_3d[i, 2]
        px, py = int(pixel[0]), int(pixel[1])
        max_cable_height_map[px-spread_width:px+spread_width, py-spread_width:py+spread_width] = max(max_cable_height_map[px-spread_width:px+spread_width, py-spread_width:py+spread_width], point_height)

    graspability_map = np.zeros(crop_img.shape)

    undercrossing_hit = False
    cage_point_mask = np.zeros(crop_img.shape)
    for pixel in pixels:
        px, py = int(pixel[0]), int(pixel[1])
        undercrossing_hit = undercrossing_hit or (max_cable_height_map[px, py] > points_3d[i, 2])
    
        cage_point_mask = np.zeros(crop_img.shape)
        if undercrossing_hit and not (max_cable_height_map[px, py] > points_3d[i, 2]):
            cage_point_mask[px, py] = 1

    ground_truth = cage_point_mask and graspability_map
