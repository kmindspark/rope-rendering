import enum
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from graspable import find_all_graspability_scores
from shapely.geometry import LineString
from statistics import mean
from tqdm import tqdm

def _and_matrices(mat1, mat2):
    if mat1.shape != mat2.shape:
        raise Exception("Dimensions don't match!")
    mat_and = np.copy(mat1)
    for i in range(mat1.shape[0]):
        for j in range(mat1.shape[1]):
            mat_and[i][j] = mat1[i][j] and mat2[i][j]
    return mat_and

RADIUS = 40
THRESHOLD = 50
SPREAD_WIDTH = 5

input_file_path = '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_medium_2/train'
out_file_path = './post_processed_sim_data/crop_cage_pinch_dataset'

if not os.path.exists(out_file_path):
    os.makedirs(out_file_path)

files = os.listdir(input_file_path)
num_files = len(files)

for file in tqdm(files):

    if not file.endswith('.npy'):
        continue

    np_data = np.load(os.path.join(input_file_path, file), allow_pickle=True).item()

    pixel_dict = np_data['pixels']
    points_3d = np_data['points_3d']

    # setting up pixels array
    pixels = np.zeros((len(pixel_dict), 2))
    for i in range(len(pixel_dict)):
        pixels[i] = np.array(pixel_dict[i])

    # setting up map from pixel to index and index to pixel
    idx_to_pixel = {}
    pixel_to_idx = {}
    for i in pixel_dict:
        idx_to_pixel[i] = pixel_dict[i][0]
        pixel_to_idx[pixel_dict[i][0]] = i 

    # finding graspable_dict (map from pixels to graspability scores)
    graspable_dict = find_all_graspability_scores(radius=RADIUS, trace_threshold=THRESHOLD, idx_to_pixel=idx_to_pixel, 
        pixel_to_idx=pixel_to_idx, points_3d=points_3d)
    # graspability_threshold = np.percentile(list(set(graspable_dict.values())), 100)
    graspability_threshold = 0

    # setting up img and img boundaries (img_dim_x, img_dim_y)
    img = np_data['img']
    img_dim_x, img_dim_y = img.shape[0], img.shape[1]

    # setting graspability map and cage_point_mask to all 0s
    graspability_map = np.zeros((img_dim_x, img_dim_y))

    # populating graspability map
    for i in range(len(pixels)):
        pixel = idx_to_pixel[i]
        px, py = int(pixel[0]), int(pixel[1])
        if px not in range(img_dim_x) or py not in range(img_dim_y):
            continue
        if pixel in graspable_dict and graspable_dict[pixel] <= graspability_threshold:
            graspability_map[px][py] = 1

    # finding all line segments
    line_segments = [None] * (len(pixels) - 1)
    for i in range(len(pixels) - 1):
        curr_pixel, next_pixel = pixels[i], pixels[i + 1]
        curr_px, curr_py = int(curr_pixel[0]), int(curr_pixel[1]) 
        next_px, next_py = int(next_pixel[0]), int(next_pixel[1]) 
        if curr_px not in range(img_dim_x) or curr_py not in range(img_dim_y):
            continue
        if next_px not in range(img_dim_x) or next_py not in range(img_dim_y):
            continue
        line_segments[i] = LineString([curr_pixel, next_pixel])

    # populating cage_point_mask to include points after, and including, first undercrossing
    undercrossing_hit, overcrossing_hit = False, False
    seen_pixels = set()
    for i in range(len(pixels) - 1):
        px, py = int(pixels[i][0]), int(pixels[i][1]) 
        if not line_segments[i]:
            continue
        if ((px, py)) in seen_pixels:
            continue
        else:
            seen_pixels.add((px, py))
        ls = line_segments[i]
        for j in range(len(line_segments)):
            if not line_segments[j]:
                continue
            other_ls = line_segments[j]
            if other_ls != ls and len(set(ls.coords).intersection(set(other_ls.coords))) == 0:
                # crossing identified (unique, intersecting line segments with no endpoints in common)
                if ls.intersects(other_ls):
                    # undercrossing identified (mean of ls' endpoints (height) < mean of other_ls' endpoints (height))
                    if mean([points_3d[i][2], points_3d[i + 1][2]]) < mean([points_3d[j][2], points_3d[j + 1][2]]):
                        # uncomment next line for debugging:
                        # print(ls, other_ls)
                        # finding first undercrossing
                        if not undercrossing_hit:
                            undercrossing_hit = True
                            continue
        # finding the first graspable point after the discovered undercrossing and terminating 
        graspability_threshold = 0
        if undercrossing_hit and graspability_map[px][py] == 1:
            cage_point = pixels[i]
            break

    # skip if no graspable cage points found
    if cage_point is None:
        print("No cage point found: " + file[:-4])
        continue

    x = []
    y = []
    color = []
    for pixel in pixels:
        px, py = int(pixel[0]), int(pixel[1])
        # ignore off-frame pixels
        if px not in range(img_dim_x) or py not in range(img_dim_y):
            continue
        x.append(px)
        y.append(py)
        if (px, py) == (int(cage_point[0]), int(cage_point[1])):
            color.append(1)
        else:
            color.append(0)

    np_post_processing_data = {}
    np_post_processing_data['img'] = img
    np_post_processing_data['pixels'] = pixels
    np_post_processing_data['cage_point'] = cage_point

    np.save(os.path.join(out_file_path, file)[:-4] + '.npy', np_post_processing_data)

    # plt.clf()
    # plt.scatter(x, y, c=color)
    # plt.annotate('Start', (x[0], y[0]))
    # plt.annotate('Cage', cage_point)
    # plt.savefig(os.path.join(out_file_path, file)[:-4] + '.png')
    # plt.clf()
    # plt.imsave(os.path.join(out_file_path, file)[:-4] + '_img.png', img, origin="lower")
