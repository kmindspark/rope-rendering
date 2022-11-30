import enum
import os
import numpy as np
import matplotlib.pyplot as plt
import math

'''
helper that normalizes map
'''
def _normalize_map(dict):
    min_score, max_score = min(dict.values()), max(dict.values())
    for k in dict:
        dict[k] = (dict[k] -  min_score) / (max_score - min_score)
    return dict

'''
helper that finds the spatial distance between two points in 3D
'''
def _get_3d_dist(point1_3d, point2_3d):
    dist_sq = 0
    for i in range(3):
        dist_sq += (point1_3d[i] - point2_3d[i]) ** 2
    return math.sqrt(dist_sq)

'''
finds the spatial distance between points corresponding to two pixels
'''
def find_3d_dist(pixel1, pixel2, pixel_to_idx, points_3d):
    idx1 = pixel_to_idx[pixel1]
    idx2 = pixel_to_idx[pixel2]
    point1_3d = points_3d[idx1]
    point2_3d = points_3d[idx2]
    return _get_3d_dist(point1_3d, point2_3d)

'''
finds the pixel distance between two pixels
'''
def find_pixel_dist(pixel1, pixel2):
    return math.sqrt((pixel1[0] - pixel2[0]) ** 2 + (pixel1[1] - pixel2[1]) ** 2)

'''
finds the trace distance between two pixels (i.e. the difference in indices in the trace)
'''
def find_trace_dist(pixel1, pixel2, pixel_to_idx):
    return abs(pixel_to_idx[pixel1] - pixel_to_idx[pixel2])

'''
explores the pixel's neighbors 
    in3D=True: explore neighbors spatially, within a sphere of specified radius
    in3D=False: explore neighbors on the image, within a circle of specified radius
outputs a score of the (# of points inside radius that are outside trace_threshold) / (total # points)
score lies between 0 and 1 (where the higher the score, the larger the chance of being ungraspable)
'''
def find_pixel_point_graspability(pixel, radius, trace_threshold, pixel_to_idx, points_3d, in3D=False):
    if in3D:
        total_points = 4 / 3 * math.pi * (radius ** 3)
    else:
        total_points = math.pi * (radius ** 2)
    total_points = 4 / 3 * math.pi * (radius ** 3)
    points_outside_trace_threshold = 0
    start_pixel_x = pixel[0] - radius
    start_pixel_y = pixel[1] - radius
    for i in range(start_pixel_x, start_pixel_x + 2 * radius):
        for j in range(start_pixel_y, start_pixel_y  + 2 * radius):
            neigh = (i, j)
            if neigh not in pixel_to_idx.keys():
                continue
            inRadius = False
            if in3D:
                inRadius = find_3d_dist(neigh, pixel, pixel_to_idx, points_3d) <= radius
            else:
                inRadius = find_pixel_dist(neigh, pixel) <= radius
            if inRadius:
                if find_trace_dist(pixel, neigh, pixel_to_idx) > trace_threshold:
                    points_outside_trace_threshold += 1
    # TODO: think about the denominator
    return points_outside_trace_threshold / total_points

def find_all_graspability_scores(radius, trace_threshold, idx_to_pixel, pixel_to_idx, points_3d, in3D=False):
    # creates dict from point to graspability
    graspability_scores = {}
    for i in range(len(idx_to_pixel)):
        pixel = idx_to_pixel[i]
        graspability_scores[pixel] = find_pixel_point_graspability(pixel, radius, trace_threshold, pixel_to_idx, 
            points_3d, in3D)
    # return _normalize_map(graspability_scores)
    return graspability_scores

def graph_graspability_heatmap(graspability_dict, img_dim_x, img_dim_y):
    x = []
    y = []
    color = []
    for pixel in graspability_dict.keys():
        px, py = pixel[0], pixel[1]
        if px not in range(img_dim_x) or py not in range(img_dim_y):
            continue
        x.append(px)
        y.append(py)
        color.append(graspability_dict[pixel])
    plt.scatter(x, y, c=color)
    plt.show()

def main():
    input_file_path = './examples'

    # loads a particular file path in 
    np_data = np.load(os.path.join(input_file_path, 'test.npy'), allow_pickle=True).item()

    RADIUS = 40
    THRESHOLD = 50
    IMG_DIM_X = 800
    IMG_DIM_Y = 800

    # pixels are ordered in the list in the order they are present in the trace
    # note: in a dict
    pixels = np_data['pixels']

    pixel_to_idx = {}
    idx_to_pixel = {}
    for i in pixels:
        idx_to_pixel[i] = pixels[i][0]
        pixel_to_idx[pixels[i][0]] = i 

    # note: in a list
    points_3d = np_data['points_3d']

    graspable_dict = find_all_graspability_scores(radius=RADIUS, trace_threshold=THRESHOLD, idx_to_pixel=idx_to_pixel, 
        pixel_to_idx=pixel_to_idx, points_3d=points_3d)
    graph_graspability_heatmap(graspable_dict, img_dim_x=IMG_DIM_X, img_dim_y=IMG_DIM_Y)

if __name__ == '__main__':
    main()
