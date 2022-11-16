import enum
import os
import numpy as np
import matplotlib.pyplot as plt
import math

'''
helper that finds the spatial distance between two points in 3D
'''
def _get_3d_dist(point1_3d, point2_3d):
    dist_sq = 0
    for i in range(3):
        dist_sq += (point1_3d[i] - point2_3d[i]) ** 2
    return math.sqrt(dist_sq)

'''
finds the spatial distance between two pixel points
'''
def find_3d_dist(point1, point2, pixel_point_to_idx, points_3d):
    idx1 = pixel_point_to_idx[point1]
    idx2 = pixel_point_to_idx[point2]
    point1_3d = points_3d[idx1]
    point2_3d = points_3d[idx2]
    return _get_3d_dist(point1_3d, point2_3d)

'''
finds the pixel distance between two pixel points
'''
def find_pixel_dist(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

'''
finds the trace distance between two pixel points
returns the difference in indices in the trace
'''
def find_trace_dist(point1, point2, pixel_point_to_idx):
    return abs(pixel_point_to_idx[point1] - pixel_point_to_idx[point2])

'''
explores the pixel's neighbors 
    in3D=True: explore neighbors spatially, within a sphere of specified radius
    in3D=False: explore neighbors on the image, within a circle of specified radius
outputs a score of the (# of points inside radius that are outside trace_threshold) / (total # points)
score lies between 0 and 1 (where the higher the score, the larger the chance of being ungraspable)
'''
def find_pixel_point_graspability(point, radius, trace_threshold, pixel_point_to_idx, points_3d, in3D=False):
    if in3D:
        total_points = 4 / 3 * math.pi * (radius ** 3)
    else:
        total_points = math.pi * (radius ** 2)
    total_points = 4 / 3 * math.pi * (radius ** 3)
    points_outside_trace_threshold = 0
    start_point_x = point[0] - radius
    start_point_y = point[1] - radius
    for i in range(start_point_x, start_point_x + 2 * radius):
        for j in range(start_point_y, start_point_y  + 2 * radius):
            neigh = (i, j)
            if neigh not in pixel_point_to_idx.keys():
                continue
            inRadius = False
            if in3D:
                inRadius = find_3d_dist(neigh, point, pixel_point_to_idx, points_3d) <= radius
            else:
                inRadius = find_pixel_dist(neigh, point) <= radius
            if inRadius:
                if find_trace_dist(point, neigh, pixel_point_to_idx) > trace_threshold:
                    points_outside_trace_threshold += 1
    # TODO: think about the denominator
    return points_outside_trace_threshold / total_points

def find_all_graspability_scores(radius, trace_threshold, pixel_idx_to_point, pixel_point_to_idx, points_3d, in3D=False):
    # creates dict from point to graspability
    graspability_scores = {}
    for i in range(len(pixel_idx_to_point)):
        point = pixel_idx_to_point[i]
        graspability_scores[point] = find_pixel_point_graspability(point, radius, trace_threshold, pixel_point_to_idx, 
            points_3d, in3D)
    return graspability_scores

def graph_graspability_heatmap(graspability_dict):
    x = []
    y = []
    color = []
    for point in graspability_dict.keys():
        x.append(point[0])
        y.append(point[1])
        color.append(graspability_dict[point])
    plt.scatter(x,y,  c = color)
    plt.show()

def main():
    input_file_path = './examples'

    # loads a particular file path in 
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
    graph_graspability_heatmap(graspable_dict)

if __name__ == '__main__':
    main()
