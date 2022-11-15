import enum
import os
import numpy as np
import matplotlib.pyplot as plt
import math

input_file_path = './examples'

#loads a particular file path in 
np_data = np.load(os.path.join(input_file_path, '000000_rgb.npy'), allow_pickle=True).item()

#pixels are ordered in the list in the order they are present in the trace
pixel_dict = np_data['pixels']
pixel_idx_to_point = {}
pixel_point_to_idx = {}
for i in range(len(pixel_dict.keys())):
    pixel_idx_to_point[i] = pixel_dict[i][0]
    pixel_point_to_idx[pixel_dict[i][0]] = i

#in a list
points_3d_idx_to_point = np_data['points_3d'] 

def find_3d_dist(point1_3d, point2_3d):
    dist_sq = 0
    for i in range(3):
        dist_sq += (point1_3d[i] - point2_3d[i]) ** 2
    # print("3d_dist", math.sqrt(dist_sq))
    return math.sqrt(dist_sq)


'''
finds the trace_dist between two pixel points
returns the difference in actual x,y,z distances.
'''
def find_trace_dist(point1, point2, pixel_point_to_idx):
    idx1 = pixel_point_to_idx[point1]
    idx2 = pixel_point_to_idx[point2]
    point1_3d = points_3d_idx_to_point[idx1]
    point2_3d = points_3d_idx_to_point[idx2]
    return find_3d_dist(point1_3d, point2_3d)


'''
finds the trace_dist between two pixel points
returns the difference in indices in the trace
'''
def find_trace_dist_naive(point1, point2, pixel_point_to_idx ):
    return abs(pixel_point_to_idx[point1] - pixel_point_to_idx[point2])

'''
explores the pixel's neighbors spatially (within a box of specifed radius)
outputs a score of the (# of points in box that are outside trace_threshold) / (total # points)

score lies between 0 and 1 (where the higher the score, the larger the chance of being ungraspable )
'''
def find_pixel_point_graspability(point, radius, trace_threshold):
    total_points = 4 * (radius ** 2)
    points_outside_trace_threshold = 0
    start_point_x = point[0] - radius
    start_point_y = point[1] - radius

    for i in range(start_point_x, start_point_x + 2 * radius):
        for j in range(start_point_y, start_point_y  + 2 * radius):
            neigh = (i,j)
            if neigh in pixel_point_to_idx.keys():
                #if the neighboring pixel is in the rope trace and is within 
                if(find_trace_dist(point, neigh, pixel_point_to_idx) > trace_threshold):
                    points_outside_trace_threshold += 1
    return points_outside_trace_threshold / total_points


def find_all_graspability_scores(radius, trace_threshold):
    #creates dict from point to graspability
    graspability_scores = {}
    for i in range(len(pixel_idx_to_point)):
        point = pixel_idx_to_point[i]
        graspability_scores[point] = find_pixel_point_graspability(point, radius, trace_threshold)
    
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
    # plt.show()
    plt.savefig('test1.png')


graspable_dict = find_all_graspability_scores(radius=20, trace_threshold=0.025)
graph_graspability_heatmap(graspable_dict)
