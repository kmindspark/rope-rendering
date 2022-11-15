import enum
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


NUM_STEPS_MIN_FOR_CROSSING = 10
DIST_THRESH = 0.1

input_file_path = './examples'
out_file_path = './processed_sim_data/crop_cage_pinch_dataset'

# if not os.path.exists(out_file_path):
#     os.makedirs(out_file_path)

input_file = os.listdir(input_file_path)
np_data = np.load(os.path.join(input_file_path, '000001_rgb.npy'), allow_pickle=True).item()
print(np_data.keys())
