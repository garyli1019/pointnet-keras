# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 15:35:46 2018

@author: Gary
"""

import numpy as np
import h5py
import os

def save_h5(h5_filename, data, label, data_dtype='float32', label_dtype='float32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
        'data', data=data,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'label', data=label,
        compression='gzip', compression_opts=1,
        dtype=label_dtype)
    h5_fout.close()

def select_points(points, num_points):
    selected_points = []
    # shuffle points
    if len(points) > num_points:
        index = np.random.choice(len(points), num_points, replace=False)
    else:
        index = np.random.choice(len(points), num_points, replace=True)
    for i in range(len(index)):
        selected_points.append(points[index[i]])
    selected_points = np.array(selected_points)
    return selected_points  # return N*3 array

# number of points select from each file
num_points = 1024
# category to seg
c = "Airplane"

path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(path, "shapenetcore_partanno_v0")
cat_file = os.path.join(dir_path, 'synsetoffset2category.txt')
categories_dict = {}
with open(cat_file, 'r') as f:
    for line in f:
        ls = line.strip().split()
        categories_dict[ls[0]] = ls[1]

cur_folder = os.path.join(dir_path, categories_dict[c])
points_path = os.path.join(cur_folder, "points")
labels_path = os.path.join(cur_folder, "points_label")


points_filename = [d for d in os.listdir(points_path)]
# filename without format
x = []
for f in points_filename:
    x.append(f.split('.')[0])
label_nums = [d for d in os.listdir(labels_path)]

# prepare train file
All_points = None
All_labels = None
# select 80% of data to train
for i in range(int(len(points_filename)*0.8)):
    cur_points_path = os.path.join(points_path,points_filename[i])
    cur_points = np.loadtxt(cur_points_path).astype('float32')
    for n in range(len(label_nums)):
        label_filename = x[i] + '.seg'
        p = os.path.join(labels_path, label_nums[n], label_filename)
        if os.path.isfile(p):
            cur_label = np.loadtxt(p).astype('float32')
            cur_label = cur_label.reshape(len(cur_label), 1)
            cur_points = np.hstack((cur_points, cur_label))
        else:
            # only load data with complete labels
            print("Label file missing", str(x[i]))
            break
    if cur_points.shape[1] == len(label_nums) + 3:
        selected_points = select_points(cur_points, num_points)
        point_cloud = selected_points[:,0:3]
        labels = selected_points[:,3:(3+len(label_nums))]
        if All_points is None:
            All_points = point_cloud
        else:
            All_points = np.vstack((All_points, point_cloud))
        if All_labels is None:
            All_labels = labels
        else:
            All_labels = np.vstack((All_labels, labels))
# save the train file
save_path = os.path.join(path, "Seg_Prep")
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_path = os.path.join(save_path, c+'.h5')
save_h5(save_path, All_points, All_labels)

# prepare test file
All_points = None
All_labels = None
# select 20% of data to test
for i in range(int(len(points_filename)*0.8), len(points_filename)):
    cur_points_path = os.path.join(points_path,points_filename[i])
    cur_points = np.loadtxt(cur_points_path).astype('float32')
    for n in range(len(label_nums)):
        label_filename = x[i] + '.seg'
        p = os.path.join(labels_path, label_nums[n], label_filename)
        if os.path.isfile(p):
            cur_label = np.loadtxt(p).astype('float32')
            cur_label = cur_label.reshape(len(cur_label), 1)
            cur_points = np.hstack((cur_points, cur_label))
        else:
            print("Label file missing", str(x[i]))
            break
    if cur_points.shape[1] == len(label_nums) + 3:
        selected_points = select_points(cur_points, num_points)
        point_cloud = selected_points[:,0:3]
        labels = selected_points[:,3:(3+len(label_nums))]
        if All_points is None:
            All_points = point_cloud
        else:
            All_points = np.vstack((All_points, point_cloud))
        if All_labels is None:
            All_labels = labels
        else:
            All_labels = np.vstack((All_labels, labels))
# save the test file
save_path = os.path.join(path, "Seg_Prep_test")
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_path = os.path.join(save_path, c+'.h5')
save_h5(save_path, All_points, All_labels) 
            
        
    
        
    