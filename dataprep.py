import numpy as np
import h5py
import os


def read_off(filename):
    num_select = 1024
    f = open(filename)
    f.readline()  # ignore the 'OFF' at the first line
    f.readline()  # ignore the second line
    All_points = []
    selected_points = []
    while True:
        new_line = f.readline()
        x = new_line.split(' ')
        if x[0] != '3':
            A = np.array(x[0:3], dtype='float32')
            All_points.append(A)
        else:
            break
    # if the numbers of points are less than 2000, extent the point set
    if len(All_points) < (num_select + 3):
        return None
    # take and shuffle points
    index = np.random.choice(len(All_points), num_select, replace=False)
    for i in range(len(index)):
        selected_points.append(All_points[index[i]])
    return selected_points  # return N*3 array


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


cur_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(cur_path, "ModelNet40")
# list of all the categories
directories = [d for d in os.listdir(dir_path)
               if os.path.isdir(os.path.join(dir_path, d))]


#######
load_dict = [["train", "PrepData"], ["test", "PrepData_test"]]
for d in load_dict:
    for i in range(len(directories)):
        label = directories[i]
        train_path = os.path.join(dir_path, directories[i], d[0])
        save_path = os.path.join(cur_path, d[1])
        All_points = None
        label = []
        # all the files in "train" floder
        for filename in os.listdir(train_path):
            # print(filename)
            if '.off' in filename:
                s = os.path.join(train_path, filename)
                points = read_off(s)
                if All_points is None:
                    if points:
                        All_points = points
                        label.append(i)
                elif points:
                    All_points = np.vstack((All_points, points))
                    label.append(i)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data_save_path = os.path.join(save_path, directories[i] + '.h5')
        save_h5(data_save_path, All_points, label)
        print(All_points.shape)
        print(len(label))
