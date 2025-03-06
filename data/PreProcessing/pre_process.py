from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from posixpath import basename
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import train_test_split, KFold

import cv2


# change DATASET ROOT to your dataset path
DATASET_ROOT = '/your/path/to/historicalMaps'


TRAIN_ROOT = os.path.join(DATASET_ROOT, 'train')
VAL_ROOT = os.path.join(DATASET_ROOT, 'val')
TEST_ROOT = os.path.join(DATASET_ROOT, 'test/images')


"""
In the loaded numpy array, only 0-6 integer labels are allowed, and they represent the annotations in the following way:

0 - frame
1 - water
2 - blocks
3 - non-built
4 - road network

"""

palette_land = {
    0 : (0, 0, 0),        # frame
    1 : (0, 0, 255),    # water
    2 : (255, 0, 255),    # blocks
    3 : (0, 255, 255),      # non-build
    4 : (255, 255, 255),      # road network
}


labels_folder = {
    'frame': 1,
    'water': 2,
    'blocks': 3,
    'non-build': 4,
    'road_network': 5,
}
# "frame",
land_classes = ["frame", "water", "blocks", "non-build",
                "road_network"]


Data_Folder = {
    'Agriculture': {
        'ROOT': DATASET_ROOT,
        'RGB': 'images/{}.png',
        'SHAPE': (500, 500),
        'GT': 'gt/{}.png',
    },
}

IMG = 'images' # RGB or IRRG, rgb/nir
GT = 'gt'
IDS = 'IDs'


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def img_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


def is_image(filename):
    return any(filename.endswith(ext) for ext in ['.png', '.jpg'])

# def prepare_gt(root_folder = TRAIN_ROOT, out_path='gt'):
#     if not os.path.exists(os.path.join(root_folder, out_path)):
#         print('----------creating groundtruth data for training./.val---------------')
#         check_mkdir(os.path.join(root_folder, out_path))
#         basname = [img_basename(f) for f in os.listdir(os.path.join(root_folder,'images'))]
#         gt_raw = basname[0]+'.png'
#
#         if gt_raw[:2] == "._":
#             gt = gt_raw[2:]
#         else:
#             gt = gt_raw
#
#         for fname in basname:
#             gtz = np.zeros((1000, 1000), dtype=int)
#             # for key in labels_folder.keys():
#                 # print(key)
#             gt_raw = fname + '.png'
#             if gt_raw[:2] == "._":
#                 gt = gt_raw[2:]
#             else:
#                 gt = gt_raw
#
#             print(os.path.join(root_folder, 'labels', gt), " das ist diese super kompliziertes auslesen von den gt images")
#             # mask = np.array(cv2.imread(os.path.join(root_folder, 'labels', gt), -1)/255, dtype=int)
#             # das mask war für die Zuordnung da... so wie bei if 1 = das else 0
#             # gtz[gtz < 1] = mask[gtz < 1]
#             # das ist BGR
#             image_BGR = cv2.imread(os.path.join(root_folder, 'labels', gt))
#             image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
#             # print(type(image_RGB), " image type")
#
#             image_grayscale = np.zeros((1000,1000) , dtype=int)
#
#             for i in range(np.shape(image_grayscale)[1]):
#                 for k in range(np.shape(image_grayscale)[0]):
#
#                     # 0 = frame
#                     if image_RGB[i,k,0] == 0 and image_RGB[i,k,1] == 0 and image_RGB[i,k,2] == 0:
#                         image_grayscale[i,k] = 0;
#                     # 1 = water
#                     elif image_RGB[i,k,0] == 0 and image_RGB[i,k,1] == 0 and image_RGB[i,k,2] == 255:
#                         image_grayscale[i,k] = 1;
#                     # 2 = blocks
#                     elif image_RGB[i,k,0] == 255 and image_RGB[i,k,1] == 0 and image_RGB[i,k,2] == 255:
#                         image_grayscale[i,k] = 2;
#                     # 3 = non-build
#                     elif image_RGB[i,k,0] == 0 and image_RGB[i,k,1] == 255 and image_RGB[i,k,2] == 255:
#                         image_grayscale[i,k] = 3;
#                     # 4 = streets
#                     elif image_RGB[i,k,0] == 255 and image_RGB[i,k,1] == 255 and image_RGB[i,k,2] == 255:
#                         image_grayscale[i,k] = 4;
#
#
#
#
#             # print(np.shape(image_grayscale), " grayscale shape")
#
#             # for key in ['boundaries', 'masks']:
#             #     mask = np.array(cv2.imread(os.path.join(root_folder, key), -1) / 255, dtype=int)
#             #     gtz[mask == 0] = 255
#             # print(os.path.join(root_folder, out_path, gt))
#             cv2.imwrite(os.path.join(root_folder, out_path, gt), image_grayscale)


def get_training_list(root_folder = TRAIN_ROOT, count_label=True):
    dict_list = {}
    basname = [img_basename(f) for f in os.listdir(os.path.join(root_folder, 'images'))]
    if count_label:
        for key in labels_folder.keys():
            no_zero_files=[]
            for fname in basname:
                gt = np.array(cv2.imread(os.path.join(root_folder, 'labels', key, fname+'.png'), -1))
                if np.count_nonzero(gt):
                    no_zero_files.append(fname)
                else:
                    continue
            dict_list[key] = no_zero_files
    return dict_list, basname


def split_train_val_test_sets(data_folder=Data_Folder, name='Agriculture', bands=['RGB'], KF=3, k=1, seeds=69278):

    train_id, t_list = get_training_list(root_folder=TRAIN_ROOT, count_label=False)
    val_id, v_list = get_training_list(root_folder=VAL_ROOT, count_label=False)

    if KF >=2:
        kf = KFold(n_splits=KF, shuffle=True, random_state=seeds)
        val_ids = np.array(v_list)
        idx = list(kf.split(np.array(val_ids)))
        if k >= KF:  # k should not be out of KF range, otherwise set k = 0
            k = 0
        t2_list, v_list = list(val_ids[idx[k][0]]), list(val_ids[idx[k][1]])
    else:
        t2_list=[]

    img_folders = [os.path.join(data_folder[name]['ROOT'], 'train', data_folder[name][band]) for band in bands]
    gt_folder = os.path.join(data_folder[name]['ROOT'], 'train', data_folder[name]['GT'])


    val_folders = [os.path.join(data_folder[name]['ROOT'], 'val', data_folder[name][band]) for band in bands]
    val_gt_folder = os.path.join(data_folder[name]['ROOT'], 'val', data_folder[name]['GT'])

    train_dict = {
        IDS: train_id,
        IMG: [[img_folder.format(id) for img_folder in img_folders] for id in t_list] +
             [[val_folder.format(id) for val_folder in val_folders] for id in t2_list],
        GT: [gt_folder.format(id) for id in t_list] + [val_gt_folder.format(id) for id in t2_list],
        'all_files': t_list + t2_list
    }
    #

    val_dict = {
        IDS: val_id,
        IMG: [[val_folder.format(id) for val_folder in val_folders] for id in v_list],
        GT: [val_gt_folder.format(id) for id in v_list],
        'all_files': v_list
    }

    test_dict = {
        IDS: val_id,
        IMG: [[val_folder.format(id) for val_folder in val_folders] for id in v_list],
        GT: [val_gt_folder.format(id) for id in v_list],
    }
    
    print('train set -------', len(train_dict[GT]))
    print('val set ---------', len(val_dict[GT]))
    return train_dict, val_dict, test_dict


def get_real_test_list(root_folder = TEST_ROOT, data_folder=Data_Folder, name='Agriculture', bands=['RGB']):
    dict_list = {}
    basname = [img_basename(f) for f in os.listdir(os.path.join(root_folder, ''))]
    dict_list['all'] = basname

    test_dict = {
        IDS: dict_list,
        IMG: [os.path.join(data_folder[name]['ROOT'], 'test', data_folder[name][band]) for band in bands],
    }

    print(test_dict)
    return test_dict
