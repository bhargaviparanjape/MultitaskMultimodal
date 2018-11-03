"""
Reads in a tsv file with pre-trained bottom up attention features and
stores it in HDF5 format.  Also store {image_id: feature_idx}
 as a pickle file.

Hierarchy of HDF5 file:

{ 'image_features': num_images x num_boxes x 2048 array of features
  'image_bb': num_images x num_boxes x 4 array of bounding boxes }
"""
from __future__ import print_function

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import csv
import h5py
import cPickle
import numpy as np
import utils
import sys
import os
import pdb


csv.field_size_limit(sys.maxsize)

## TODO : Additional field 'gold_box'
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
data_root = sys.argv[1]
coco_root = sys.argv[2]
infile = os.path.join(data_root, 'trainval_subset.tsv')
train_data_file = os.path.join(data_root, 'train36_small1.hdf5')
val_data_file = os.path.join(data_root, 'val36_small1.hdf5')
train_indices_file = os.path.join(data_root, 'val36_imgid2idx_small.pkl')
val_indices_file = os.path.join(data_root, 'val36_imgid2idx_small.pkl')
train_ids_file = os.path.join(data_root, 'val_ids_small.pkl')
val_ids_file = os.path.join(data_root, 'val_ids_small.pkl')

feature_length = 2048
num_fixed_boxes = 36
GOLD = 1


if __name__ == '__main__':
    h_train = h5py.File(train_data_file, "w")
    h_val = h5py.File(val_data_file, "w")

    if os.path.exists(train_ids_file) and os.path.exists(val_ids_file):
        train_imgids = cPickle.load(open(train_ids_file))
        val_imgids = cPickle.load(open(val_ids_file))
    else:
        train_imgids = utils.load_imageid(os.path.join(coco_root, 'train2014'))
        val_imgids = utils.load_imageid(os.path.join(coco_root, 'val2014'))
        cPickle.dump(train_imgids, open(train_ids_file, 'wb'))
        cPickle.dump(val_imgids, open(val_ids_file, 'wb'))

    train_indices = {}
    val_indices = {}
    
    images_in_train_tsv_count = 0
    images_in_val_tsv_count = 0
    tsv_train_img_ids = []
    tsv_val_img_ids = []
    with open(infile, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            if item['image_id'] in train_imgids:
                images_in_train_tsv_count += 1
                tsv_train_img_ids.append(item['image_id'])
            else:
                images_in_val_tsv_count += 1
                tsv_val_img_ids.append(item['image_id'])
    train_img_features = h_train.create_dataset(
        'image_features', (images_in_train_tsv_count, num_fixed_boxes, feature_length), 'f')
    train_img_bb = h_train.create_dataset(
        'image_bb', (images_in_train_tsv_count, num_fixed_boxes, 4), 'f')
    train_spatial_img_features = h_train.create_dataset(
        'spatial_features', (images_in_train_tsv_count, num_fixed_boxes, 6), 'f')
    train_gold_id = h_train.create_dataset(
        'gold_box', (images_in_train_tsv_count, 1), 'i8')

    val_img_bb = h_val.create_dataset(
        'image_bb', (images_in_val_tsv_count, num_fixed_boxes, 4), 'f')
    val_img_features = h_val.create_dataset(
        'image_features', (images_in_val_tsv_count, num_fixed_boxes, feature_length), 'f')
    val_spatial_img_features = h_val.create_dataset(
        'spatial_features', (images_in_val_tsv_count, num_fixed_boxes, 6), 'f')
    val_gold_id = h_val.create_dataset(
        'gold_box', (images_in_val_tsv_count, 1), 'i8')

    train_counter = 0
    val_counter = 0

    print("reading tsv...")
    with open(infile, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            item['num_boxes'] = int(item['num_boxes'])
            image_id = int(item['image_id'])
            image_w = float(item['image_w'])
            image_h = float(item['image_h'])
            bboxes = np.frombuffer(
                base64.decodestring(item['boxes']),
                dtype=np.float32).reshape((item['num_boxes'], -1))

            box_width = bboxes[:, 2] - bboxes[:, 0]
            box_height = bboxes[:, 3] - bboxes[:, 1]
            scaled_width = box_width / image_w
            scaled_height = box_height / image_h
            scaled_x = bboxes[:, 0] / image_w
            scaled_y = bboxes[:, 1] / image_h

            box_width = box_width[..., np.newaxis]
            box_height = box_height[..., np.newaxis]
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]

            spatial_features = np.concatenate(
                (scaled_x,
                 scaled_y,
                 scaled_x + scaled_width,
                 scaled_y + scaled_height,
                 scaled_width,
                 scaled_height),
                axis=1)

            if image_id in train_imgids:
                continue
                train_imgids.remove(image_id)
                train_indices[image_id] = train_counter
                train_img_bb[train_counter, :, :] = bboxes
                train_img_features[train_counter, :, :] = np.frombuffer(
                    base64.decodestring(item['features']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                train_spatial_img_features[train_counter, :, :] = spatial_features
                ## TODO : Replace with output from tsv file
                train_gold_id[train_counter, :] = np.random.randint(num_fixed_boxes)
                train_counter += 1
            elif image_id in val_imgids:
                val_imgids.remove(image_id)
                val_indices[image_id] = val_counter
                val_img_bb[val_counter, :, :] = bboxes
                val_img_features[val_counter, :, :] = np.frombuffer(
                    base64.decodestring(item['features']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                val_spatial_img_features[val_counter, :, :] = spatial_features
                val_gold_id[val_counter, :] = np.random.randint(num_fixed_boxes)
                val_counter += 1
            else:
                assert False, 'Unknown image id: %d' % image_id

    if len(train_imgids) != 0:
        print('Warning: train_image_ids is not empty')

    if len(val_imgids) != 0:
        print('Warning: val_image_ids is not empty')

    cPickle.dump(train_indices, open(train_indices_file, 'wb'))
    cPickle.dump(val_indices, open(val_indices_file, 'wb'))
    h_train.close()
    h_val.close()
    print("done!")
