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
infile = os.path.join(data_root, 'full_2014.tsv')
train_data_file = os.path.join(data_root, 'train36.hdf5')
val_data_file = os.path.join(data_root, 'val36.hdf5')
train_indices_file = os.path.join(data_root, 'train36_imgid2idx.pkl')
val_indices_file = os.path.join(data_root, 'val36_imgid2idx.pkl')
train_ids_file = os.path.join(data_root, 'train_ids.pkl')
val_ids_file = os.path.join(data_root, 'val_ids.pkl')

feature_length = 2048
num_fixed_boxes = 36


if __name__ == '__main__':
    h_train = h5py.File(train_data_file, "w")
    h_val = h5py.File(val_data_file, "w")

    if os.path.exists(train_ids_file) and os.path.exists(val_ids_file):
        train_imgids = cPickle.load(open(train_ids_file))
        val_imgids = cPickle.load(open(val_ids_file))
    else:
        #train_imgids = utils.load_imageid(os.path.join(data_root, 'google_refexp_train_201511_coco_aligned_and_labeled_filtered.json'))
        #val_imgids = utils.load_imageid(os.path.join(data_root, 'google_refexp_val_201511_coco_aligned_and_labeled_filtered.json'))
        #vqa_train_imgids = [458752, 458752, 458752, 458752, 262146]
        #vqa_val_imgids = [262148, 262148, 262148, 393225, 393225]
        train_imgids = [287140, 370252, 19399, 581605, 452892]
        val_imgids = [114786, 283431, 499274, 569987, 190805]

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
            if int(item['image_id']) in train_imgids:
                images_in_train_tsv_count += 1
                tsv_train_img_ids.append(int(item['image_id']))
            elif int(item['image_id']) in val_imgids:
                images_in_val_tsv_count += 1
                tsv_val_img_ids.append(int(item['image_id']))
    train_img_features = h_train.create_dataset(
        'image_features', (images_in_train_tsv_count, num_fixed_boxes, feature_length), 'f')
    train_img_bb = h_train.create_dataset(
        'image_bb', (images_in_train_tsv_count, num_fixed_boxes, 4), 'f')
    train_spatial_img_features = h_train.create_dataset(
        'spatial_features', (images_in_train_tsv_count, num_fixed_boxes, 6), 'f')

    val_img_bb = h_val.create_dataset(
        'image_bb', (images_in_val_tsv_count, num_fixed_boxes, 4), 'f')
    val_img_features = h_val.create_dataset(
        'image_features', (images_in_val_tsv_count, num_fixed_boxes, feature_length), 'f')
    val_spatial_img_features = h_val.create_dataset(
        'spatial_features', (images_in_val_tsv_count, num_fixed_boxes, 6), 'f')

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
            if image_id in tsv_train_img_ids:
                tsv_train_img_ids.remove(image_id)
                train_indices[image_id] = train_counter
                train_img_bb[train_counter, :, :] = bboxes
                train_img_features[train_counter, :, :] = np.frombuffer(
                    base64.decodestring(item['features']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                train_spatial_img_features[train_counter, :, :] = spatial_features
                train_counter += 1
            elif image_id in tsv_val_img_ids:
                tsv_val_img_ids.remove(image_id)
                val_indices[image_id] = val_counter
                val_img_bb[val_counter, :, :] = bboxes
                val_img_features[val_counter, :, :] = np.frombuffer(
                    base64.decodestring(item['features']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                val_spatial_img_features[val_counter, :, :] = spatial_features
                val_counter += 1
            else:
                print('Unknown image id: %d' % image_id)

    if len(tsv_train_img_ids) != 0:
        print('Warning: train_image_ids is not empty')

    if len(tsv_val_img_ids) != 0:
        print('Warning: val_image_ids is not empty')

    cPickle.dump(train_indices, open(train_indices_file, 'wb'))
    cPickle.dump(val_indices, open(val_indices_file, 'wb'))
    h_train.close()
    h_val.close()
    print("done!")
