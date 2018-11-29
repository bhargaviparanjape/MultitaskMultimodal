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
#infile = os.path.join(data_root, 'full_2014.tsv')
infile = "/usr3/data/aschaudh/11777/baseline_data/11777/bottom-up-attention-vqa/data/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv"
train_indices_file = os.path.join(data_root, 'ref_train36_imgid2idx.pkl')
val_indices_file = os.path.join(data_root, 'ref_val36_imgid2idx.pkl')
train_ids_file = os.path.join(data_root, 'ref_train_ids.pkl')
val_ids_file = os.path.join(data_root, 'ref_val_ids.pkl')

feature_length = 2048
num_fixed_boxes = 36


if __name__ == '__main__':


    if os.path.exists(train_ids_file) and os.path.exists(val_ids_file):
        train_imgids = cPickle.load(open(train_ids_file))
        val_imgids = cPickle.load(open(val_ids_file))
    else:
        train_imgids = utils.load_imageid(os.path.join(data_root, 'google_refexp_train_201511_coco_aligned_and_labeled_filtered.json'))
        val_imgids = utils.load_imageid(os.path.join(data_root, 'google_refexp_val_201511_coco_aligned_and_labeled_filtered.json'))

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

    train_counter = 0
    val_counter = 0

    print("reading tsv...")
    with open(infile, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:

            item['num_boxes'] = int(item['num_boxes'])
            image_id = int(item['image_id'])

            if image_id in tsv_train_img_ids:
                tsv_train_img_ids.remove(image_id)
                train_indices[image_id] = train_counter

                train_counter += 1
            elif image_id in tsv_val_img_ids:
                tsv_val_img_ids.remove(image_id)
                val_indices[image_id] = val_counter

                val_counter += 1
            else:
                print('Unknown image id: %d' % image_id)

    if len(tsv_train_img_ids) != 0:
        print('Warning: train_image_ids is not empty')

    if len(tsv_val_img_ids) != 0:
        print('Warning: val_image_ids is not empty')

    cPickle.dump(train_indices, open(train_indices_file, 'wb'))
    cPickle.dump(val_indices, open(val_indices_file, 'wb'))

    print("done!")
