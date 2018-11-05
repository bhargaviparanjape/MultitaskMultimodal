import argparse
import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import json

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filename', type=str, default='../../../../../data/refexp/google_refexp_val_201511_coco_aligned.json')
    parser.add_argument('--boxes_filename', type=str, default='../../../../../data/boxes/trainval_subset.tsv')
    parser.add_argument('--output_filename', type=str, default=None)
    args = parser.parse_args()
    return args

def area_bbox(bbox):
    """Return the area of a bounding box."""
    if bbox[2] <= 0 or bbox[3] <= 0:
        return 0.0
    return float(bbox[2]) * float(bbox[3])

def iou_bboxes(bbox1, bbox2):
    """Standard intersection over Union ratio between two bounding boxes."""
    bbox_ov_x = max(bbox1[0], bbox2[0])
    bbox_ov_y = max(bbox1[1], bbox2[1])
    bbox_ov_w = min(bbox1[0] + bbox1[2] - 1, bbox2[0] + bbox2[2] - 1) - bbox_ov_x + 1
    bbox_ov_h = min(bbox1[1] + bbox1[3] - 1, bbox2[1] + bbox2[3] - 1) - bbox_ov_y + 1

    area1 = area_bbox(bbox1)
    area2 = area_bbox(bbox2)
    area_o = area_bbox([bbox_ov_x, bbox_ov_y, bbox_ov_w, bbox_ov_h])
    area_u = area1 + area2 - area_o
    if area_u < 0.000001:
        return 0.0
    else:
        return area_o / area_u

def load_boxes(boxes_filename):
    csv.field_size_limit(sys.maxsize)
    in_data = {}
    with open(boxes_filename, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])   
            item['num_boxes'] = int(item['num_boxes'])
            b = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape((item['num_boxes'],-1))
            b = np.array(b)
            b[:, 2] -= b[:, 0]
            b[:, 3] -= b[:, 1]
            item['boxes'] = b
            in_data[item['image_id']] = item
    return in_data

def max_vec(scores):
    labels = np.zeros(len(scores))
    labels[scores.index(max(scores))] = 1
    return labels

def get_labeled_data(input_filename, boxes):

    with open(input_filename, 'r') as fp:
        data_aligned = json.load(fp)

    labeled_data = {}
    labeled_data['images'] = data_aligned['images']
    labeled_data['refexps'] = data_aligned['refexps']
    labeled_data['annotations'] = {}

    for annotation_id, annotation in data_aligned['annotations'].items():
        img_id = annotation['image_id']
        if int(img_id) in boxes:
            new_annotation = dict(annotation)
            scores = [iou_bboxes(b, annotation['bbox']) for b in boxes[int(img_id)]['boxes']]
            labels = max_vec(scores)
            new_annotation['boxes'] = [list(map(float, b)) for b in boxes[int(img_id)]['boxes']]
            new_annotation['iou_scores'] = scores
            new_annotation['labels'] = list(labels)
            labeled_data['annotations'][annotation_id] = new_annotation
            del new_annotation['segmentation']

    print('Number of images in boxes_dataset = %d' % len(boxes))
    print('Number of images in input_dataset = %d' % len([int(i['image_id']) for i in data_aligned['annotations'].values()]))
    print('Number of input annotations = %d' % len(data_aligned['annotations']))
    print('Number of images labeled = %d' % len([int(i['image_id']) for i in labeled_data['annotations'].values()]))
    print('Number of annotations labeled = %d' % len(labeled_data['annotations']))

    return labeled_data

if __name__ == '__main__':
    args = parse_args()
    input_filename = args.input_filename
    output_filename = args.output_filename
    boxes_filename = args.boxes_filename

    if not output_filename:
        output_filename = input_filename.replace('.json', '_and_labeled.json')
    
    boxes = load_boxes(boxes_filename)
    labeled_data = get_labeled_data(input_filename, boxes)

    # Save labeled data
    with open(output_filename, 'w') as fp:
        json.dump(labeled_data, fp)
