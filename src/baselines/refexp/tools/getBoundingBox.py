#!/usr/bin/env python

# --------------------------------------------------------
# R-FCN
# Copyright (c) 2016 Yuwen Xiong
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg,cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import json,codecs

classes = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'ResNet-50': ('ResNet-50',
                  'resnet50_rfcn_final.caffemodel'),
	'ResNet-101':('ResNet-101',
		    'resnet101_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name,fout):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
    image_name = (args.prefix + "_{0}.jpg").format(image_name.zfill(12))
    
    #import pdb; pdb.set_trace()
    im_file = os.path.join(args.data_dir, image_name)
    im = plt.imread(im_file) # orignally cv2.imread()

    # Detect all object classes and regress object bounds
    timer = Timer()

    # Visualize detections for each class
    CONF_THRESH = 0.4
    conf_thresh = 0.4
    min_boxes=10
    max_boxes=20
    NMS_THRESH = 0.3
    
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regression bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    print("Class shapes: {0}".format(cls_prob.shape[1]))
    #attr_prob = net.blobs['attr_prob'].data
    #pool5 = net.blobs['pool5_flat'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1,cls_prob.shape[1]):
	cls_scores = scores[:, cls_ind]
	dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
	keep = np.array(nms(dets, cfg.TEST.NMS))
	max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    #Sorting by scores
    keep_boxes = np.argsort(max_conf)[::-1]

    if len(keep_boxes) < min_boxes:
	keep_boxes = keep_boxes[:min_boxes]
    elif len(keep_boxes) > max_boxes:
	keep_boxes = keep_boxes[:max_boxes]

############################

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if args.display:
	plt.imshow(im)

    boxes = cls_boxes[keep_boxes]
    scores = cls_scores[keep_boxes]
    assert len(boxes) == len(scores)

    objects = np.argmax(cls_prob[keep_boxes][:,1:], axis=1)
    attr_thresh = 0.1
    #attr = np.argmax(attr_prob[keep_boxes][:,1:], axis=1)
    #attr_conf = np.max(attr_prob[keep_boxes][:,1:], axis=1)
    output_obj = {'boxes':boxes.tolist(),'scores':scores.tolist()}

    if args.display:
	for i in range(len(keep_boxes)):
	    bbox = boxes[i]
	    if bbox[0] == 0:
		bbox[0] = 1
	    if bbox[1] == 0:
		bbox[1] = 1
	    cls = classes[objects[i]+1]
	    #if attr_conf[i] > attr_thresh:
	    #    cls = attributes[attr[i]+1] + " " + cls
	    plt.gca().add_patch(
		plt.Rectangle((bbox[0], bbox[1]),
                      bbox[2] - bbox[0],
                      bbox[3] - bbox[1], fill=False,
                      edgecolor='red', linewidth=2, alpha=0.5))
	    #plt.gca().text(bbox[0], bbox[1] - 2,'%s' % (""),
	    #        bbox=dict(facecolor='blue', alpha=0.5),
	    #        fontsize=10, color='white')
    
	plt.savefig(im_file.replace(".jpg","_demonew.jpg"))
    
    print 'boxes=%d' % (len(keep_boxes))
    fout.write(json.dumps(output_obj) + "\n")

    '''
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
    '''
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ResNet-101]',
                        choices=NETS.keys(), default='ResNet-101')

    parser.add_argument('--imageIds',type=str, help='List of image ids',
			default='/usr2/home/aschaudh/11777/image_ids.txt')
    
    parser.add_argument('--data_dir',type=str, default='/usr2/home/aschaudh/11777/bottom-up-attention-vqa/data/train2014/')
    
    parser.add_argument('--prefix',type=str, default='COCO_train2014')

    parser.add_argument('--output',type=str, default='/usr2/home/aschaudh/11777/bottom-up-attention-vqa/data/boundingBoxes/')
    
    parser.add_argument('--display',default=False, action="store_true")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    #prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
    #                        'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')

    prototxt = 'models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    weights = 'data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel'
    net = caffe.Net(prototxt, weights , caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    #for i in xrange(2):
    #    _, _= im_detect(net, im)

    #im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #            '001763.jpg', '004545.jpg']

    with open(args.imageIds) as fin:
	im_names = fin.readlines()
	im_names = [im_name.strip() for im_name in im_names]

    
    fout = codecs.open(args.output + args.prefix + ".box.jsonl","w", encoding='utf-8')
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name,fout)

    #plt.show()
