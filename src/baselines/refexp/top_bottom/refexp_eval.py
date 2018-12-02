import argparse
import scipy
import numpy as np
import sys
import os
import json
import tempfile

# MAKE Sure that google_refexp_py_lib is in your python libary search path
# before you run API in this toolbox. You can use something as follows:

base_dir = '/home/ubuntu/11777/src/baselines/refexp/Google_Refexp_toolbox'
sys.path.append('%s/google_refexp_py_lib' % base_dir)
from refexp_eval import RefexpEvalComprehension
from refexp_eval import RefexpEvalGeneration
from common_utils import draw_bbox


def evaluate(predictions_file):
    # Set coco_data_path and Google Refexp dataset validation set path
    coco_data_path = '%s/external/coco/annotations/instances_train2014.json' % base_dir
    refexp_dataset_path = '%s/google_refexp_dataset_release/google_refexp_val_201511_coco_aligned.json' % base_dir
    eval_compreh = RefexpEvalComprehension(refexp_dataset_path, coco_data_path)
    data_path = '/home/ubuntu/11777/data/refexp/'

    with open('%s/google_refexp_val_201511_coco_aligned_and_labeled.json' % data_path, 'r') as fp:
        labeled_data = json.load(fp)

    predictions = []
    with open('%s' % predictions_file, 'r') as fp:
        for line in fp:
            pred = json.loads(line)
            ann = labeled_data['annotations'][str(pred['annotation_id'])]
            b = ann['boxes']
            l = pred['logits']
            predictions.append({
                'annotation_id': pred['annotation_id'],
                'refexp_id': pred['refexp_id'],
    #             'predicted_bounding_boxes': [ann['boxes'][pred['predicted_id']]]
                'predicted_bounding_boxes': zip(*sorted(zip(*[l, b]), reverse=True))[1]
            })

    fd, eval_path = tempfile.mkstemp()
    with os.fdopen(fd, 'w') as tmp:
        json.dump(predictions, tmp)

    eval_compreh.evaluate(eval_path, thresh_k=1)
    eval_compreh.evaluate(eval_path, thresh_k=2)
    eval_compreh.evaluate(eval_path, thresh_k=3)

    os.remove(eval_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction_file', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(args.predictions_file)
