#!/usr/bin/env bash
python -u main.py \
    --train_expressions /usr2/home/aschaudh/11777/Google_Refexp_toolbox/google_refexp_dataset_release/google_refexp_train_201511.json \
    --valid_expressions /usr2/home/aschaudh/11777/Google_Refexp_toolbox/google_refexp_dataset_release/google_refexp_val_201511.json  \
    --train_question_path /usr2/home/aschaudh/11777/bottom-up-attention-vqa/data/v2_OpenEnded_mscoco_train2014_questions.json \
    --valid_question_path /usr2/home/aschaudh/11777/bottom-up-attention-vqa/data/v2_OpenEnded_mscoco_val2014_questions.json \
    --test_question_path /usr2/home/aschaudh/11777/bottom-up-attention-vqa/data/v2_OpenEnded_mscoco_test2015_questions.json \
    --train_answer_path /usr2/home/aschaudh/11777/bottom-up-attention-vqa/data/cache/train_target.pkl\
    --valid_answer_path /usr2/home/aschaudh/11777/bottom-up-attention-vqa/data/cache/val_target.pkl \
    --label2answer_path /usr2/home/aschaudh/11777/bottom-up-attention-vqa/data/cache/trainval_label2ans.pkl \
    --imageids /usr2/home/aschaudh/11777/image_ids.txt \
    --lr 0.001  2>&1 | tee debug 
