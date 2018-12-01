python -u src/multitask/top_bottom/main.py  \
    --data_root /usr3/data/aschaudh/11777/baseline_data/11777/bottom-up-attention-vqa/data \
    --task vqa \
    --mode ref2vqa \
    --model_file /usr3/data/aschaudh/11777/code/MultitaskMultimodal/saved_models/ref/model_refexp_best.pth \
    --epochs 10 \
    --learning_rate 0.000001 \
    --output /usr3/data/aschaudh/11777/code/MultitaskMultimodal/saved_models/ref2vqa_lr_1e-6 \
    --dictionary /usr3/data/aschaudh/11777/baseline_data/11777/bottom-up-attention-vqa/data/dictionary_common.pkl 2>&1 | tee logs/ref2vqa.log 
