python -u src/multitask/top_bottom/main.py  \
    --data_root /usr3/data/aschaudh/11777/baseline_data/11777/bottom-up-attention-vqa/data \
    --task vqa \
    --epochs 10 \
    --run_as low_resource \
    --output /usr3/data/aschaudh/11777/code/MultitaskMultimodal/saved_models/vqa_lowres \
    --dictionary /usr3/data/aschaudh/11777/baseline_data/11777/bottom-up-attention-vqa/data/dictionary_common.pkl \
    --batch_size 256 2>&1 | tee logs/vqa_lowresource.log
