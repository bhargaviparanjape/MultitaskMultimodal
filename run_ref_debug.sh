python -u src/multitask/top_bottom/main.py  \
    --data_root /usr3/data/aschaudh/11777/baseline_data/11777/bottom-up-attention-vqa/debug \
    --task ref \
    --output /usr3/data/aschaudh/11777/code/MultitaskMultimodal/saved_models/ref-debug \
    --run_as debug \
    --batch_size 2 \
    --dictionary /usr3/data/aschaudh/11777/baseline_data/11777/bottom-up-attention-vqa/data/dictionary_common.pkl 2>&1 | tee ref.log
