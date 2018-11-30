python -u src/multitask/top_bottom/main.py  \
    --data_root /usr3/data/aschaudh/11777/baseline_data/11777/bottom-up-attention-vqa/data \
    --task ref_vqa \
    --output /usr3/data/aschaudh/11777/code/MultitaskMultimodal/saved_models/multiplex \
    --dictionary /usr3/data/aschaudh/11777/baseline_data/11777/bottom-up-attention-vqa/data/dictionary_common.pkl \
    --mt_mode multiplex 2>&1 | tee multitask_multiplex.log 
