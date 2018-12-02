python -u src/multitask/top_bottom/main.py  \
    --data_root /usr3/data/aschaudh/11777/baseline_data/11777/bottom-up-attention-vqa/debug \
    --task ref \
    --mode vqa2ref \
    --model_file /usr3/data/aschaudh/11777/code/MultitaskMultimodal/saved_models/vqa/model_vqa.pth \
    --learning_rate 0.0001 \
    --run_as debug \
    --epochs 5 \
    --output /usr3/data/aschaudh/11777/code/MultitaskMultimodal/saved_models/vqa2ref-debug \
    --dictionary /usr3/data/aschaudh/11777/baseline_data/11777/bottom-up-attention-vqa/data/dictionary_common.pkl 2>&1 | tee vqa2ref.log 
