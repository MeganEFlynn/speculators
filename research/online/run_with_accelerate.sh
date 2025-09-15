export PYTHONPATH=".:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --multi_gpu --num_processes 6 --mixed_precision bf16 train/main_train.py \
    --basepath /proving-grounds/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659  \
    --configpath train/llama3_8_B.json \
    --epoch 4 \
    --cpdir checkpoints \
    --data_num 8000 \
    --bs 1 \
    --gradient-accumulation-steps 4 \
    --lr 8e-5 \
    --forward_num_total 3 \
    --num_data_workers 4