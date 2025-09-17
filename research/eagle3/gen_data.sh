#This file is adapted from https://github.com/HArmonizedSS/HASS (arxiv: https://arxiv.org/abs/2408.15766)
#Which is a fork of the Eagle repository: https://github.com/SafeAILab/EAGLE (arxiv: https://arxiv.org/abs/2401.15077)

# to get the dataset, run: wget https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V4.3_unfiltered_cleaned_split.json .

CUDA_VISIBLE_DEVICES=6,7 python -m ge_data.allocation \
--outdir dataDirectory/sharegpt \
--data_path /proving-grounds/cache/megan/ShareGPT_V4.3_unfiltered_cleaned_split.json \
--model_path openai/gpt-oss-20b \
--chat_template gpt-oss \
--dataset sharegpt \
--split gen \
--samples 4 \
--total_gpus 2
