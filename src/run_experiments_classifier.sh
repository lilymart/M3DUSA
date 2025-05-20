#!/bin/bash

dataset_name="mumin" #politifact
num_layers=3 #2

models=("only_text_news" "only_net_edges_cl" "only_net_edges_mps_cl" "only_net_edges_meta_cl" "only_net_edges_mps_meta_cl" "EF_256_cl" "EF_all_cl" "LF_concat" "LF_avg_pool" "LF_max_pool" "LF_weighted" "LF_attention" "LF_gated" "LF_bilinear")

training_seeds=(42 123 12345 123123 2025)
#num_seeds=${#training_seeds[@]}

base_dir='/data'
split='60_15_25'
task="_${split}_${num_layers}layers"
embeddings_dir="$base_dir/embeddings$task"
models_dir="$base_dir/best_models$task"
results_dir="$base_dir/results$task"
losses_dir="$base_dir/losses$task"


export PYTHONPATH=$PYTHONPATH:XXXXX

for model in "${models[@]}"
do
    for seed_index in "${!training_seeds[@]}"  # Iterate over seed indices
    do
        seed="${training_seeds[$seed_index]}"  # Get actual seed value
        echo "##### Running experiment on dataset $dataset_name with split $split for model: $model with seed: $seed #####"
        python src/main_late_fusion.py "$dataset_name" "$split" "$model" "$seed_index" "$seed" "$embeddings_dir" "$models_dir" "$results_dir" "$losses_dir"
        echo "-----------------------------------------"
    done
done