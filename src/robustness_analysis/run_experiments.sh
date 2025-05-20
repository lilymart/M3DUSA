#!/bin/bash

dataset_name="mumin" #politifact
num_layers=3 #2

#modes=("only_net_edges" "only_net_edges_mps" "only_net_edges_meta" "only_net_edges_mps_meta" "EF_all" "EF_256")
#mode="EF_256" #di default per load_politifact_heterodata
drop_percentages=(0 2 4 8 16 24)
drop_methods_names=("masking_unimodal_text" "masking_unimodal_net" "masking_multimodal" "masking_multimodal_mixed")

training_seeds=(42 123 12345 123123 2025)
#num_seeds=${#training_seeds[@]}

base_dir='/data/robustness_analysis'
split='60_15_25'
task="_${split}_${num_layers}layers"
embeddings_dir="$base_dir/embeddings$task"
models_dir="$base_dir/best_models$task"
results_dir="$base_dir/results$task"
losses_dir="$base_dir/losses$task"

for dir in "$embeddings_dir" "$models_dir" "$results_dir" "$losses_dir"; do
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    fi
done

export PYTHONPATH=$PYTHONPATH:XXXXX

for drop_perc in "${drop_percentages[@]}"
do
   for drop_method_name in "${drop_methods_names[@]}"
   do
      for seed_index in "${!training_seeds[@]}"  # Iterate over seed indices
      do
          seed="${training_seeds[$seed_index]}"  # Get actual seed value
          echo "### Dropping $drop_perc% according to method $drop_method_name for seed: $seed ###"
          python src/robustness_analysis/EF_main_ES.py "$dataset_name" "$split" "$drop_perc" "$drop_method_name" "$seed_index" "$seed" "$embeddings_dir" "$models_dir" "$results_dir" "$losses_dir"
      done
    done
done