#!/bin/bash
seeds=(-8358938852454912011)

num_runs=5

base_command="python evolution_strategy.py --num_parents 20 --level 1 --self_adaptive --selection_type \"all\" --optimization \"max\" --generation 50 --num_offsprings 20 --learning_factor 10 --agent agents/level1_deprecated.py --opponent starterAIs/SS_Starter.py --opponent_weight_file \"\" --verbose"

# 遍历所有 seeds
for seed in "${seeds[@]}"
do
  for i in $(seq 1 $num_runs)
  do
    run_command="$base_command --seed $seed --run_name \"level1_${seed}_${i}\""
    eval $run_command
  done
done