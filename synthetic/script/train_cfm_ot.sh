#!/bin/bash

#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --job-name=ot_train
#SBATCH --output=%x-%j.out

cd /home/konstantin.mark/projects/flow_guidance/synthetic || exit

for pair in "circle s_curve" "8gaussian moon" "uniform 8gaussian"; do
    set -- $pair
    x0_dist=$1
    x1_dist=$2
    python guided_flow/train/cfm.py --x0_dist $x0_dist --x1_dist $x1_dist --cfm 'ot_cfm'
    python guided_flow/train/value_guidance_matching.py --x0_dist $x0_dist --x1_dist $x1_dist
done