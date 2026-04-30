#!/bin/bash

ratios=(0 0.25 0.5 0.75 0.8 1)

for r in "${ratios[@]}"; do
    echo "Running with keep_ratio=$r"

    
    python3 eval_realworldqa.py --split test --limit 100 --visionzip_keep_ratio $r --t2v
    
    # python3 eval_mmmu.py \
    #     --split validation \
    #     --limit 500 \
    #     --visionzip_keep_ratio $r --t2v \
    #     --subjects Accounting Biology Computer_Science
    
  #   python3 eval_mathvista.py \
  # --split testmini \
  # --limit 100 \
  # --visionzip_keep_ratio $r \
  # --t2v \
  # --max_new_tokens 4
        

    echo "--------------------------------------"
done 