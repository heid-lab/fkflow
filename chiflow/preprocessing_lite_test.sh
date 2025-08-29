#!/bin/bash

DATA_PATH="tests"
FULL_CSV="$DATA_PATH/rdb7_full.csv"
FULL_XYZ="$DATA_PATH/rdb7_full.xyz"
SAVE_DIR="$DATA_PATH/processed_data"

python preprocessing_lite.py \
    --csv_file "$FULL_CSV" \
    --xyz_file "$FULL_XYZ" \
    --save_dir "$SAVE_DIR"
