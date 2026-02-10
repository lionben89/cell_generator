#!/bin/bash

# Define the base directory
base_dir="${DATA_MODELS_PATH:-/mnt/new_groups/assafza_group/assafza/lion_models_clean}/train_test_list"

# Use find to locate all *.csv files and sed to replace the text
find "$base_dir" -type f -name "*.csv" -exec sed -i 's|***|'"$DATA_MODELS_PATH"/train_test_list'|g' {} +

echo "Replacement complete."
