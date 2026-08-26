#!/bin/bash

# Define the base directory
base_dir="${SEARCH_PATH}"
replace_to="${REPLACE_TO}"

# Replace *** in every CSV file
find "$base_dir" -type f -name "*.csv" \
  -exec sed -i "s|\*\*\*|${replace_to}|g" {} +

echo "Replacement complete."