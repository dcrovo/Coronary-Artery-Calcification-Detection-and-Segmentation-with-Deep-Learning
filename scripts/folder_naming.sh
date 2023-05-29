#!/bin/bash

#Define the folder path 
directory="/home/danielcrovo/Documents/01.Study/01.MSc/02.MSc AI/Deep Learning/Proyecto/data_mask"

# Iterate through each folder in the directory
for folder in "$directory"/*; do
  # Check if it is a directory
  if [ -d "$folder" ]; then
    # Remove all letters from the folder name
    last_part=$(basename "$folder")

    new_folder=$(echo "$last_part" | tr -d [:alpha:])
    new_folder=$(echo "$new_folder" | tr -d ' ')
    new_dir="$directory/$new_folder"
    # Rename the folder
    mv "$folder" "$new_dir"
    
    echo "Renamed $folder to $new_dir"
  fi
done
