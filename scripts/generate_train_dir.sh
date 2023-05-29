#!/bin/bash

source_path="/path/to/source/folder"
check_path="/path/to/check"

if [ -d "$check_path" ]; then
    echo "Check path exists. Copying folder..."
    cp -r "$source_path" "/path/to/destination/folder"
    echo "Folder copied successfully!"
else
    echo "Check path does not exist."
fi