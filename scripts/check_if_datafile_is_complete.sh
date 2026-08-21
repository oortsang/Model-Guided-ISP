#!/bin/bash
# Check whether the datafile pointed to is complete or not
# Note that this only checks for the "file_completion" field and therefore may miss complete files
# Usage: ./scripts/check_if_datafile_is_complete.sh file.h5

data_file_name=$1
file_completion_h5_output=$(h5dump -w 1 -d "file_completion" $data_file_name)

# match on the file_completion field being marked true or not
file_complete=$(echo $file_completion_h5_output | grep "(0): TRUE"  | wc -l)

if [ $file_complete -eq 1 ]; then
    echo "true"
else
    echo "false"
fi
