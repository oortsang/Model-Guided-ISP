#!/bin/bash
# Check all the HDF5 in a given dataset directory
# Usage:
# 1. Default:
#     ./check_if_dataset_is_complete.sh dataset
#     where the data files are something like dataset/train_measurement_nu_4/measurements_0.h5
# 2. Specify the inner directories as a list or bash pattern (which gets expanded to a list)
#     ./check_if_dataset_is_complete.sh dataset train_measurements_nu_{4,8}
#     ./check_if_dataset_is_complete.sh dataset train_measurements_nu_4 train_measurements_nu_8 ...

inner_dir_args=${@:2}

dataset_base_dir=$1
if [[ -z $inner_dir_args ]] then
    inner_dir_list="${dataset_base_dir}/*"
else
    inner_dir_list=$(echo ${inner_dir_args} | sed "s|[^ ]*|${dataset_base_dir}/&|g")
fi

incomplete_file_list=""
echo "Completed data files:"

complete_counter=0
incomplete_counter=0
for inner_dir_name in ${inner_dir_list}; do
    for file in ${inner_dir_name}/*.h5; do
        completion=$(./check_if_datafile_is_complete.sh $file)
        # echo "${file}: ${completion}"
        if [[ ${completion} == "false" ]]; then
            incomplete_file_list+="- ${file}\n"
            incomplete_counter=$(($incomplete_counter + 1))
        else
            echo "- ${file}"
            complete_counter=$(($complete_counter + 1))
        fi
    done
done
printf "${complete_counter} complete files found in all\n\n"
printf "Incomplete data files:\n${incomplete_file_list}"
echo "${incomplete_counter} incomplete files found in all"
