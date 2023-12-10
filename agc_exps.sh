#!/bin/bash

# # Get the desired compression gain
# read -p "Enter the desired compression gain (in dB): " compression_gain
mkdir -p agc_exp_csvs_new
# Replace the existing value in main.c using sed with capture groups
for compression_gain in `seq 7 15`; do
    for targetLevelDbfs in `seq 3 9`; do
        sed -i -E "s/agcConfig.compressionGaindB = ([0-9]+\.?[0-9]*);/agcConfig.compressionGaindB = $compression_gain;/g" main.c
        sed -i -E "s/agcConfig.targetLevelDbfs = ([0-9]+\.?[0-9]*);/agcConfig.targetLevelDbfs = $targetLevelDbfs;/g" main.c
        # Build the project
        cd build
        make
        cd ..
        rm blind_data_16k/*
        rm blind_data_16k_agc/*
        rm blind_data_agc/*
        python batch_process_48k.py
        python calc_folder_sigmos.py blind_data_agc agc_exp_csvs_new/gain${compression_gain}_target${targetLevelDbfs}.csv
    done
done
# # Check if the replacement was successful
# replaced_value=$(grep -E "agcConfig.compressionGaindB = ([0-9]+\.?[0-9]*);" main.c)

# if [[ -z "$replaced_value" ]]; then
#   echo "Error: Could not find and replace the agcConfig.compressionGaindB variable."
#   exit 1
# fi

# # Extract the replaced value for confirmation
# replaced_gain=$(echo "$replaced_value" | grep -Eo "[0-9]+\.?[0-9]*")

# if [[ "$replaced_gain" == "$compression_gain" ]]; then
#   echo "Compression gain successfully updated to $compression_gain dB."
# else
#   echo "Error: Unexpected value found during replacement. Please check the output."
# fi
