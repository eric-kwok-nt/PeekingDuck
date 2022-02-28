CONFIGS_PATH="configs/*.yml"
# CONFIGS_PATH="configs/movenet_singlepose_thunder_single.yml"

# Create logs folder if it doesn't exist
if [[ ! -d logs ]]
then
    mkdir logs
    echo "Logs folder created!"
fi

benchmarking() {
    # Loop through all the config files as stated in CONFIGS_PATH
    for FILE in $CONFIGS_PATH
    do 
        FILENAME=${FILE/*\//}
        FILENAME=${FILENAME/\.*/}
        echo "Running config file "${FILENAME}".yml"
        python ../PeekingDuck --log_level debug --config_path $FILE \
        | tee -a "logs/"${FILENAME}".txt" \
        | grep "Startup time\|setup time\|Avg FPS over all processed"
        echo 
    done
}
benchmarking