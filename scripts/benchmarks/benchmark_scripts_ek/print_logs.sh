LOGS_PATH="logs/*.txt"

print_logs() {
    for FILE in $LOGS_PATH
    do
            FILENAME=${FILE/*\//}
            FILENAME=${FILENAME/\.*/}
            echo ${FILENAME}
            cat $FILE | grep "Startup time\|setup time\|Avg FPS over all processed"
            echo
    done
}
print_logs