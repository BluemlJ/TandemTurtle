#!/bin/bash

# PORT should be something like 8080
# POSTITION should be something like a/A/b/B or empty

POSITION=''

while getopts p:s: option; do
    case "${option}" in
        port)
            echo "You set flag -p"
            PORT=${OPTARG};;
        position)
            echo "You set flag -b"
            POSITION=${OPTARG};;
    esac
done

# start our Agent
python main.py single_agent 0 $PORT $POSITION