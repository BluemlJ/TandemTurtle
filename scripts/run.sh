#!/bin/bash

# PORT should be something like 8080
# POSTITION should be something like a/A/b/B or empty

POSITION=''

while getopts port:id:position: option; do
    case "${option}" in
        port)
            PORT=${OPTARG};;
        id)
            ID=${OPTARG};;
        position)
            POSITION=${OPTARG};;
    esac
done

# start our Agent
python main.py single_agent 0 $PORT $ID $POSITION