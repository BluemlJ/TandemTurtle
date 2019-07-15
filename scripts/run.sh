#!/bin/bash

# PORT should be something like 8080
# POSTITION should be something like a/A/b/B or empty

POSITION=''

while getopts address:tid:gid: option; do
    case "${option}" in
        address)
            ADDRESS=${OPTARG};;
        tid)
            TID=${OPTARG};;
        gid)
            GID=${OPTARG};;
    esac
done

# start our Agent
python main.py single_agent 0 $ADDRESS $TID $GID