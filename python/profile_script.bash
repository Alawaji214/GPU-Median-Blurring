#!/bin/bash

# Get the script name without the .py extension
script_name=$(basename "$1" .py)

# Run nsys profiling with the script name as the output file name
nsys profile --stats=true --output /home/bandr1994/COE506-Project/python/"$script_name" python "$@"