#!/bin/bash

# Define the CUDA program
program="image_cuda_median"

# Loop over the 5 noisy images
for i in {1..5}
do
  # Define the input and output image names
  input_image="noisy_image${i}.jpg"

  echo "-------------------------"
  echo "Processing ${input_image}"

  # Run nsys profile
  /opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin/nsys profile --stats=true -o profile${i} ./${program} ${input_image}  2>&1 

  echo "-------------------------"
done