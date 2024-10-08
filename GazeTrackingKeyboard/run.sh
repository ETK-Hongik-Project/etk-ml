#!/bin/bash

# Function to display the help message
usage() {
    echo "Usage: $0 -e <env_name> -t <tar_file> -d <dataset_path> -o <output_path> -m <model_path>"
    echo
    echo "Options:"
    echo "  -e <env_name>        Name of the conda environment to activate."
    echo "  -t <tar_file>        Path to the initial tar file to extract."
    echo "  -d <dataset_path>    Path to the original dataset."
    echo "  -o <output_path>     Path to the output processed dataset."
    echo "  -m <model_path>      Path to save the trained model."
    exit 1
}

# Check for the correct number of arguments
if [ "$#" -lt 10 ]; then
    usage
fi

# Parse command-line arguments
while getopts "e:t:d:o:m:" opt; do
  case $opt in
    e)
      env_name=$OPTARG
      ;;
    t)
      tar_file=$OPTARG
      ;;
    d)
      dataset_path=$OPTARG
      ;;
    o)
      output_path=$OPTARG
      ;;
    m)
      model_path=$OPTARG
      ;;
    *)
      usage
      ;;
  esac
done

# Validate that all required arguments are provided
if [[ -z "$env_name" || -z "$tar_file" || -z "$dataset_path" || -z "$output_path" || -z "$model_path" ]]; then
    echo "Error: All arguments -e, -t, -d, -o, and -m must be provided."
    usage
fi

# Step 1: Activate the specified conda environment
echo "Activating conda environment: $env_name"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$env_name"

# Step 2: Create the GazeCapture directory
echo "Creating GazeCapture directory..."
mkdir -p GazeCapture

# Step 3: Extract the initial tar file into the GazeCapture directory
echo "Extracting $tar_file into GazeCapture directory..."
tar -xf "$tar_file" -C ./GazeCapture

# Step 4: Change directory to GazeCapture
echo "Changing directory to GazeCapture..."
cd GazeCapture || { echo "Failed to change directory to GazeCapture"; exit 1; }

# Step 5: Extract all .tar.gz files inside GazeCapture directory
echo "Extracting all .tar.gz files in GazeCapture..."
ls | grep ".tar.gz$" | xargs -I {} tar -xf {}

# Step 6: Remove the .tar.gz files after extraction
echo "Removing .tar.gz files after extraction..."
ls | grep ".tar.gz$" | xargs -I {} rm {}

# Step 7: Prepare the dataset using prepareDataset.py
echo "Preparing dataset using prepareDataset.py..."
python prepareDataset.py --dataset_path "$dataset_path" --output_path "$output_path"

# Step 8: Train the model using train_gtk.py
echo "Training model using train_gtk.py..."
python train_gtk.py --data-path "$output_path" --model-path "$model_path"

echo "All operations completed successfully."