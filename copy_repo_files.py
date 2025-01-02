import os
import shutil

# Define the source directory (cloned repo) and target directory (flat directory)
source_dir = "..\\pytorch"  # Replace with the path to the cloned repo
target_dir = "pytorch_corpus"  # Replace with the path to the flat directory

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Walk through the source directory to find all .py files
for root, _, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".py"):  # Check if the file is a Python file
            # Construct full file paths
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_dir, file)

            # Copy the file to the target directory
            shutil.copy2(source_file, target_file)

print(f"All .py files have been copied to {target_dir}.")