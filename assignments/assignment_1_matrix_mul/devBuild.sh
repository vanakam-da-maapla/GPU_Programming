#!/bin/bash

# Define an array of C source files
# C_FILES=("main_1_1.cu" "main_1_2.cu" "main_2.cu" "main_3.cu" "main_4.cu")
# EXECUTABLES=("matmul_1d", "matmul_2d", "matmul_tiled", "mattrans_basic", "mattrans_tiled")

 C_FILES=("main_1_1.cu" "main_1_2.cu" "main_2.cu" "main_3.cu" "main_4.cu")
 EXECUTABLES=("matmul_1d" "matmul_2d" "matmul_tiled" "mattrans_basic" "mattrans_tiled")
 CUDA_EXCEPTION_HANDLER = "cudaExceptionHandler.cu"
 FILE_OPERATIONS = "fileoperations.cpp"

 C_FILES=("{$C_FILES[0]}")
 EXECUTABLES=("{$EXECUTABLES[0]}")

# Check if the number of files and executable names match
if [[ "${#C_FILES[@]}" -ne "${#EXECUTABLES[@]}" ]]; then
  echo "Error: The number of source files and executable names do not match."
  exit 1
fi

# Loop through the arrays using an index
for ((i=0; i<${#C_FILES[@]}; i++)); do
  c_file="${C_FILES[i]}"
  executable="${EXECUTABLES[i]}"
  
  echo "Compiling $c_file to $executable..."
  
  # Use nvcc to compile, specifying the output executable name with the -o flag
  nvcc "$c_file" "$CUDA_EXCEPTION_HANDLER" "$FILE_OPERATIONS"  -o "$executable"
  
  # Check for compilation errors
  if [ $? -eq 0 ]; then
    echo "Compilation successful. Executable: $executable"
    "./$executable"
    echo "" # Add a blank line for readability
  else
    echo "Compilation of $c_file failed. Skipping..."
    echo "" # Add a blank line for readability
  fi
done

echo "Script finished."