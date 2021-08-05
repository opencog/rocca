# set -x

python_files=$(find . -name "*.py")
# echo "[INFO] Static check the following files: ${python_files}"
mypy ${python_files} | grep -v "error: Skipping analyzing"
