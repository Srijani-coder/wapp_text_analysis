#!/bin/bash

# Check if geckodriver already exists
if [ -x "/usr/local/bin/geckodriver" ]; then
  echo "geckodriver already installed"
  exit 0
fi

# Download geckodriver using Python
python3 <<EOF
import urllib.request
import tarfile
import os

url = "https://github.com/mozilla/geckodriver/releases/download/v0.30.0/geckodriver-v0.30.0-linux64.tar.gz"
file_path = "geckodriver-v0.30.0-linux64.tar.gz"
download_path = "/usr/local/bin/geckodriver"

try:
    print("Downloading geckodriver...")
    urllib.request.urlretrieve(url, file_path)

    print("Extracting geckodriver...")
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall()

    os.chmod("geckodriver", 0o755)
    os.rename("geckodriver", download_path)
    os.remove(file_path)
    print("geckodriver installed successfully.")
except Exception as e:
    print(f"An error occurred: {e}")
    exit(1)
EOF

if [ $? -ne 0 ]; then
  echo "Failed to install geckodriver"
  exit 1
fi

echo "geckodriver installation script completed successfully."

