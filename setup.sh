#!/bin/bash

# Download geckodriver using curl instead of wget
curl -L -o geckodriver-v0.30.0-linux64.tar.gz "https://github.com/mozilla/geckodriver/releases/download/v0.30.0/geckodriver-v0.30.0-linux64.tar.gz"

# Extract the downloaded file
tar -xzf geckodriver-v0.30.0-linux64.tar.gz

# Make the driver executable
chmod +x geckodriver

# Move it to /usr/local/bin
mv geckodriver /usr/local/bin/
