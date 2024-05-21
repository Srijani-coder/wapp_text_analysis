#!/bin/bash

# Ensure wget and tar are installed
sudo apt-get update
sudo apt-get install -y wget tar

# Download and install geckodriver
wget https://github.com/mozilla/geckodriver/releases/download/v0.30.0/geckodriver-v0.30.0-linux64.tar.gz
tar -xvzf geckodriver-v0.30.0-linux64.tar.gz
chmod +x geckodriver
sudo mv geckodriver /usr/local/bin/
rm geckodriver-v0.30.0-linux64.tar.gz
