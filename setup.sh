#!/bin/bash
# Install geckodriver
wget https://github.com/mozilla/geckodriver/releases/latest/download/geckodriver-v0.30.0-linux64.tar.gz
tar -xvzf geckodriver-v0.30.0-linux64.tar.gz
rm geckodriver-v0.30.0-linux64.tar.gz
chmod +x geckodriver
mv geckodriver /usr/local/bin/

