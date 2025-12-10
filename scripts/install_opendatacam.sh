#!/bin/bash

# OpenDataCam Installation Script
# This script installs OpenDataCam for automated CCTV traffic data extraction

set -e

echo "========================================="
echo "OpenDataCam Installation Script"
echo "========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "Docker installed successfully."
else
    echo "Docker is already installed."
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "Docker Compose installed successfully."
else
    echo "Docker Compose is already installed."
fi

# Create OpenDataCam directory
OPENDATACAM_DIR="/home/ubuntu/cctv-traffic-analysis/opendatacam"
cd $OPENDATACAM_DIR

# Download OpenDataCam installation script
echo ""
echo "Downloading OpenDataCam installation script..."
wget -N https://raw.githubusercontent.com/opendatacam/opendatacam/v3.0.2/docker/install-opendatacam.sh

# Give exec permission
chmod 777 install-opendatacam.sh

echo ""
echo "========================================="
echo "OpenDataCam installation script downloaded successfully."
echo "========================================="
echo ""
echo "To install OpenDataCam, run one of the following commands:"
echo ""
echo "For Jetson Nano:"
echo "  cd $OPENDATACAM_DIR && ./install-opendatacam.sh --platform nano"
echo ""
echo "For Jetson Xavier / Xavier NX:"
echo "  cd $OPENDATACAM_DIR && ./install-opendatacam.sh --platform xavier"
echo ""
echo "For Desktop/Server with NVIDIA GPU:"
echo "  cd $OPENDATACAM_DIR && ./install-opendatacam.sh --platform desktop"
echo ""
echo "After installation, OpenDataCam will be available at http://localhost:8080"
echo ""
