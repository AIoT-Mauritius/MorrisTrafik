#!/bin/bash

###############################################################################
# OpenDataCam Live Stream Configuration Script
# This script configures OpenDataCam to process live HLS streams from my.t
###############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}OpenDataCam Live Stream Configuration${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if OpenDataCam is installed
if [ ! -d "/var/local/opendatacam" ] && [ ! -d "$HOME/opendatacam" ]; then
    echo -e "${RED}✗ OpenDataCam not found${NC}"
    echo "Please run install_opendatacam.sh first"
    exit 1
fi

# Find OpenDataCam directory
if [ -d "/var/local/opendatacam" ]; then
    ODC_DIR="/var/local/opendatacam"
elif [ -d "$HOME/opendatacam" ]; then
    ODC_DIR="$HOME/opendatacam"
else
    echo -e "${RED}✗ Could not locate OpenDataCam directory${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found OpenDataCam at: $ODC_DIR${NC}"
echo ""

# Prompt for stream selection
echo "Available streams:"
echo "  1) Caudan North"
echo "  2) Caudan South"
echo "  3) La Chaussee"
echo "  4) Place D'Armes"
echo "  5) To City Center"
echo "  6) Casernes Police Station"
echo ""
read -p "Select stream (1-6): " stream_choice

# Map choice to stream URL
case $stream_choice in
    1)
        STREAM_NAME="Caudan North"
        STREAM_URL="https://stream.myt.mu/prod/CAUDAN_NORTH.stream_720p/playlist.m3u8"
        ;;
    2)
        STREAM_NAME="Caudan South"
        STREAM_URL="https://stream.myt.mu/prod/CAUDAN_SOUTH.stream_720p/playlist.m3u8"
        ;;
    3)
        STREAM_NAME="La Chaussee"
        STREAM_URL="https://stream.myt.mu/prod/LA_CHAUSSEE_STREET.stream_720p/playlist.m3u8"
        ;;
    4)
        STREAM_NAME="Place D'Armes"
        STREAM_URL="https://stream.myt.mu/prod/PLACE_DARMES.stream_720p/playlist.m3u8"
        ;;
    5)
        STREAM_NAME="To City Center"
        STREAM_URL="https://stream.myt.mu/prod/CASERNES_CITY_CENTRE.stream_720p/playlist.m3u8"
        ;;
    6)
        STREAM_NAME="Casernes Police Station"
        STREAM_URL="https://stream.myt.mu/prod/CASERNES_BRABANT_STREET.stream_720p/playlist.m3u8"
        ;;
    *)
        echo -e "${RED}✗ Invalid selection${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Selected: $STREAM_NAME${NC}"
echo -e "${GREEN}URL: $STREAM_URL${NC}"
echo ""

# Backup existing config
if [ -f "$ODC_DIR/config.json" ]; then
    BACKUP_FILE="$ODC_DIR/config.json.backup.$(date +%Y%m%d_%H%M%S)"
    echo "Backing up existing config to: $BACKUP_FILE"
    cp "$ODC_DIR/config.json" "$BACKUP_FILE"
fi

# Create GStreamer pipeline for HLS
GSTREAMER_PIPELINE="souphttpsrc location=$STREAM_URL is-live=true ! hlsdemux ! queue ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=1280,height=720 ! appsink"

# Update config.json
echo "Updating OpenDataCam configuration..."

# Read existing config or create new one
if [ -f "$ODC_DIR/config.json" ]; then
    CONFIG=$(cat "$ODC_DIR/config.json")
else
    CONFIG='{}'
fi

# Update VIDEO_INPUT to use remote_hls_gstreamer
cat > "$ODC_DIR/config.json" << EOF
{
  "OPENDATACAM_VERSION": "3.0.2",
  "PATH_TO_YOLO_DARKNET": "/var/local/darknet",
  "VIDEO_INPUT": "remote_hls_gstreamer",
  "VIDEO_INPUTS_PARAMS": {
    "file": "opendatacam_videos/demo-video.mp4",
    "usbcam": "v4l2src device=/dev/video0 ! video/x-raw, framerate=30/1, width=640, height=480 ! videoconvert ! appsink",
    "raspberrycam": "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280, height=720, framerate=119/1, format=NV12 ! nvvidconv ! video/x-raw, format=BGRx, width=640, height=360 ! videoconvert ! video/x-raw, format=BGR ! appsink",
    "remote_hls_gstreamer": "$GSTREAMER_PIPELINE",
    "remote_hls_gstreamer_rpi": "$GSTREAMER_PIPELINE"
  },
  "NEURAL_NETWORK": "yolov4",
  "NEURAL_NETWORK_PARAMS": {
    "yolov4": {
      "data": "cfg/coco.data",
      "cfg": "cfg/yolov4.cfg",
      "weights": "yolov4.weights"
    }
  },
  "TRACKER_SETTINGS": {
    "objectMaxAreaInPercentageOfFrame": 80,
    "confidence_threshold": 0.2,
    "iouLimit": 0.05,
    "unMatchedFrameTolerance": 5
  },
  "COUNTER_SETTINGS": {
    "minAngleWithCountingLineThreshold": 5,
    "computeTrajectoryBasedOnNbOfPastFrame": 5
  },
  "DISPLAY_SETTINGS": {
    "counterEnabled": true,
    "pathfinderEnabled": false,
    "heatmapEnabled": false
  },
  "PORTS": {
    "app": 8080,
    "darknet_json_stream": 8070,
    "darknet_mjpeg_stream": 8090
  },
  "VALID_CLASSES": ["*"],
  "COUNTER_COLOR": "#FFE700",
  "PATHFINDER_COLOR": "#1E88E5"
}
EOF

echo -e "${GREEN}✓ Configuration updated${NC}"
echo ""

# Restart OpenDataCam if running
echo "Checking if OpenDataCam is running..."
if docker ps | grep -q opendatacam; then
    echo "Restarting OpenDataCam..."
    docker restart opendatacam
    echo -e "${GREEN}✓ OpenDataCam restarted${NC}"
else
    echo -e "${YELLOW}⚠ OpenDataCam is not running${NC}"
    echo "Start it with: docker start opendatacam"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Configuration Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Stream: $STREAM_NAME"
echo "OpenDataCam URL: http://localhost:8080"
echo ""
echo "Next steps:"
echo "  1. Open http://localhost:8080 in your browser"
echo "  2. Define counting lines for your analysis"
echo "  3. Start recording data"
echo ""
