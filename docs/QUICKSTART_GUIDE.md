# Quick Start Guide: CCTV Traffic Data Extraction System

This guide will help you get started with the automated CCTV traffic data extraction and analysis system in just a few steps.

## Prerequisites

Before you begin, ensure you have the following installed:

- Docker and Docker Compose
- Python 3.8 or later
- SUMO (Simulation of Urban MObility)
- NVIDIA GPU with CUDA 11 and cuDNN 8 (for OpenDataCam)

## Step 1: Install OpenDataCam

Navigate to the scripts directory and run the installation script:

```bash
cd /path/to/cctv-traffic-analysis/scripts
./install_opendatacam.sh
```

Follow the on-screen instructions to select your platform (desktop, Jetson Nano, or Jetson Xavier).

After installation, OpenDataCam will be available at http://localhost:8080

## Step 2: Install Python Dependencies

Install the required Python packages:

```bash
cd /path/to/cctv-traffic-analysis
pip install -r requirements.txt
```

## Step 3: Prepare Your Video Data

Place your CCTV video files in the `data/raw` directory:

```bash
cp /path/to/your/video.mp4 /path/to/cctv-traffic-analysis/data/raw/
```

## Step 4: Extract Traffic Data

### 4.1 Configure OpenDataCam

1. Open http://localhost:8080 in your web browser
2. Upload your video file or configure a live camera feed
3. Define counting lines at entry and exit points of the intersection
4. Start the analysis

### 4.2 Run the Extraction Script

Once OpenDataCam has processed your video, extract the data:

```bash
cd /path/to/cctv-traffic-analysis/scripts
python extract_traffic_data.py --url http://localhost:8080 --output ../data/processed
```

This will generate CSV files with flow rate and speed data.

## Step 5: Generate SUMO Network

Generate a SUMO network for your target location:

```bash
cd /path/to/cctv-traffic-analysis/scripts
python generate_sumo_network.py \
    --name "caudan_roundabout" \
    --bbox 57.4980 -20.1620 57.5020 -20.1590 \
    --output ../data/sumo
```

Replace the bounding box coordinates with your target location.

## Step 6: Transform Data for SUMO

Transform the extracted traffic data into SUMO-compatible formats:

```bash
cd /path/to/cctv-traffic-analysis/scripts
python transform_data_for_sumo.py \
    --flow-rate ../data/processed/your_video_flow_rate.csv \
    --speed ../data/processed/your_video_speed.csv \
    --output ../data/sumo
```

**Important**: After running this script, you need to manually update the edge IDs in the generated `flows.rou.xml` file to match your SUMO network.

## Step 7: Visualize Your Data

Generate visualizations to understand your traffic patterns:

```bash
cd /path/to/cctv-traffic-analysis/scripts
python visualize_traffic_data.py \
    --flow-rate ../data/processed/your_video_flow_rate.csv \
    --speed ../data/processed/your_video_speed.csv \
    --output ../docs/figures
```

Check the `docs/figures` directory for the generated plots.

## Step 8: Train AI Models (Optional)

### 8.1 Train LSTM Congestion Predictor

```bash
cd /path/to/cctv-traffic-analysis/models
python lstm_congestion_predictor.py \
    --data ../data/processed/combined_traffic_data.csv \
    --output ../models/lstm_pca \
    --epochs 100
```

### 8.2 Train DQN Junction Agent

```bash
cd /path/to/cctv-traffic-analysis/models
python dqn_junction_agent.py \
    --sumo-cfg ../data/sumo/simulation.sumocfg \
    --junction-id "your_junction_id" \
    --episodes 100 \
    --output ../models/dqn_lja
```

## Troubleshooting

### OpenDataCam not starting

- Check that Docker is running: `docker ps`
- Verify NVIDIA GPU is available: `nvidia-smi`
- Check OpenDataCam logs: `docker logs opendatacam`

### SUMO network generation fails

- Verify SUMO is installed: `netconvert --version`
- Check internet connection (required for downloading OSM data)
- Verify bounding box coordinates are correct

### Python scripts fail

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)
- Verify file paths are correct

## Next Steps

- Read the full [Technical Documentation](TECHNICAL_DOCUMENTATION.md) for detailed information
- Experiment with different video sources and locations
- Customize the AI models for your specific use case
- Integrate the system with real-time traffic control systems

## Support

For issues and questions, please refer to the project README and technical documentation.
