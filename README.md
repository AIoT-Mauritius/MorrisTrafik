# Automated CCTV Traffic Data Extraction and Analysis System

This project provides a complete system for automated data extraction and analysis from live public CCTV feeds. It leverages open-source tools like OpenDataCam and SUMO to build a data pipeline for training and evaluating AI-based adaptive traffic control systems.

## Table of Contents

- [System Overview](#system-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [AI Models](#ai-models)
- [Contributing](#contributing)
- [License](#license)

## System Overview

The system is designed to perform the following tasks:

1.  **Data Acquisition**: Record video footage from public CCTV feeds.
2.  **Data Extraction**: Use OpenDataCam to extract traffic data (vehicle counts, speeds, trajectories) from the recorded videos.
3.  **Simulation Environment**: Generate a realistic SUMO simulation of the target intersection using data from OpenStreetMap.
4.  **Data Transformation**: Process the extracted traffic data to calibrate the SUMO simulation.
5.  **AI-based Traffic Control**: Train and evaluate AI agents (LSTM for congestion prediction and DQN for traffic light control) in the SUMO environment.

## Features

-   Automated data extraction from video files using OpenDataCam.
-   Generation of realistic SUMO traffic simulations from OpenStreetMap data.
-   Calibration of SUMO simulations using real-world traffic data.
-   Implementation of an LSTM-based Predictive Congestion Agent (PCA).
-   Implementation of a DQN-based Local Junction Agent (LJA) for adaptive traffic light control.
-   Comprehensive data visualization and analysis tools.

## Project Structure

```
/cctv-traffic-analysis
|-- config/                  # Configuration files
|-- data/
|   |-- raw/                 # Raw video files
|   `-- processed/           # Processed traffic data (CSVs)
|-- docs/
|   `-- figures/             # Generated figures and plots
|-- models/
|   |-- lstm_congestion_predictor.py
|   `-- dqn_junction_agent.py
|-- opendatacam/             # OpenDataCam installation and data
|-- scripts/
|   |-- install_opendatacam.sh
|   |-- extract_traffic_data.py
|   |-- generate_sumo_network.py
|   |-- transform_data_for_sumo.py
|   `-- visualize_traffic_data.py
`-- README.md
```

## Installation

1.  **Prerequisites**:
    *   Docker and Docker-Compose
    *   NVIDIA GPU with CUDA 11 and cuDNN 8 (for OpenDataCam)
    *   SUMO (https://sumo.dlr.de/docs/Installing/index.html)
    *   Python 3.8+ with dependencies (pandas, numpy, tensorflow, etc.)

2.  **Install OpenDataCam**:

    ```bash
    cd /home/ubuntu/cctv-traffic-analysis/scripts
    ./install_opendatacam.sh
    ```

    Follow the on-screen instructions to complete the OpenDataCam installation.

3.  **Install Python Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Data Acquisition**: Place your recorded video files in the `/home/ubuntu/cctv-traffic-analysis/data/raw` directory.

2.  **Data Extraction**: Run the `extract_traffic_data.py` script to process the videos with OpenDataCam and extract traffic data.

    ```bash
    cd /home/ubuntu/cctv-traffic-analysis/scripts
    python extract_traffic_data.py
    ```

3.  **Generate SUMO Network**: Generate the SUMO network for your target location.

    ```bash
    cd /home/ubuntu/cctv-traffic-analysis/scripts
    python generate_sumo_network.py --name "your_location_name" --bbox <min_lon> <min_lat> <max_lon> <max_lat>
    ```

4.  **Transform Data for SUMO**: Transform the extracted traffic data into SUMO-compatible formats.

    ```bash
    cd /home/ubuntu/cctv-traffic-analysis/scripts
    python transform_data_for_sumo.py --flow-rate ../data/processed/your_video_flow_rate.csv --speed ../data/processed/your_video_speed.csv
    ```

5.  **Train AI Models**:

    *   **LSTM Congestion Predictor**:

        ```bash
        cd /home/ubuntu/cctv-traffic-analysis/models
        python lstm_congestion_predictor.py --data ../data/processed/your_combined_data.csv
        ```

    *   **DQN Junction Agent**:

        ```bash
        cd /home/ubuntu/cctv-traffic-analysis/models
        python dqn_junction_agent.py --sumo-cfg ../data/sumo/simulation.sumocfg --junction-id "your_junction_id"
        ```

6.  **Visualize Data**: Generate visualizations of the traffic data.

    ```bash
    cd /home/ubuntu/cctv-traffic-analysis/scripts
    python visualize_traffic_data.py --flow-rate ../data/processed/your_video_flow_rate.csv --speed ../data/processed/your_video_speed.csv
    ```

## AI Models

### Predictive Congestion Agent (PCA) - LSTM

The PCA is an LSTM-based neural network that predicts future congestion based on historical traffic data. It takes a sequence of traffic flow rates, average speeds, and time features as input and outputs a Congestion Risk Score (CRS).

### Local Junction Agent (LJA) - DQN

The LJA is a Deep Q-Network agent that learns to control traffic lights in the SUMO simulation. It uses the current state of the intersection (queue lengths, traffic light phase) and the predicted CRS from the PCA to make decisions about traffic light phasing.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.


## Live Stream Analysis (my.t Traffic Watch)

This system now supports live data extraction from my.t Traffic Watch streams.

### Automated Live Analysis

The `run_live_analysis.py` script provides an end-to-end pipeline for live data collection and analysis:

```bash
cd /home/ubuntu/cctv-traffic-analysis/scripts

# Run the full pipeline for Caudan North for 30 minutes
python3 run_live_analysis.py --stream caudan_north --duration 30
```

This will:
1.  Capture a 30-minute video from the live stream.
2.  Check for OpenDataCam and guide you through processing.
3.  Generate visualizations and SUMO data.

### Manual Stream Capture

You can also capture streams manually:

```bash
cd /home/ubuntu/cctv-traffic-analysis/scripts

# List available streams
python3 capture_myt_streams.py --list-streams

# Capture Caudan South stream for 1 hour
python3 capture_myt_streams.py --stream caudan_south --duration 60
```

### Configure OpenDataCam for Live Streaming

To process a live stream directly with OpenDataCam (without recording first):

```bash
cd /home/ubuntu/cctv-traffic-analysis/scripts
./configure_opendatacam_live.sh
```

This script will prompt you to select a stream and will automatically update your OpenDataCam configuration and restart it.
