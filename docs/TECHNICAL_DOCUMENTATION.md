# Technical Documentation: Automated CCTV Traffic Data Extraction and Analysis System

**Author**: Manus AI  
**Date**: December 10, 2025  
**Version**: 1.0

## Executive Summary

This document provides comprehensive technical documentation for the Automated CCTV Traffic Data Extraction and Analysis System. The system is designed to extract traffic data from public CCTV feeds, process this data for use in traffic simulations, and train AI-based agents for adaptive traffic control. The implementation leverages open-source tools including OpenDataCam for computer vision-based traffic analysis and SUMO for microscopic traffic simulation.

## 1. Introduction

### 1.1 Background

Adaptive Traffic Control Systems (ATCS) represent a significant advancement in urban traffic management, utilizing real-time data and intelligent algorithms to optimize traffic flow. However, the development and validation of such systems require substantial amounts of high-quality traffic data, which is often difficult and expensive to obtain through traditional methods. This project addresses this challenge by providing an automated pipeline for extracting traffic data from publicly available CCTV feeds.

### 1.2 Objectives

The primary objectives of this system are to automate the extraction of key traffic parameters from video footage, generate realistic traffic simulations calibrated with real-world data, and provide a framework for training and evaluating AI-based traffic control agents. Specifically, the system aims to extract vehicle flow rates and average speeds from CCTV footage, create SUMO simulation environments based on real road network geometries, and implement both predictive and reactive AI agents for traffic management.

### 1.3 Scope

This documentation covers the complete system architecture, implementation details of all components, usage instructions for each module, and integration guidelines for the AI agents. The system is designed to be modular and extensible, allowing researchers and practitioners to adapt it to various traffic scenarios and locations.

## 2. System Architecture

### 2.1 Overview

The system architecture consists of three primary stages: Data Acquisition and Extraction, Simulation Environment Setup, and AI-based Traffic Control. Each stage is implemented as a collection of independent modules that communicate through well-defined data formats.

The Data Acquisition and Extraction stage involves recording video footage from public CCTV feeds and processing these videos using OpenDataCam to extract traffic metrics. OpenDataCam employs YOLO-based object detection to identify and classify vehicles, DeepSort tracking algorithms to follow vehicles across frames, and virtual counting lines to measure traffic flow at specific locations.

The Simulation Environment Setup stage uses OpenStreetMap data to generate realistic road network geometries and transforms the extracted traffic data into SUMO-compatible formats for simulation calibration. The SUMO simulation provides a controlled environment for training and testing traffic control strategies.

The AI-based Traffic Control stage implements two complementary agents: an LSTM-based Predictive Congestion Agent that forecasts future traffic conditions and a DQN-based Local Junction Agent that makes real-time traffic light control decisions. These agents work together to optimize traffic flow while minimizing congestion.

### 2.2 Data Flow

The data flow through the system follows a sequential pipeline. Video footage is first processed by OpenDataCam, which outputs time-series data containing vehicle counts, speeds, and trajectories. This raw data is then transformed into aggregated metrics suitable for both LSTM training and SUMO calibration. The SUMO simulation uses the calibrated traffic demand to generate realistic traffic scenarios, which serve as the training environment for the DQN agent. The LSTM agent provides congestion predictions that inform the DQN agent's decision-making process.

### 2.3 Technology Stack

The system is built on a foundation of proven open-source technologies. OpenDataCam provides the computer vision capabilities for traffic analysis, utilizing YOLO for object detection and tracking algorithms for trajectory extraction. SUMO serves as the microscopic traffic simulation platform, offering detailed modeling of individual vehicle movements and interactions. The AI components are implemented using TensorFlow and Keras for deep learning, with Python serving as the primary programming language. Docker containerization ensures consistent deployment across different hardware platforms.

## 3. Component Details

### 3.1 OpenDataCam Integration

OpenDataCam is deployed as a Docker container, providing a web-based interface for video processing and data extraction. The system supports both real-time camera feeds and pre-recorded video files, making it suitable for both live monitoring and historical analysis.

Configuration of OpenDataCam involves defining virtual counting lines at strategic locations within the camera's field of view. These lines correspond to entry and exit points of the intersection or road segment being analyzed. When a detected vehicle crosses a counting line, OpenDataCam records the timestamp, vehicle type, and direction of travel.

The data extraction script (`extract_traffic_data.py`) interfaces with OpenDataCam's REST API to retrieve both counter data and tracker data. Counter data provides aggregated vehicle counts per time interval, while tracker data contains detailed trajectory information for each detected vehicle. This dual-level data collection enables both macroscopic flow analysis and microscopic behavior modeling.

### 3.2 SUMO Network Generation

The SUMO network generation process begins with downloading road network data from OpenStreetMap for the specified geographic area. The `generate_sumo_network.py` script uses the Overpass API to query OpenStreetMap for all highway elements within a bounding box defined by latitude and longitude coordinates.

The downloaded OSM data is then processed by SUMO's netconvert tool, which converts the raw geographic data into a SUMO network file. This conversion includes inferring traffic light positions, determining lane configurations, and setting appropriate speed limits based on road classifications. The script also creates a type file that defines vehicle characteristics for different road classes.

The resulting network file captures the geometric and topological properties of the real-world road network, providing a realistic foundation for traffic simulation. This approach ensures that the simulation environment accurately reflects the physical constraints and traffic patterns of the target location.

### 3.3 Data Transformation Pipeline

The data transformation pipeline bridges the gap between the raw traffic data extracted from videos and the structured formats required by SUMO and the AI models. The `transform_data_for_sumo.py` script performs several key transformations.

First, it calculates vehicle type distributions by analyzing the frequency of different vehicle classes in the extracted data. This distribution is used to generate realistic traffic demand patterns in the SUMO simulation. Second, it computes average speeds for each vehicle type, which are used to parameterize SUMO's vehicle type definitions. Third, it analyzes temporal patterns in the traffic data to identify hourly flow variations, enabling the simulation to replicate realistic daily traffic cycles.

The script outputs three primary files: a vehicle types file defining the characteristics of each vehicle class, a flows file specifying the traffic demand over time, and a SUMO configuration file that ties together the network, demand, and simulation parameters. These files collectively define a complete SUMO simulation scenario calibrated to match the observed traffic conditions.

### 3.4 LSTM Congestion Predictor

The LSTM-based Predictive Congestion Agent is implemented in `lstm_congestion_predictor.py` and serves as an early warning system for traffic congestion. The model architecture consists of three stacked LSTM layers with dropout regularization to prevent overfitting.

The input to the model is a sequence of historical traffic observations, typically covering the past four hours at 15-minute intervals. Each observation includes traffic flow rate, average speed, hour of day, and day of week. This temporal context allows the LSTM to learn recurring patterns such as morning and evening rush hours.

The output is a Congestion Risk Score ranging from 0 to 1, where higher values indicate higher likelihood of congestion in the next 15 minutes. This prediction horizon is chosen to provide sufficient lead time for proactive traffic management interventions while maintaining prediction accuracy.

Training the LSTM model requires a substantial dataset of historical traffic observations. The model uses mean squared error as the loss function and employs early stopping to prevent overfitting. Once trained, the model can generate real-time predictions by processing recent traffic data through the learned LSTM layers.

### 3.5 DQN Junction Agent

The DQN-based Local Junction Agent is implemented in `dqn_junction_agent.py` and provides real-time adaptive traffic light control. The agent learns an optimal control policy through trial and error in the SUMO simulation environment.

The state space of the DQN agent includes queue lengths for all controlled lanes, the current traffic light phase, and the predicted Congestion Risk Score from the LSTM agent. This comprehensive state representation enables the agent to make informed decisions based on both current conditions and anticipated future developments.

The action space consists of two discrete actions: extending the current green phase or switching to the next phase in the traffic light cycle. This simplified action space balances control flexibility with learning efficiency, as a larger action space would require more training data to explore adequately.

The reward function is designed to minimize total vehicle delay and queue length. Specifically, the reward is the negative sum of total queue length and weighted waiting time. By maximizing this reward through reinforcement learning, the agent learns to minimize congestion.

The DQN algorithm uses experience replay to break correlations between consecutive training samples and employs a separate target network to stabilize learning. The agent gradually reduces its exploration rate (epsilon) as training progresses, transitioning from random exploration to exploitation of learned knowledge.

### 3.6 Visualization Tools

The visualization module (`visualize_traffic_data.py`) provides comprehensive graphical analysis of the extracted traffic data. It generates six primary types of visualizations.

Flow rate time series plots show how traffic volume varies over time, enabling identification of peak periods and unusual patterns. Flow rate by vehicle type plots use stacked area charts to illustrate the composition of traffic across different vehicle classes. Hourly flow pattern plots aggregate data by hour of day to reveal daily traffic cycles.

Speed distribution histograms show the statistical distribution of vehicle speeds, with markers for mean and median values. Speed by vehicle type box plots compare speed distributions across different vehicle classes, revealing behavioral differences between cars, trucks, buses, and other vehicles. Vehicle type distribution pie charts provide a high-level overview of traffic composition.

These visualizations serve multiple purposes: they validate the quality of extracted data, provide insights into traffic patterns that inform simulation calibration, and support the interpretation of AI model predictions and performance.

## 4. Installation and Setup

### 4.1 System Requirements

The system requires a Linux-based operating system, preferably Ubuntu 22.04 or later. Hardware requirements include an NVIDIA GPU with CUDA 11 and cuDNN 8 for OpenDataCam, at least 8GB of RAM for SUMO simulations, and sufficient storage for video files and extracted data.

Software prerequisites include Docker and Docker Compose for OpenDataCam deployment, SUMO version 1.10 or later for traffic simulation, and Python 3.8 or later with TensorFlow, pandas, numpy, and other scientific computing libraries.

### 4.2 Installation Steps

Begin by cloning or extracting the project files to a local directory. Run the OpenDataCam installation script (`install_opendatacam.sh`) which will download and configure the OpenDataCam Docker container. Install SUMO following the official installation guide for your operating system. Install Python dependencies using pip and the provided requirements.txt file.

Verify the installation by starting OpenDataCam and accessing its web interface at http://localhost:8080, running a simple SUMO simulation to confirm proper installation, and testing the Python scripts with sample data.

### 4.3 Configuration

OpenDataCam configuration involves setting up virtual counting lines in the web interface to match the geometry of your target intersection. SUMO configuration requires updating edge IDs in the flows file to match the generated network. AI model configuration includes adjusting hyperparameters such as learning rates, batch sizes, and network architectures based on your specific use case.

## 5. Usage Guide

### 5.1 Data Acquisition

Place recorded video files in the `data/raw` directory. Videos should be in a format supported by OpenDataCam, such as MP4 or AVI. For best results, videos should have a resolution of at least 720p and a frame rate of 25-30 fps. The camera angle should provide a clear view of the intersection or road segment with minimal occlusion.

### 5.2 Data Extraction

Start the OpenDataCam container and access the web interface. Upload your video file or configure a live camera feed. Define counting lines at entry and exit points of the intersection. Start the analysis and wait for processing to complete. Once finished, run the `extract_traffic_data.py` script to retrieve the data via the OpenDataCam API.

The extraction process outputs two CSV files: one containing flow rate data with timestamps, counter IDs, vehicle types, and counts, and another containing speed data with track IDs, vehicle types, average speeds, and trajectory lengths.

### 5.3 Network Generation

Run the `generate_sumo_network.py` script with appropriate parameters for your target location. Specify the location name, bounding box coordinates, and output directory. The script will download OSM data, generate the SUMO network, and create supporting files.

Verify the generated network by opening it in SUMO-GUI and visually inspecting the road geometry, lane configurations, and traffic light positions. Make manual adjustments if necessary using SUMO's netedit tool.

### 5.4 Data Transformation

Run the `transform_data_for_sumo.py` script with paths to the extracted flow rate and speed CSV files. The script will calculate vehicle distributions, average speeds, and hourly flow patterns, then generate SUMO-compatible vehicle types, flows, and configuration files.

Review the generated files and update the edge IDs in the flows file to match your specific network. The edge IDs can be found by inspecting the network file or using SUMO-GUI.

### 5.5 Model Training

For LSTM training, prepare a combined dataset with historical traffic data including flow rates, speeds, and time features. Run the `lstm_congestion_predictor.py` script with appropriate parameters for epochs, batch size, and test set size. Monitor training progress and evaluate the model on the test set.

For DQN training, ensure the SUMO configuration is correct and the simulation runs properly. Run the `dqn_junction_agent.py` script with the SUMO configuration file and junction ID. The agent will train through multiple episodes, gradually improving its control policy. Monitor the episode scores to track learning progress.

### 5.6 Visualization

Run the `visualize_traffic_data.py` script with paths to the flow rate and speed CSV files. The script will generate all visualizations and save them to the specified output directory. Review the visualizations to gain insights into traffic patterns and validate data quality.

## 6. Integration and Deployment

### 6.1 System Integration

The complete system can be integrated into a continuous monitoring and control pipeline. Video footage from CCTV cameras is periodically processed by OpenDataCam to extract current traffic conditions. The LSTM model generates congestion predictions based on recent historical data. The DQN agent uses these predictions along with real-time simulation state to make traffic light control decisions.

For real-world deployment, the SUMO simulation would be replaced with actual traffic light control systems. The DQN agent's learned policy can be exported and implemented in traffic signal controllers, while the LSTM predictions can inform higher-level traffic management strategies.

### 6.2 Performance Considerations

OpenDataCam processing speed depends on video resolution, frame rate, and available GPU resources. For real-time processing, ensure that the GPU can handle the computational load. The LSTM model has minimal inference latency and can generate predictions in milliseconds. The DQN agent's decision-making is also fast, but the SUMO simulation itself may be computationally intensive for large networks.

### 6.3 Scalability

The system can be scaled to handle multiple intersections by deploying multiple OpenDataCam instances and training separate DQN agents for each junction. The LSTM model can be trained on aggregated data from multiple locations to improve generalization. For city-wide deployment, consider distributed computing architectures and cloud-based processing.

## 7. Troubleshooting

### 7.1 Common Issues

If OpenDataCam fails to detect vehicles, check that the YOLO model is properly loaded and the video quality is sufficient. If SUMO simulations crash, verify that the network file is valid and all edge IDs referenced in the flows file exist. If AI models fail to converge during training, adjust learning rates, increase training data, or modify network architectures.

### 7.2 Performance Optimization

To improve OpenDataCam performance, reduce video resolution or frame rate if real-time processing is not required. For SUMO simulations, simplify the network by removing unnecessary details or reduce the simulation time step. For AI model training, use GPU acceleration and batch processing to speed up training.

## 8. Future Enhancements

### 8.1 Potential Improvements

Future versions of the system could incorporate additional data sources such as weather conditions, special events, and public transit schedules to improve prediction accuracy. Advanced computer vision techniques such as vehicle re-identification could enable origin-destination analysis. Multi-agent reinforcement learning could coordinate traffic lights across multiple intersections for network-wide optimization.

### 8.2 Research Directions

The system provides a foundation for various research directions including the comparison of different deep learning architectures for congestion prediction, investigation of transfer learning approaches to apply models trained on one location to another, and exploration of explainable AI techniques to understand and validate the decision-making processes of the AI agents.

## 9. Conclusion

This automated CCTV traffic data extraction and analysis system provides a comprehensive solution for developing and evaluating AI-based adaptive traffic control systems. By leveraging open-source tools and modern deep learning techniques, the system enables researchers and practitioners to work with real-world traffic data without the need for expensive data collection infrastructure. The modular architecture and detailed documentation facilitate customization and extension for specific use cases and research objectives.

## References

- OpenDataCam Project: https://opendata.cam/
- SUMO Documentation: https://sumo.dlr.de/docs/
- TensorFlow Documentation: https://www.tensorflow.org/
- OpenStreetMap: https://www.openstreetmap.org/
