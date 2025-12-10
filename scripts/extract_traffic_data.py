#!/usr/bin/env python3
"""
Traffic Data Extraction Script for OpenDataCam
This script extracts traffic data from OpenDataCam API and processes it for SUMO simulation
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import argparse
import os


class OpenDataCamExtractor:
    """
    Class to interact with OpenDataCam API and extract traffic data
    """
    
    def __init__(self, base_url="http://localhost:8080", output_dir="../data/processed"):
        """
        Initialize the extractor
        
        Args:
            base_url: Base URL of OpenDataCam instance
            output_dir: Directory to save processed data
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.api_url = f"{base_url}/api"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def check_connection(self):
        """Check if OpenDataCam is accessible"""
        try:
            response = requests.get(f"{self.api_url}/status")
            if response.status_code == 200:
                print(f"✓ Successfully connected to OpenDataCam at {self.base_url}")
                return True
            else:
                print(f"✗ Failed to connect to OpenDataCam: Status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Failed to connect to OpenDataCam: {e}")
            return False
    
    def get_recordings(self):
        """Get list of recordings from OpenDataCam"""
        try:
            response = requests.get(f"{self.api_url}/recordings")
            if response.status_code == 200:
                recordings = response.json()
                print(f"Found {len(recordings)} recordings")
                return recordings
            else:
                print(f"Failed to get recordings: Status {response.status_code}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"Error getting recordings: {e}")
            return []
    
    def get_counter_data(self, recording_id):
        """
        Get counter data for a specific recording
        
        Args:
            recording_id: ID of the recording
            
        Returns:
            Counter data as JSON
        """
        try:
            response = requests.get(f"{self.api_url}/recording/{recording_id}/counter")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get counter data: Status {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error getting counter data: {e}")
            return None
    
    def get_tracker_data(self, recording_id):
        """
        Get tracker data for a specific recording
        
        Args:
            recording_id: ID of the recording
            
        Returns:
            Tracker data as JSON
        """
        try:
            response = requests.get(f"{self.api_url}/recording/{recording_id}/tracker")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get tracker data: Status {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error getting tracker data: {e}")
            return None
    
    def calculate_speed_from_trajectory(self, trajectory, fps=30, pixel_to_meter=0.1):
        """
        Calculate average speed from trajectory data
        
        Args:
            trajectory: List of trajectory points with x, y coordinates
            fps: Frames per second of the video
            pixel_to_meter: Conversion factor from pixels to meters
            
        Returns:
            Average speed in m/s
        """
        if len(trajectory) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i]['x'] - trajectory[i-1]['x']
            dy = trajectory[i]['y'] - trajectory[i-1]['y']
            distance = np.sqrt(dx**2 + dy**2) * pixel_to_meter
            total_distance += distance
        
        # Calculate time duration
        time_duration = len(trajectory) / fps
        
        # Calculate average speed
        avg_speed = total_distance / time_duration if time_duration > 0 else 0.0
        
        return avg_speed
    
    def process_counter_data(self, counter_data, recording_name):
        """
        Process counter data into flow rate metrics
        
        Args:
            counter_data: Raw counter data from OpenDataCam
            recording_name: Name of the recording
            
        Returns:
            DataFrame with processed flow rate data
        """
        if not counter_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(counter_data)
        
        # Add timestamp column
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Group by counter and time interval (1 minute)
        df['time_interval'] = df['timestamp'].dt.floor('1min')
        
        # Calculate flow rate (vehicles per minute)
        flow_rate = df.groupby(['counter_id', 'time_interval', 'object_class']).size().reset_index(name='vehicle_count')
        flow_rate['flow_rate'] = flow_rate['vehicle_count']  # Already per minute
        
        # Save to CSV
        output_file = os.path.join(self.output_dir, f"{recording_name}_flow_rate.csv")
        flow_rate.to_csv(output_file, index=False)
        print(f"✓ Saved flow rate data to {output_file}")
        
        return flow_rate
    
    def process_tracker_data(self, tracker_data, recording_name, fps=30, pixel_to_meter=0.1):
        """
        Process tracker data to extract speed metrics
        
        Args:
            tracker_data: Raw tracker data from OpenDataCam
            recording_name: Name of the recording
            fps: Frames per second of the video
            pixel_to_meter: Conversion factor from pixels to meters
            
        Returns:
            DataFrame with processed speed data
        """
        if not tracker_data:
            return pd.DataFrame()
        
        speed_data = []
        
        for track in tracker_data:
            track_id = track.get('id')
            object_class = track.get('class')
            trajectory = track.get('trajectory', [])
            
            # Calculate average speed for this track
            avg_speed = self.calculate_speed_from_trajectory(trajectory, fps, pixel_to_meter)
            
            speed_data.append({
                'track_id': track_id,
                'object_class': object_class,
                'avg_speed_ms': avg_speed,
                'trajectory_length': len(trajectory)
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(speed_data)
        
        # Calculate average speed by object class
        avg_speed_by_class = df.groupby('object_class')['avg_speed_ms'].mean().reset_index()
        
        # Save to CSV
        output_file = os.path.join(self.output_dir, f"{recording_name}_speed.csv")
        df.to_csv(output_file, index=False)
        print(f"✓ Saved speed data to {output_file}")
        
        return df
    
    def extract_all_data(self):
        """Extract data from all recordings"""
        if not self.check_connection():
            return
        
        recordings = self.get_recordings()
        
        if not recordings:
            print("No recordings found")
            return
        
        for recording in recordings:
            recording_id = recording.get('_id')
            recording_name = recording.get('filename', recording_id)
            
            print(f"\nProcessing recording: {recording_name}")
            
            # Get and process counter data
            counter_data = self.get_counter_data(recording_id)
            if counter_data:
                self.process_counter_data(counter_data, recording_name)
            
            # Get and process tracker data
            tracker_data = self.get_tracker_data(recording_id)
            if tracker_data:
                self.process_tracker_data(tracker_data, recording_name)
        
        print("\n✓ Data extraction completed")


def main():
    parser = argparse.ArgumentParser(description='Extract traffic data from OpenDataCam')
    parser.add_argument('--url', default='http://localhost:8080', help='OpenDataCam base URL')
    parser.add_argument('--output', default='../data/processed', help='Output directory for processed data')
    parser.add_argument('--fps', type=int, default=30, help='Video frames per second')
    parser.add_argument('--pixel-to-meter', type=float, default=0.1, help='Pixel to meter conversion factor')
    
    args = parser.parse_args()
    
    extractor = OpenDataCamExtractor(base_url=args.url, output_dir=args.output)
    extractor.extract_all_data()


if __name__ == "__main__":
    main()
