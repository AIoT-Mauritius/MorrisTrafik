#!/usr/bin/env python3
"""
Traffic Data Visualization Script
This script creates visualizations for traffic data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import os


class TrafficDataVisualizer:
    """
    Class to visualize traffic data
    """
    
    def __init__(self, output_dir="../docs/figures"):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def plot_flow_rate_time_series(self, flow_data, save_name="flow_rate_time_series.png"):
        """
        Plot flow rate time series
        
        Args:
            flow_data: DataFrame with flow rate data
            save_name: Name of the output file
        """
        plt.figure(figsize=(14, 6))
        
        # Convert time_interval to datetime
        flow_data['time_interval'] = pd.to_datetime(flow_data['time_interval'])
        
        # Group by time interval and sum across all counters
        flow_by_time = flow_data.groupby('time_interval')['flow_rate'].sum()
        
        # Plot
        plt.plot(flow_by_time.index, flow_by_time.values, linewidth=2, color='#2E86AB')
        plt.fill_between(flow_by_time.index, flow_by_time.values, alpha=0.3, color='#2E86AB')
        
        plt.xlabel('Time', fontsize=12, fontweight='bold')
        plt.ylabel('Flow Rate (vehicles/minute)', fontsize=12, fontweight='bold')
        plt.title('Traffic Flow Rate Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, save_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Flow rate time series saved to {output_path}")
    
    def plot_flow_rate_by_vehicle_type(self, flow_data, save_name="flow_rate_by_vehicle_type.png"):
        """
        Plot flow rate by vehicle type
        
        Args:
            flow_data: DataFrame with flow rate data
            save_name: Name of the output file
        """
        plt.figure(figsize=(12, 6))
        
        # Convert time_interval to datetime
        flow_data['time_interval'] = pd.to_datetime(flow_data['time_interval'])
        
        # Group by time interval and vehicle type
        flow_by_type = flow_data.groupby(['time_interval', 'object_class'])['flow_rate'].sum().unstack(fill_value=0)
        
        # Plot stacked area chart
        flow_by_type.plot(kind='area', stacked=True, alpha=0.7, ax=plt.gca())
        
        plt.xlabel('Time', fontsize=12, fontweight='bold')
        plt.ylabel('Flow Rate (vehicles/minute)', fontsize=12, fontweight='bold')
        plt.title('Traffic Flow Rate by Vehicle Type', fontsize=14, fontweight='bold')
        plt.legend(title='Vehicle Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, save_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Flow rate by vehicle type saved to {output_path}")
    
    def plot_hourly_flow_pattern(self, flow_data, save_name="hourly_flow_pattern.png"):
        """
        Plot hourly flow pattern
        
        Args:
            flow_data: DataFrame with flow rate data
            save_name: Name of the output file
        """
        plt.figure(figsize=(12, 6))
        
        # Convert time_interval to datetime
        flow_data['time_interval'] = pd.to_datetime(flow_data['time_interval'])
        
        # Extract hour
        flow_data['hour'] = flow_data['time_interval'].dt.hour
        
        # Group by hour and calculate mean flow rate
        hourly_flow = flow_data.groupby('hour')['flow_rate'].mean()
        
        # Plot bar chart
        bars = plt.bar(hourly_flow.index, hourly_flow.values, color='#A23B72', alpha=0.8, edgecolor='black')
        
        # Highlight peak hours
        peak_hour = hourly_flow.idxmax()
        bars[peak_hour].set_color('#F18F01')
        
        plt.xlabel('Hour of Day', fontsize=12, fontweight='bold')
        plt.ylabel('Average Flow Rate (vehicles/minute)', fontsize=12, fontweight='bold')
        plt.title('Average Hourly Traffic Flow Pattern', fontsize=14, fontweight='bold')
        plt.xticks(range(24))
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, save_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Hourly flow pattern saved to {output_path}")
    
    def plot_speed_distribution(self, speed_data, save_name="speed_distribution.png"):
        """
        Plot speed distribution
        
        Args:
            speed_data: DataFrame with speed data
            save_name: Name of the output file
        """
        plt.figure(figsize=(12, 6))
        
        # Plot histogram
        plt.hist(speed_data['avg_speed_ms'], bins=30, color='#06A77D', alpha=0.7, edgecolor='black')
        
        # Add mean line
        mean_speed = speed_data['avg_speed_ms'].mean()
        plt.axvline(mean_speed, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_speed:.2f} m/s')
        
        plt.xlabel('Average Speed (m/s)', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title('Distribution of Vehicle Speeds', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, save_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Speed distribution saved to {output_path}")
    
    def plot_speed_by_vehicle_type(self, speed_data, save_name="speed_by_vehicle_type.png"):
        """
        Plot speed by vehicle type
        
        Args:
            speed_data: DataFrame with speed data
            save_name: Name of the output file
        """
        plt.figure(figsize=(10, 6))
        
        # Create box plot
        speed_data.boxplot(column='avg_speed_ms', by='object_class', ax=plt.gca())
        
        plt.xlabel('Vehicle Type', fontsize=12, fontweight='bold')
        plt.ylabel('Average Speed (m/s)', fontsize=12, fontweight='bold')
        plt.title('Speed Distribution by Vehicle Type', fontsize=14, fontweight='bold')
        plt.suptitle('')  # Remove default title
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, save_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Speed by vehicle type saved to {output_path}")
    
    def plot_vehicle_type_distribution(self, flow_data, save_name="vehicle_type_distribution.png"):
        """
        Plot vehicle type distribution
        
        Args:
            flow_data: DataFrame with flow rate data
            save_name: Name of the output file
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate total counts by vehicle type
        vehicle_counts = flow_data.groupby('object_class')['vehicle_count'].sum()
        
        # Create pie chart
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#C73E1D', '#6A4C93']
        plt.pie(vehicle_counts.values, labels=vehicle_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        
        plt.title('Vehicle Type Distribution', fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, save_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Vehicle type distribution saved to {output_path}")
    
    def generate_all_visualizations(self, flow_rate_file, speed_file):
        """
        Generate all visualizations
        
        Args:
            flow_rate_file: Path to flow rate CSV file
            speed_file: Path to speed CSV file
        """
        print(f"\n{'='*50}")
        print("Generating Traffic Data Visualizations")
        print(f"{'='*50}\n")
        
        # Load data
        flow_data = pd.read_csv(flow_rate_file)
        speed_data = pd.read_csv(speed_file)
        
        # Generate visualizations
        self.plot_flow_rate_time_series(flow_data)
        self.plot_flow_rate_by_vehicle_type(flow_data)
        self.plot_hourly_flow_pattern(flow_data)
        self.plot_speed_distribution(speed_data)
        self.plot_speed_by_vehicle_type(speed_data)
        self.plot_vehicle_type_distribution(flow_data)
        
        print(f"\n{'='*50}")
        print(f"✓ All visualizations saved to {self.output_dir}")
        print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize traffic data')
    parser.add_argument('--flow-rate', required=True, help='Path to flow rate CSV file')
    parser.add_argument('--speed', required=True, help='Path to speed CSV file')
    parser.add_argument('--output', default='../docs/figures', help='Output directory for figures')
    
    args = parser.parse_args()
    
    visualizer = TrafficDataVisualizer(output_dir=args.output)
    visualizer.generate_all_visualizations(args.flow_rate, args.speed)


if __name__ == "__main__":
    main()
