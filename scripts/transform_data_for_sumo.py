#!/usr/bin/env python3
"""
Data Transformation Script for SUMO Calibration
This script transforms extracted traffic data into SUMO-compatible formats
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import argparse
import os
from datetime import datetime, timedelta


class SUMODataTransformer:
    """
    Class to transform traffic data for SUMO simulation
    """
    
    def __init__(self, flow_rate_file, speed_file, output_dir="../data/sumo"):
        """
        Initialize the transformer
        
        Args:
            flow_rate_file: Path to flow rate CSV file
            speed_file: Path to speed CSV file
            output_dir: Directory to save SUMO files
        """
        self.flow_rate_file = flow_rate_file
        self.speed_file = speed_file
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.flow_data = None
        self.speed_data = None
        self.load_data()
    
    def load_data(self):
        """Load traffic data from CSV files"""
        try:
            self.flow_data = pd.read_csv(self.flow_rate_file)
            print(f"✓ Loaded flow rate data: {len(self.flow_data)} records")
        except Exception as e:
            print(f"✗ Failed to load flow rate data: {e}")
        
        try:
            self.speed_data = pd.read_csv(self.speed_file)
            print(f"✓ Loaded speed data: {len(self.speed_data)} records")
        except Exception as e:
            print(f"✗ Failed to load speed data: {e}")
    
    def calculate_vehicle_distribution(self):
        """Calculate vehicle type distribution from flow data"""
        if self.flow_data is None:
            return {}
        
        # Group by object class and sum vehicle counts
        vehicle_dist = self.flow_data.groupby('object_class')['vehicle_count'].sum()
        total_vehicles = vehicle_dist.sum()
        
        # Calculate percentages
        vehicle_dist_pct = (vehicle_dist / total_vehicles * 100).to_dict()
        
        print("\nVehicle Type Distribution:")
        for vehicle_type, percentage in vehicle_dist_pct.items():
            print(f"  {vehicle_type}: {percentage:.2f}%")
        
        return vehicle_dist_pct
    
    def calculate_average_speeds(self):
        """Calculate average speeds by vehicle type"""
        if self.speed_data is None:
            return {}
        
        # Group by object class and calculate mean speed
        avg_speeds = self.speed_data.groupby('object_class')['avg_speed_ms'].mean().to_dict()
        
        print("\nAverage Speeds by Vehicle Type:")
        for vehicle_type, speed in avg_speeds.items():
            print(f"  {vehicle_type}: {speed:.2f} m/s ({speed * 3.6:.2f} km/h)")
        
        return avg_speeds
    
    def calculate_hourly_flow_rates(self):
        """Calculate hourly flow rates"""
        if self.flow_data is None:
            return {}
        
        # Convert time_interval to datetime
        self.flow_data['time_interval'] = pd.to_datetime(self.flow_data['time_interval'])
        
        # Extract hour
        self.flow_data['hour'] = self.flow_data['time_interval'].dt.hour
        
        # Group by hour and calculate mean flow rate
        hourly_flow = self.flow_data.groupby('hour')['flow_rate'].mean().to_dict()
        
        print("\nHourly Flow Rates (vehicles/minute):")
        for hour, flow_rate in sorted(hourly_flow.items()):
            print(f"  {hour:02d}:00 - {flow_rate:.2f}")
        
        return hourly_flow
    
    def generate_vehicle_types_file(self, avg_speeds):
        """
        Generate SUMO vehicle types file
        
        Args:
            avg_speeds: Dictionary of average speeds by vehicle type
        """
        output_file = os.path.join(self.output_dir, "vehicle_types.rou.xml")
        
        # Create root element
        root = ET.Element("routes")
        
        # Map object classes to SUMO vehicle types
        vehicle_type_mapping = {
            'car': {'vClass': 'passenger', 'color': '1,0,0'},
            'truck': {'vClass': 'truck', 'color': '0,1,0'},
            'bus': {'vClass': 'bus', 'color': '0,0,1'},
            'motorcycle': {'vClass': 'motorcycle', 'color': '1,1,0'},
            'bicycle': {'vClass': 'bicycle', 'color': '0,1,1'},
            'person': {'vClass': 'pedestrian', 'color': '1,0,1'},
        }
        
        # Create vehicle types
        for obj_class, speed in avg_speeds.items():
            vehicle_type = obj_class.lower()
            
            # Get vehicle class and color
            v_class = vehicle_type_mapping.get(vehicle_type, {}).get('vClass', 'passenger')
            color = vehicle_type_mapping.get(vehicle_type, {}).get('color', '1,1,1')
            
            # Create vType element
            vtype_attrs = {
                'id': vehicle_type,
                'vClass': v_class,
                'speedDev': '0.1',
                'color': color,
            }
            
            # Add speed if available
            if speed > 0:
                vtype_attrs['maxSpeed'] = f"{speed:.2f}"
            
            ET.SubElement(root, "vType", vtype_attrs)
        
        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        
        print(f"\n✓ Vehicle types file generated: {output_file}")
    
    def generate_flows_file(self, hourly_flow, vehicle_dist):
        """
        Generate SUMO flows file
        
        Args:
            hourly_flow: Dictionary of hourly flow rates
            vehicle_dist: Dictionary of vehicle type distribution
        """
        output_file = os.path.join(self.output_dir, "flows.rou.xml")
        
        # Create root element
        root = ET.Element("routes")
        
        # Add vehicle types reference
        ET.SubElement(root, "include", {"href": "vehicle_types.rou.xml"})
        
        # Generate flows for each hour
        for hour, flow_rate in sorted(hourly_flow.items()):
            begin_time = hour * 3600  # Convert hour to seconds
            end_time = (hour + 1) * 3600
            
            # Calculate vehicles per hour
            vehicles_per_hour = flow_rate * 60
            
            # Generate flows for each vehicle type
            for vehicle_type, percentage in vehicle_dist.items():
                vehicle_type_lower = vehicle_type.lower()
                
                # Calculate flow for this vehicle type
                type_vehicles_per_hour = vehicles_per_hour * (percentage / 100)
                
                if type_vehicles_per_hour > 0:
                    flow_attrs = {
                        'id': f"flow_{vehicle_type_lower}_{hour}",
                        'type': vehicle_type_lower,
                        'begin': str(begin_time),
                        'end': str(end_time),
                        'vehsPerHour': f"{type_vehicles_per_hour:.2f}",
                        'from': "edge_in",  # Placeholder - needs to be updated with actual edge IDs
                        'to': "edge_out",   # Placeholder - needs to be updated with actual edge IDs
                    }
                    
                    ET.SubElement(root, "flow", flow_attrs)
        
        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        
        print(f"✓ Flows file generated: {output_file}")
        print("\nNote: Please update 'from' and 'to' edge IDs in the flows file")
        print("      with actual edge IDs from your SUMO network.")
    
    def generate_sumo_config(self, net_file="caudan_roundabout.net.xml"):
        """
        Generate SUMO configuration file
        
        Args:
            net_file: Name of the network file
        """
        output_file = os.path.join(self.output_dir, "simulation.sumocfg")
        
        # Create root element
        root = ET.Element("configuration")
        
        # Input section
        input_elem = ET.SubElement(root, "input")
        ET.SubElement(input_elem, "net-file", {"value": net_file})
        ET.SubElement(input_elem, "route-files", {"value": "flows.rou.xml"})
        
        # Time section
        time_elem = ET.SubElement(root, "time")
        ET.SubElement(time_elem, "begin", {"value": "0"})
        ET.SubElement(time_elem, "end", {"value": "86400"})  # 24 hours
        ET.SubElement(time_elem, "step-length", {"value": "1"})
        
        # Processing section
        processing_elem = ET.SubElement(root, "processing")
        ET.SubElement(processing_elem, "time-to-teleport", {"value": "-1"})
        
        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        
        print(f"✓ SUMO configuration file generated: {output_file}")
    
    def transform_all(self):
        """Transform all data for SUMO"""
        print(f"\n{'='*50}")
        print("Transforming Traffic Data for SUMO")
        print(f"{'='*50}\n")
        
        # Calculate metrics
        vehicle_dist = self.calculate_vehicle_distribution()
        avg_speeds = self.calculate_average_speeds()
        hourly_flow = self.calculate_hourly_flow_rates()
        
        # Generate SUMO files
        if avg_speeds:
            self.generate_vehicle_types_file(avg_speeds)
        
        if hourly_flow and vehicle_dist:
            self.generate_flows_file(hourly_flow, vehicle_dist)
        
        self.generate_sumo_config()
        
        print(f"\n{'='*50}")
        print("✓ Data transformation completed")
        print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description='Transform traffic data for SUMO simulation')
    parser.add_argument('--flow-rate', required=True, help='Path to flow rate CSV file')
    parser.add_argument('--speed', required=True, help='Path to speed CSV file')
    parser.add_argument('--output', default='../data/sumo', help='Output directory')
    
    args = parser.parse_args()
    
    transformer = SUMODataTransformer(
        flow_rate_file=args.flow_rate,
        speed_file=args.speed,
        output_dir=args.output
    )
    
    transformer.transform_all()


if __name__ == "__main__":
    main()
