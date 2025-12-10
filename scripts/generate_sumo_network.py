#!/usr/bin/env python3
"""
SUMO Network Generation Script
This script generates a SUMO network from OpenStreetMap data for the specified location
"""

import subprocess
import os
import argparse
import xml.etree.ElementTree as ET


class SUMONetworkGenerator:
    """
    Class to generate SUMO network from OpenStreetMap data
    """
    
    def __init__(self, location_name, bbox, output_dir="../data/sumo"):
        """
        Initialize the generator
        
        Args:
            location_name: Name of the location (e.g., "caudan_roundabout")
            bbox: Bounding box as tuple (min_lon, min_lat, max_lon, max_lat)
            output_dir: Directory to save SUMO network files
        """
        self.location_name = location_name
        self.bbox = bbox
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # File paths
        self.osm_file = os.path.join(output_dir, f"{location_name}.osm.xml")
        self.net_file = os.path.join(output_dir, f"{location_name}.net.xml")
        self.poly_file = os.path.join(output_dir, f"{location_name}.poly.xml")
        self.type_file = os.path.join(output_dir, f"{location_name}.typ.xml")
    
    def download_osm_data(self):
        """Download OSM data for the specified bounding box"""
        print(f"Downloading OSM data for {self.location_name}...")
        
        # Construct Overpass API query
        min_lon, min_lat, max_lon, max_lat = self.bbox
        overpass_query = f"""
        [out:xml][timeout:25];
        (
          way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
          node(w);
        );
        out body;
        >;
        out skel qt;
        """
        
        # Use curl to download OSM data
        overpass_url = "https://overpass-api.de/api/interpreter"
        
        try:
            result = subprocess.run(
                ["curl", "-X", "POST", "-d", overpass_query, overpass_url, "-o", self.osm_file],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ OSM data downloaded to {self.osm_file}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to download OSM data: {e}")
            return False
    
    def create_type_file(self):
        """Create a type file for edge types"""
        print("Creating type file...")
        
        # Create a basic type file
        root = ET.Element("types")
        
        # Add edge types
        edge_types = [
            {"id": "highway.motorway", "priority": "13", "numLanes": "3", "speed": "44.44"},
            {"id": "highway.trunk", "priority": "12", "numLanes": "2", "speed": "27.78"},
            {"id": "highway.primary", "priority": "11", "numLanes": "2", "speed": "22.22"},
            {"id": "highway.secondary", "priority": "10", "numLanes": "2", "speed": "16.67"},
            {"id": "highway.tertiary", "priority": "9", "numLanes": "1", "speed": "13.89"},
            {"id": "highway.residential", "priority": "8", "numLanes": "1", "speed": "13.89"},
        ]
        
        for edge_type in edge_types:
            ET.SubElement(root, "type", edge_type)
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(self.type_file, encoding="utf-8", xml_declaration=True)
        print(f"✓ Type file created at {self.type_file}")
    
    def generate_network(self):
        """Generate SUMO network using netconvert"""
        print("Generating SUMO network...")
        
        # Check if netconvert is available
        try:
            subprocess.run(["netconvert", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ netconvert not found. Please install SUMO first.")
            print("  Visit: https://sumo.dlr.de/docs/Installing/index.html")
            return False
        
        # Run netconvert
        cmd = [
            "netconvert",
            "--osm-files", self.osm_file,
            "--output-file", self.net_file,
            "--geometry.remove",
            "--ramps.guess",
            "--junctions.join",
            "--tls.guess-signals",
            "--tls.discard-simple",
            "--tls.join",
            "--output.street-names",
            "--output.original-names",
            "--junctions.corner-detail", "5",
            "--roundabouts.guess",
        ]
        
        # Add type file if it exists
        if os.path.exists(self.type_file):
            cmd.extend(["--type-files", self.type_file])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ SUMO network generated at {self.net_file}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to generate network: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    def generate_all(self):
        """Generate complete SUMO network"""
        print(f"\n{'='*50}")
        print(f"Generating SUMO Network for {self.location_name}")
        print(f"{'='*50}\n")
        
        # Download OSM data
        if not self.download_osm_data():
            return False
        
        # Create type file
        self.create_type_file()
        
        # Generate network
        if not self.generate_network():
            return False
        
        print(f"\n{'='*50}")
        print("✓ SUMO network generation completed successfully")
        print(f"{'='*50}\n")
        print(f"Network file: {self.net_file}")
        print(f"\nTo visualize the network, run:")
        print(f"  sumo-gui -n {self.net_file}")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Generate SUMO network from OpenStreetMap')
    parser.add_argument('--name', default='caudan_roundabout', help='Location name')
    parser.add_argument('--bbox', nargs=4, type=float, 
                        default=[57.4980, -20.1620, 57.5020, -20.1590],
                        help='Bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--output', default='../data/sumo', help='Output directory')
    
    args = parser.parse_args()
    
    generator = SUMONetworkGenerator(
        location_name=args.name,
        bbox=args.bbox,
        output_dir=args.output
    )
    
    generator.generate_all()


if __name__ == "__main__":
    main()
