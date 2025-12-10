"""
SUMO Network Generator
Generates SUMO network files from OpenStreetMap data
"""

import os
import subprocess
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SUMONetworkGenerator:
    """
    Generate SUMO network from OpenStreetMap data
    """
    
    # Predefined locations
    LOCATIONS = {
        'caudan': {
            'name': 'Caudan Roundabout, Port Louis',
            'bbox': '-20.1650,57.5000,-20.1550,57.5100',  # South,West,North,East
            'center': (-20.1600, 57.5050)
        },
        'port_louis': {
            'name': 'Port Louis City Center',
            'bbox': '-20.1700,57.4950,-20.1500,57.5150',
            'center': (-20.1600, 57.5050)
        }
    }
    
    def __init__(self, output_dir: str = "sumo_network"):
        """
        Initialize network generator
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.osm_file = None
        self.net_file = None
        self.poly_file = None
    
    def download_osm_data(self, bbox: str, output_name: str = "map") -> str:
        """
        Download OSM data for given bounding box
        
        Args:
            bbox: Bounding box as "south,west,north,east"
            output_name: Name for output file
            
        Returns:
            Path to downloaded OSM file
        """
        self.osm_file = self.output_dir / f"{output_name}.osm"
        
        # Parse bbox
        south, west, north, east = map(float, bbox.split(','))
        
        # Construct Overpass API query
        overpass_url = "https://overpass-api.de/api/map"
        bbox_param = f"?bbox={west},{south},{east},{north}"
        
        logger.info(f"Downloading OSM data for bbox: {bbox}")
        
        try:
            import requests
            response = requests.get(overpass_url + bbox_param, timeout=60)
            response.raise_for_status()
            
            with open(self.osm_file, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"OSM data saved to: {self.osm_file}")
            return str(self.osm_file)
        
        except Exception as e:
            logger.error(f"Failed to download OSM data: {e}")
            logger.info("Creating sample OSM file for demonstration...")
            self._create_sample_osm()
            return str(self.osm_file)
    
    def _create_sample_osm(self):
        """Create a sample OSM file for testing"""
        sample_osm = '''<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <bounds minlat="-20.1650" minlon="57.5000" maxlat="-20.1550" maxlon="57.5100"/>
  
  <!-- Sample nodes for a simple intersection -->
  <node id="1" lat="-20.1600" lon="57.5050"/>
  <node id="2" lat="-20.1600" lon="57.5070"/>
  <node id="3" lat="-20.1580" lon="57.5050"/>
  <node id="4" lat="-20.1620" lon="57.5050"/>
  <node id="5" lat="-20.1600" lon="57.5030"/>
  
  <!-- Roads -->
  <way id="100">
    <nd ref="5"/>
    <nd ref="1"/>
    <nd ref="2"/>
    <tag k="highway" v="primary"/>
    <tag k="name" v="North Road"/>
    <tag k="lanes" v="2"/>
    <tag k="maxspeed" v="50"/>
  </way>
  
  <way id="101">
    <nd ref="4"/>
    <nd ref="1"/>
    <nd ref="3"/>
    <tag k="highway" v="primary"/>
    <tag k="name" v="East Road"/>
    <tag k="lanes" v="2"/>
    <tag k="maxspeed" v="50"/>
  </way>
</osm>
'''
        with open(self.osm_file, 'w') as f:
            f.write(sample_osm)
        
        logger.info(f"Sample OSM file created: {self.osm_file}")
    
    def generate_network(self, osm_file: Optional[str] = None) -> str:
        """
        Generate SUMO network from OSM file using netconvert
        
        Args:
            osm_file: Path to OSM file (uses downloaded file if None)
            
        Returns:
            Path to generated network file
        """
        if osm_file:
            self.osm_file = Path(osm_file)
        
        if not self.osm_file or not self.osm_file.exists():
            raise FileNotFoundError("OSM file not found. Download OSM data first.")
        
        self.net_file = self.output_dir / "network.net.xml"
        
        # netconvert command
        cmd = [
            'netconvert',
            '--osm-files', str(self.osm_file),
            '--output-file', str(self.net_file),
            '--geometry.remove',
            '--roundabouts.guess',
            '--ramps.guess',
            '--junctions.join',
            '--tls.guess-signals',
            '--tls.discard-simple',
            '--tls.join',
            '--tls.default-type', 'actuated',
            '--no-turnarounds',
            '--verbose'
        ]
        
        logger.info("Generating SUMO network with netconvert...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info(f"Network generated successfully: {self.net_file}")
                return str(self.net_file)
            else:
                logger.error(f"netconvert failed: {result.stderr}")
                # Create a minimal network file for testing
                self._create_sample_network()
                return str(self.net_file)
        
        except FileNotFoundError:
            logger.error("netconvert not found. Please install SUMO.")
            logger.info("Creating sample network file for demonstration...")
            self._create_sample_network()
            return str(self.net_file)
        
        except Exception as e:
            logger.error(f"Error running netconvert: {e}")
            self._create_sample_network()
            return str(self.net_file)
    
    def _create_sample_network(self):
        """Create a sample SUMO network file"""
        sample_network = '''<?xml version="1.0" encoding="UTF-8"?>
<net version="1.16" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <location netOffset="0.00,0.00" convBoundary="0.00,0.00,1000.00,1000.00"/>
    
    <edge id="north_in" from="n_start" to="center" priority="2">
        <lane id="north_in_0" index="0" speed="13.89" length="400.00" shape="500.00,900.00 500.00,520.00"/>
        <lane id="north_in_1" index="1" speed="13.89" length="400.00" shape="503.20,900.00 503.20,520.00"/>
    </edge>
    
    <edge id="south_in" from="s_start" to="center" priority="2">
        <lane id="south_in_0" index="0" speed="13.89" length="400.00" shape="500.00,100.00 500.00,480.00"/>
        <lane id="south_in_1" index="1" speed="13.89" length="400.00" shape="496.80,100.00 496.80,480.00"/>
    </edge>
    
    <edge id="east_in" from="e_start" to="center" priority="2">
        <lane id="east_in_0" index="0" speed="13.89" length="400.00" shape="900.00,500.00 520.00,500.00"/>
        <lane id="east_in_1" index="1" speed="13.89" length="400.00" shape="900.00,496.80 520.00,496.80"/>
    </edge>
    
    <edge id="west_in" from="w_start" to="center" priority="2">
        <lane id="west_in_0" index="0" speed="13.89" length="400.00" shape="100.00,500.00 480.00,500.00"/>
        <lane id="west_in_1" index="1" speed="13.89" length="400.00" shape="100.00,503.20 480.00,503.20"/>
    </edge>
    
    <junction id="center" type="traffic_light" x="500.00" y="500.00" incLanes="north_in_0 north_in_1 south_in_0 south_in_1 east_in_0 east_in_1 west_in_0 west_in_1" intLanes="" shape="498.40,520.00 504.80,520.00 504.80,480.00 498.40,480.00 480.00,498.40 480.00,504.80 520.00,504.80 520.00,498.40">
        <request index="0" response="0000" foes="0000"/>
    </junction>
    
    <junction id="n_start" type="dead_end" x="500.00" y="900.00"/>
    <junction id="s_start" type="dead_end" x="500.00" y="100.00"/>
    <junction id="e_start" type="dead_end" x="900.00" y="500.00"/>
    <junction id="w_start" type="dead_end" x="100.00" y="500.00"/>
</net>
'''
        with open(self.net_file, 'w') as f:
            f.write(sample_network)
        
        logger.info(f"Sample network file created: {self.net_file}")
    
    def generate_from_location(self, location: str) -> Dict[str, str]:
        """
        Generate network for predefined location
        
        Args:
            location: Location key (e.g., 'caudan', 'port_louis')
            
        Returns:
            Dictionary with file paths
        """
        if location not in self.LOCATIONS:
            raise ValueError(f"Unknown location: {location}. Available: {list(self.LOCATIONS.keys())}")
        
        loc_data = self.LOCATIONS[location]
        logger.info(f"Generating network for: {loc_data['name']}")
        
        # Download OSM data
        osm_file = self.download_osm_data(loc_data['bbox'], location)
        
        # Generate network
        net_file = self.generate_network(osm_file)
        
        # Save location info
        info_file = self.output_dir / "location_info.json"
        with open(info_file, 'w') as f:
            json.dump(loc_data, f, indent=2)
        
        return {
            'osm_file': osm_file,
            'net_file': net_file,
            'info_file': str(info_file)
        }
    
    def get_network_info(self) -> Dict:
        """Get information about generated network"""
        if not self.net_file or not self.net_file.exists():
            return {}
        
        # Try to parse network file for basic info
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(self.net_file)
            root = tree.getroot()
            
            edges = root.findall('.//edge')
            junctions = root.findall('.//junction')
            
            return {
                'network_file': str(self.net_file),
                'num_edges': len(edges),
                'num_junctions': len(junctions)
            }
        except Exception as e:
            logger.error(f"Error parsing network file: {e}")
            return {'network_file': str(self.net_file)}


def main():
    parser = argparse.ArgumentParser(description='Generate SUMO network from OSM data')
    parser.add_argument('--location', '-l', choices=['caudan', 'port_louis'],
                       help='Predefined location')
    parser.add_argument('--bbox', '-b', help='Custom bounding box (south,west,north,east)')
    parser.add_argument('--osm-file', '-o', help='Use existing OSM file')
    parser.add_argument('--output-dir', '-d', default='sumo_network', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    generator = SUMONetworkGenerator(args.output_dir)
    
    if args.location:
        # Generate from predefined location
        files = generator.generate_from_location(args.location)
        print(f"\nNetwork generated for {args.location}:")
        for key, path in files.items():
            print(f"  {key}: {path}")
    
    elif args.bbox:
        # Generate from custom bbox
        osm_file = generator.download_osm_data(args.bbox)
        net_file = generator.generate_network(osm_file)
        print(f"\nNetwork generated:")
        print(f"  OSM file: {osm_file}")
        print(f"  Network file: {net_file}")
    
    elif args.osm_file:
        # Generate from existing OSM file
        net_file = generator.generate_network(args.osm_file)
        print(f"\nNetwork generated:")
        print(f"  Network file: {net_file}")
    
    else:
        print("Error: Specify --location, --bbox, or --osm-file")
        return
    
    # Print network info
    info = generator.get_network_info()
    if info:
        print(f"\nNetwork info:")
        for key, value in info.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
