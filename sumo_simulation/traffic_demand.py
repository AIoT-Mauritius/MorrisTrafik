"""
Traffic Demand Generator for SUMO
Generates realistic traffic demand based on extracted CCTV data
"""

import xml.etree.ElementTree as ET
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficDemandGenerator:
    """
    Generate SUMO route files with realistic traffic demand
    """
    
    VEHICLE_TYPES = {
        'car': {
            'vClass': 'passenger',
            'accel': 2.6,
            'decel': 4.5,
            'sigma': 0.5,
            'length': 5.0,
            'minGap': 2.5,
            'maxSpeed': 55.0,
            'color': '1,1,0'
        },
        'truck': {
            'vClass': 'truck',
            'accel': 1.3,
            'decel': 4.0,
            'sigma': 0.5,
            'length': 12.0,
            'minGap': 3.0,
            'maxSpeed': 40.0,
            'color': '0,0,1'
        },
        'bus': {
            'vClass': 'bus',
            'accel': 1.2,
            'decel': 4.0,
            'sigma': 0.5,
            'length': 12.0,
            'minGap': 3.0,
            'maxSpeed': 45.0,
            'color': '0,1,1'
        },
        'motorcycle': {
            'vClass': 'motorcycle',
            'accel': 3.0,
            'decel': 5.0,
            'sigma': 0.3,
            'length': 2.5,
            'minGap': 1.5,
            'maxSpeed': 60.0,
            'color': '1,0,1'
        }
    }
    
    def __init__(self, network_file: str, output_dir: str = "sumo_network"):
        """
        Initialize demand generator
        
        Args:
            network_file: Path to SUMO network file
            output_dir: Output directory
        """
        self.network_file = Path(network_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.edges = []
        self.routes = []
        
        self._parse_network()
    
    def _parse_network(self):
        """Parse network file to extract edges"""
        try:
            tree = ET.parse(self.network_file)
            root = tree.getroot()
            
            # Extract all edges
            for edge in root.findall('.//edge'):
                edge_id = edge.get('id')
                # Filter out internal edges
                if edge_id and not edge_id.startswith(':'):
                    self.edges.append(edge_id)
            
            logger.info(f"Found {len(self.edges)} edges in network")
        
        except Exception as e:
            logger.error(f"Error parsing network file: {e}")
            # Use default edges for sample network
            self.edges = ['north_in', 'south_in', 'east_in', 'west_in']
    
    def generate_routes(self, num_routes: int = 10) -> List[Tuple[str, str]]:
        """
        Generate random routes (origin-destination pairs)
        
        Args:
            num_routes: Number of routes to generate
            
        Returns:
            List of (from_edge, to_edge) tuples
        """
        if len(self.edges) < 2:
            logger.warning("Not enough edges to generate routes")
            return []
        
        routes = []
        for _ in range(num_routes):
            from_edge = random.choice(self.edges)
            to_edge = random.choice([e for e in self.edges if e != from_edge])
            routes.append((from_edge, to_edge))
        
        self.routes = routes
        logger.info(f"Generated {len(routes)} routes")
        return routes
    
    def create_route_file(self, 
                         duration: int = 3600,
                         vehicles_per_hour: int = 1000,
                         vehicle_distribution: Dict[str, float] = None) -> str:
        """
        Create SUMO route file with traffic demand
        
        Args:
            duration: Simulation duration in seconds
            vehicles_per_hour: Number of vehicles per hour
            vehicle_distribution: Distribution of vehicle types (defaults to realistic mix)
            
        Returns:
            Path to generated route file
        """
        if vehicle_distribution is None:
            vehicle_distribution = {
                'car': 0.75,
                'truck': 0.10,
                'bus': 0.05,
                'motorcycle': 0.10
            }
        
        # Generate routes if not already done
        if not self.routes:
            self.generate_routes(num_routes=min(10, len(self.edges)))
        
        route_file = self.output_dir / "routes.rou.xml"
        
        # Create XML structure
        root = ET.Element('routes')
        
        # Add vehicle types
        for vtype, params in self.VEHICLE_TYPES.items():
            vtype_elem = ET.SubElement(root, 'vType')
            vtype_elem.set('id', vtype)
            for key, value in params.items():
                vtype_elem.set(key, str(value))
        
        # Calculate vehicle insertion rate
        interval = 3600 / vehicles_per_hour  # seconds between vehicles
        
        # Generate vehicles
        vehicle_id = 0
        current_time = 0
        
        while current_time < duration:
            # Select vehicle type based on distribution
            vtype = random.choices(
                list(vehicle_distribution.keys()),
                weights=list(vehicle_distribution.values())
            )[0]
            
            # Select random route
            from_edge, to_edge = random.choice(self.routes)
            
            # Create vehicle
            vehicle = ET.SubElement(root, 'vehicle')
            vehicle.set('id', f'veh_{vehicle_id}')
            vehicle.set('type', vtype)
            vehicle.set('depart', f'{current_time:.2f}')
            vehicle.set('from', from_edge)
            vehicle.set('to', to_edge)
            
            vehicle_id += 1
            current_time += interval + random.uniform(-interval * 0.2, interval * 0.2)
        
        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space='  ')
        tree.write(route_file, encoding='utf-8', xml_declaration=True)
        
        logger.info(f"Generated {vehicle_id} vehicles in route file: {route_file}")
        return str(route_file)
    
    def create_from_traffic_data(self, traffic_data_file: str) -> str:
        """
        Create route file based on extracted traffic data
        
        Args:
            traffic_data_file: Path to traffic_data.json from video processor
            
        Returns:
            Path to generated route file
        """
        import json
        
        try:
            with open(traffic_data_file, 'r') as f:
                data = json.load(f)
            
            # Extract vehicle counts
            line_counts = data.get('virtual_lines', [])
            
            if not line_counts:
                logger.warning("No line count data found, using default demand")
                return self.create_route_file()
            
            # Calculate average vehicles per hour from line counts
            total_count = sum(line['total_count'] for line in line_counts)
            video_duration = data['video_info']['total_frames'] / data['video_info']['fps']
            vehicles_per_hour = int((total_count / video_duration) * 3600)
            
            logger.info(f"Calculated demand: {vehicles_per_hour} vehicles/hour")
            
            # Extract vehicle type distribution
            vehicle_distribution = {}
            total_by_type = {}
            
            for line in line_counts:
                for vtype, count in line['count_by_type'].items():
                    total_by_type[vtype] = total_by_type.get(vtype, 0) + count
            
            total_vehicles = sum(total_by_type.values())
            if total_vehicles > 0:
                for vtype, count in total_by_type.items():
                    vehicle_distribution[vtype] = count / total_vehicles
            
            logger.info(f"Vehicle distribution: {vehicle_distribution}")
            
            # Generate route file
            return self.create_route_file(
                duration=3600,
                vehicles_per_hour=vehicles_per_hour,
                vehicle_distribution=vehicle_distribution if vehicle_distribution else None
            )
        
        except Exception as e:
            logger.error(f"Error processing traffic data: {e}")
            logger.info("Falling back to default demand generation")
            return self.create_route_file()
    
    def create_config_file(self, route_file: str, duration: int = 3600) -> str:
        """
        Create SUMO configuration file
        
        Args:
            route_file: Path to route file
            duration: Simulation duration
            
        Returns:
            Path to config file
        """
        config_file = self.output_dir / "simulation.sumocfg"
        
        root = ET.Element('configuration')
        
        # Input section
        input_elem = ET.SubElement(root, 'input')
        net_elem = ET.SubElement(input_elem, 'net-file')
        net_elem.set('value', str(self.network_file.name))
        route_elem = ET.SubElement(input_elem, 'route-files')
        route_elem.set('value', Path(route_file).name)
        
        # Time section
        time_elem = ET.SubElement(root, 'time')
        begin_elem = ET.SubElement(time_elem, 'begin')
        begin_elem.set('value', '0')
        end_elem = ET.SubElement(time_elem, 'end')
        end_elem.set('value', str(duration))
        
        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space='  ')
        tree.write(config_file, encoding='utf-8', xml_declaration=True)
        
        logger.info(f"Configuration file created: {config_file}")
        return str(config_file)


def main():
    parser = argparse.ArgumentParser(description='Generate SUMO traffic demand')
    parser.add_argument('--network', '-n', required=True, help='SUMO network file')
    parser.add_argument('--output-dir', '-o', default='sumo_network', help='Output directory')
    parser.add_argument('--duration', '-d', type=int, default=3600, 
                       help='Simulation duration (seconds)')
    parser.add_argument('--vehicles-per-hour', '-v', type=int, default=1000,
                       help='Vehicles per hour')
    parser.add_argument('--traffic-data', '-t', help='Traffic data JSON file')
    
    args = parser.parse_args()
    
    generator = TrafficDemandGenerator(args.network, args.output_dir)
    
    if args.traffic_data:
        # Generate from traffic data
        route_file = generator.create_from_traffic_data(args.traffic_data)
    else:
        # Generate with specified parameters
        route_file = generator.create_route_file(
            duration=args.duration,
            vehicles_per_hour=args.vehicles_per_hour
        )
    
    # Create config file
    config_file = generator.create_config_file(route_file, args.duration)
    
    print(f"\nTraffic demand generated:")
    print(f"  Route file: {route_file}")
    print(f"  Config file: {config_file}")
    print(f"\nTo run simulation:")
    print(f"  sumo-gui -c {config_file}")


if __name__ == "__main__":
    main()
