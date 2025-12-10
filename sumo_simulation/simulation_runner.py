"""
SUMO Simulation Runner with TraCI
Runs traffic simulation and provides interface for AI agents
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import SUMO TraCI
try:
    import traci
    import traci.constants as tc
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False
    logger.warning("TraCI not available. Install SUMO and add tools to PYTHONPATH")


class SUMOSimulation:
    """
    SUMO simulation manager with TraCI interface
    """
    
    def __init__(self, config_file: str, gui: bool = False):
        """
        Initialize simulation
        
        Args:
            config_file: Path to SUMO config file
            gui: Use SUMO GUI instead of command-line
        """
        self.config_file = Path(config_file)
        self.gui = gui
        self.running = False
        
        # Simulation state
        self.current_step = 0
        self.vehicles = {}
        self.junctions = []
        self.traffic_lights = []
        
        # Statistics
        self.total_vehicles = 0
        self.completed_vehicles = 0
        self.total_waiting_time = 0
        self.total_travel_time = 0
    
    def start(self) -> bool:
        """Start SUMO simulation"""
        if not TRACI_AVAILABLE:
            logger.error("TraCI not available")
            return False
        
        if not self.config_file.exists():
            logger.error(f"Config file not found: {self.config_file}")
            return False
        
        # Determine SUMO binary
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        
        # Start TraCI
        try:
            traci.start([sumo_binary, "-c", str(self.config_file),
                        "--waiting-time-memory", "100",
                        "--time-to-teleport", "-1"])
            
            self.running = True
            
            # Get network info
            self.junctions = traci.junction.getIDList()
            self.traffic_lights = traci.trafficlight.getIDList()
            
            logger.info(f"Simulation started with {len(self.traffic_lights)} traffic lights")
            return True
        
        except Exception as e:
            logger.error(f"Failed to start simulation: {e}")
            return False
    
    def step(self) -> bool:
        """
        Execute one simulation step
        
        Returns:
            True if simulation continues, False if ended
        """
        if not self.running:
            return False
        
        try:
            traci.simulationStep()
            self.current_step += 1
            
            # Update vehicle information
            self._update_vehicles()
            
            # Check if simulation ended
            if traci.simulation.getMinExpectedNumber() <= 0:
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error in simulation step: {e}")
            return False
    
    def _update_vehicles(self):
        """Update vehicle information"""
        # Get current vehicles
        current_vehicles = traci.vehicle.getIDList()
        
        # Add new vehicles
        for veh_id in current_vehicles:
            if veh_id not in self.vehicles:
                self.vehicles[veh_id] = {
                    'id': veh_id,
                    'start_time': self.current_step,
                    'waiting_time': 0,
                    'travel_time': 0
                }
                self.total_vehicles += 1
        
        # Update existing vehicles
        for veh_id in list(self.vehicles.keys()):
            if veh_id in current_vehicles:
                waiting_time = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                self.vehicles[veh_id]['waiting_time'] = waiting_time
                self.vehicles[veh_id]['travel_time'] = self.current_step - self.vehicles[veh_id]['start_time']
            else:
                # Vehicle completed
                self.completed_vehicles += 1
                self.total_waiting_time += self.vehicles[veh_id]['waiting_time']
                self.total_travel_time += self.vehicles[veh_id]['travel_time']
                del self.vehicles[veh_id]
    
    def get_state(self) -> Dict:
        """
        Get current simulation state
        
        Returns:
            Dictionary with simulation state
        """
        state = {
            'step': self.current_step,
            'time': self.current_step,  # 1 step = 1 second
            'active_vehicles': len(self.vehicles),
            'total_vehicles': self.total_vehicles,
            'completed_vehicles': self.completed_vehicles
        }
        
        # Add traffic light states
        if TRACI_AVAILABLE and self.running:
            tl_states = {}
            for tl_id in self.traffic_lights:
                tl_states[tl_id] = {
                    'state': traci.trafficlight.getRedYellowGreenState(tl_id),
                    'phase': traci.trafficlight.getPhase(tl_id),
                    'next_switch': traci.trafficlight.getNextSwitch(tl_id)
                }
            state['traffic_lights'] = tl_states
        
        return state
    
    def get_junction_state(self, junction_id: str) -> Dict:
        """
        Get state of specific junction
        
        Args:
            junction_id: Junction/traffic light ID
            
        Returns:
            Junction state dictionary
        """
        if not TRACI_AVAILABLE or not self.running:
            return {}
        
        try:
            # Get lanes controlled by this traffic light
            controlled_lanes = traci.trafficlight.getControlledLanes(junction_id)
            
            # Get queue lengths
            queue_lengths = {}
            for lane in controlled_lanes:
                queue_length = traci.lane.getLastStepHaltingNumber(lane)
                queue_lengths[lane] = queue_length
            
            # Get traffic light state
            tl_state = traci.trafficlight.getRedYellowGreenState(junction_id)
            tl_phase = traci.trafficlight.getPhase(junction_id)
            
            return {
                'junction_id': junction_id,
                'traffic_light_state': tl_state,
                'phase': tl_phase,
                'queue_lengths': queue_lengths,
                'total_queue': sum(queue_lengths.values())
            }
        
        except Exception as e:
            logger.error(f"Error getting junction state: {e}")
            return {}
    
    def set_traffic_light_phase(self, junction_id: str, phase: int):
        """
        Set traffic light phase
        
        Args:
            junction_id: Traffic light ID
            phase: Phase number to set
        """
        if not TRACI_AVAILABLE or not self.running:
            return
        
        try:
            traci.trafficlight.setPhase(junction_id, phase)
        except Exception as e:
            logger.error(f"Error setting traffic light phase: {e}")
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics"""
        stats = {
            'total_steps': self.current_step,
            'total_vehicles': self.total_vehicles,
            'completed_vehicles': self.completed_vehicles,
            'active_vehicles': len(self.vehicles),
            'total_waiting_time': self.total_waiting_time,
            'total_travel_time': self.total_travel_time
        }
        
        if self.completed_vehicles > 0:
            stats['avg_waiting_time'] = self.total_waiting_time / self.completed_vehicles
            stats['avg_travel_time'] = self.total_travel_time / self.completed_vehicles
        else:
            stats['avg_waiting_time'] = 0
            stats['avg_travel_time'] = 0
        
        return stats
    
    def run(self, max_steps: Optional[int] = None, 
            agent_callback: Optional[callable] = None) -> Dict:
        """
        Run complete simulation
        
        Args:
            max_steps: Maximum steps to run (None for unlimited)
            agent_callback: Callback function for AI agent (called each step)
            
        Returns:
            Final statistics
        """
        if not self.start():
            return {}
        
        logger.info("Running simulation...")
        
        step_count = 0
        
        try:
            while self.step():
                step_count += 1
                
                # Call agent if provided
                if agent_callback:
                    state = self.get_state()
                    agent_callback(self, state)
                
                # Check max steps
                if max_steps and step_count >= max_steps:
                    logger.info(f"Reached maximum steps: {max_steps}")
                    break
                
                # Progress update
                if step_count % 100 == 0:
                    logger.info(f"Step {step_count}: {len(self.vehicles)} active vehicles")
        
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        
        finally:
            self.stop()
        
        stats = self.get_statistics()
        logger.info(f"Simulation complete: {stats['completed_vehicles']} vehicles")
        
        return stats
    
    def stop(self):
        """Stop simulation"""
        if self.running and TRACI_AVAILABLE:
            try:
                traci.close()
                self.running = False
                logger.info("Simulation stopped")
            except Exception as e:
                logger.error(f"Error stopping simulation: {e}")
    
    def save_statistics(self, output_file: str):
        """Save statistics to JSON file"""
        stats = self.get_statistics()
        
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Run SUMO traffic simulation')
    parser.add_argument('--config', '-c', required=True, help='SUMO config file')
    parser.add_argument('--gui', '-g', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--max-steps', '-m', type=int, help='Maximum simulation steps')
    parser.add_argument('--output', '-o', help='Output file for statistics')
    
    args = parser.parse_args()
    
    # Create simulation
    sim = SUMOSimulation(args.config, gui=args.gui)
    
    # Run simulation
    stats = sim.run(max_steps=args.max_steps)
    
    # Print statistics
    print("\n" + "="*50)
    print("SIMULATION STATISTICS")
    print("="*50)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Save statistics
    if args.output:
        sim.save_statistics(args.output)


if __name__ == "__main__":
    main()
