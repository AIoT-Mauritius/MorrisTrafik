"""
Integration Script for Running SUMO Simulation with AI Agents
This script ties together the LSTM Congestion Predictor (PCA) and the DQN Junction Agent (LJA)
to run an adaptive traffic light control simulation.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import modules from the project
from sumo_simulation.simulation_runner import SUMOSimulation
from models.lstm_congestion_predictor import LSTMCongestionPredictor
from models.dqn_junction_agent import DQNJunctionAgent, SUMOEnvironment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration (Should be loaded from config files in a real scenario) ---
SUMO_CFG_FILE = "config/simulation.sumocfg"
JUNCTION_ID = "center"  # Assuming a single junction for simplicity
LSTM_MODEL_DIR = "models/lstm_pca"
DQN_MODEL_DIR = "models/dqn_lja"
TRAFFIC_DATA_CSV = "data/processed/traffic_features.csv"
# ---------------------------------------------------------------------------

class AdaptiveTrafficControlSystem:
    """
    Manages the integrated simulation and control loop
    """
    def __init__(self, sumo_cfg, junction_id, lstm_model_dir, dqn_model_dir, traffic_data_csv, gui=False):
        self.sumo_cfg = sumo_cfg
        self.junction_id = junction_id
        self.gui = gui
        
        # Load LSTM Predictor (PCA)
        self.pca = LSTMCongestionPredictor()
        try:
            self.pca.load(lstm_model_dir)
            logger.info("PCA (LSTM) model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load PCA model: {e}. Training a dummy model.")
            self._train_dummy_pca(traffic_data_csv)
        
        # Initialize DQN Agent (LJA)
        # Note: DQN agent is typically trained offline. Here we assume it's loaded.
        # If not found, we'll initialize a new one (which will need training).
        self.lja_env = SUMOEnvironment(sumo_cfg, junction_id, gui=False)
        self.lja_env.start_simulation() # Start once to get state/action sizes
        
        self.lja = DQNJunctionAgent(state_size=self.lja_env.state_size, action_size=self.lja_env.action_size)
        try:
            self.lja.load(os.path.join(dqn_model_dir, "dqn_model.h5"))
            logger.info("LJA (DQN) model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load LJA model: {e}. Agent is untrained.")
        
        self.lja_env.close() # Close the initial environment
        
        # Simulation Runner
        self.sim = SUMOSimulation(sumo_cfg, gui=gui)
        
        # Historical data for PCA prediction
        self.historical_data = pd.read_csv(traffic_data_csv)
        self.historical_data['timestamp'] = pd.to_datetime(self.historical_data['timestamp'])
        self.historical_data = self.historical_data.set_index('timestamp')
        self.historical_data['hour'] = self.historical_data.index.hour
        self.historical_data['day_of_week'] = self.historical_data.index.dayofweek
        
        # Buffer for real-time data collection (to feed PCA)
        self.real_time_buffer = deque(maxlen=self.pca.sequence_length)
        
        # Simulation step counter for control frequency
        self.control_frequency = 5 # Control decision every 5 steps (seconds)
        self.pca_prediction_frequency = 60 # Predict every 60 steps (seconds)
        self.last_pca_prediction = 0.0
        
    def _train_dummy_pca(self, traffic_data_csv):
        """Trains a minimal PCA model if loading fails, for demonstration purposes."""
        try:
            df = pd.read_csv(traffic_data_csv)
            X, y = self.pca.prepare_data(df)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
            
            self.pca.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            self.pca.train(X_train, y_train, X_val, y_val, epochs=1, batch_size=32)
            logger.info("Dummy PCA model trained successfully.")
        except Exception as e:
            logger.error(f"Failed to train dummy PCA model: {e}. PCA will return 0.0.")
            
    def _get_real_time_features(self, sim_manager: SUMOSimulation) -> Dict:
        """
        Extracts features from the current simulation step to feed the PCA.
        In a real scenario, this would come from the video_processor.
        """
        # Placeholder for real-time data extraction logic
        # For now, we'll use the simulation state as a proxy
        junction_state = sim_manager.get_junction_state(self.junction_id)
        
        # Calculate proxy for flow rate and avg speed
        # This is a simplification. In reality, this data comes from OpenDataCam.
        total_queue = junction_state.get('total_queue', 0)
        active_vehicles = sim_manager.get_state().get('active_vehicles', 0)
        
        # Mock features based on simulation state
        flow_rate_proxy = active_vehicles / 10.0 # Arbitrary scaling
        avg_speed_proxy = 10.0 - (total_queue / 5.0) # Inverse relationship
        
        # Get time features
        current_time = datetime.now() # Using real time as a proxy for simulation time
        
        return {
            'flow_rate': flow_rate_proxy,
            'avg_speed': avg_speed_proxy,
            'hour': current_time.hour,
            'day_of_week': current_time.weekday()
        }

    def control_callback(self, sim_manager: SUMOSimulation, state: Dict):
        """
        Callback function executed at every simulation step.
        This is where the AI agents make decisions.
        """
        current_step = sim_manager.current_step
        
        # 1. PCA Prediction (Less frequent)
        if current_step % self.pca_prediction_frequency == 0:
            # Get real-time features and update buffer
            real_time_features = self._get_real_time_features(sim_manager)
            self.real_time_buffer.append(real_time_features)
            
            # Predict CRS if buffer is full
            if len(self.real_time_buffer) == self.pca.sequence_length:
                recent_df = pd.DataFrame(list(self.real_time_buffer))
                try:
                    self.last_pca_prediction = self.pca.predict_from_history(recent_df)
                    logger.info(f"Step {current_step}: Predicted CRS = {self.last_pca_prediction:.4f}")
                except Exception as e:
                    logger.error(f"PCA prediction failed: {e}")
                    self.last_pca_prediction = 0.0 # Fallback
        
        # 2. LJA Control (More frequent)
        if current_step % self.control_frequency == 0:
            # Get current state for DQN
            # The state includes the predicted CRS from PCA
            current_state_vector = self.lja_env.get_state(predicted_crs=self.last_pca_prediction)
            
            # LJA decides action
            action = self.lja.act(current_state_vector, training=False)
            
            # Execute action (0: extend, 1: change phase)
            if action == 1:
                # Change phase
                current_phase = sim_manager.get_junction_state(self.junction_id).get('phase', 0)
                next_phase = (current_phase + 1) % len(sim_manager.traffic_lights) # Simplified phase logic
                sim_manager.set_traffic_light_phase(self.junction_id, next_phase)
                logger.debug(f"Step {current_step}: LJA chose to CHANGE phase to {next_phase}")
            else:
                # Extend phase (do nothing, as the simulation step already advances time)
                logger.debug(f"Step {current_step}: LJA chose to EXTEND phase")
                pass # The simulation runner handles the time step
    
    def run_simulation(self, max_steps):
        """Run the integrated simulation"""
        logger.info(f"Starting integrated simulation for {max_steps} steps...")
        
        # Run the simulation with the control callback
        final_stats = self.sim.run(max_steps=max_steps, agent_callback=self.control_callback)
        
        return final_stats


def main():
    parser = argparse.ArgumentParser(description='Run Adaptive Traffic Control Simulation')
    parser.add_argument('--sumo-cfg', default=SUMO_CFG_FILE, help='Path to SUMO config file')
    parser.add_argument('--junction-id', default=JUNCTION_ID, help='ID of the junction to control')
    parser.add_argument('--lstm-dir', default=LSTM_MODEL_DIR, help='Directory for LSTM model')
    parser.add_argument('--dqn-dir', default=DQN_MODEL_DIR, help='Directory for DQN model')
    parser.add_argument('--data-csv', default=TRAFFIC_DATA_CSV, help='Traffic data CSV for PCA training')
    parser.add_argument('--max-steps', type=int, default=3600, help='Maximum simulation steps')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    
    args = parser.parse_args()
    
    # Check for SUMO config file existence (placeholder check)
    if not Path(args.sumo_cfg).exists():
        logger.error(f"SUMO config file not found at {args.sumo_cfg}. Please run network generation first.")
        # Exit gracefully or create a dummy config for demonstration
        # For now, we'll assume the user has run the necessary scripts to create this file
        # In a real scenario, we would run the network generation script here.
        return

    # Initialize and run the system
    system = AdaptiveTrafficControlSystem(
        sumo_cfg=args.sumo_cfg,
        junction_id=args.junction_id,
        lstm_model_dir=args.lstm_dir,
        dqn_model_dir=args.dqn_dir,
        traffic_data_csv=args.data_csv,
        gui=args.gui
    )
    
    final_stats = system.run_simulation(max_steps=args.max_steps)
    
    print("\n" + "="*50)
    print("ADAPTIVE TRAFFIC CONTROL SIMULATION COMPLETE")
    print("="*50)
    for key, value in final_stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    # This is a placeholder for the actual main execution logic.
    # The user needs to run the network generation and model training scripts first.
    print("Please run network generation and model training scripts before running this integration script.")
    print("Example: python scripts/generate_sumo_network.py --location caudan")
    print("Example: python models/lstm_congestion_predictor.py --data data/processed/traffic_features.csv --output models/lstm_pca")
    print("Example: python models/dqn_junction_agent.py --sumo-cfg config/simulation.sumocfg --junction-id center --episodes 100 --output models/dqn_lja")
    
    # For a full run, the user would execute:
    # python scripts/run_ai_simulation.py --sumo-cfg config/simulation.sumocfg --junction-id center
    pass
