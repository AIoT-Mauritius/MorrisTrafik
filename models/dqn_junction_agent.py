#!/usr/bin/env python3
"""
DQN-based Local Junction Agent (LJA)
This module implements a Deep Q-Network for adaptive traffic light control
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
import traci
import os
import argparse


class DQNJunctionAgent:
    """
    Deep Q-Network agent for traffic light control
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Initialize the DQN agent
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Experience replay buffer
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        
        # Q-Networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """
        Build the neural network for Q-value estimation
        
        Returns:
            Keras model
        """
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_model(self):
        """Update target network weights with current model weights"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (uses epsilon-greedy)
            
        Returns:
            Selected action
        """
        if training and np.random.rand() <= self.epsilon:
            # Explore: random action
            return random.randrange(self.action_size)
        
        # Exploit: best action according to Q-network
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """
        Train the network using experience replay
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random minibatch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Predict Q-values for current states
        current_q_values = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next states using target network
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update Q-values using Bellman equation
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        self.model.fit(states, current_q_values, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, model_path):
        """
        Save the model
        
        Args:
            model_path: Path to save the model
        """
        self.model.save(model_path)
        print(f"✓ Model saved to {model_path}")
    
    def load(self, model_path):
        """
        Load the model
        
        Args:
            model_path: Path to load the model from
        """
        self.model = keras.models.load_model(model_path)
        self.update_target_model()
        print(f"✓ Model loaded from {model_path}")


class SUMOEnvironment:
    """
    SUMO simulation environment for traffic light control
    """
    
    def __init__(self, sumo_cfg, junction_id, gui=False):
        """
        Initialize SUMO environment
        
        Args:
            sumo_cfg: Path to SUMO configuration file
            junction_id: ID of the junction to control
            gui: Whether to use SUMO GUI
        """
        self.sumo_cfg = sumo_cfg
        self.junction_id = junction_id
        self.gui = gui
        
        # Traffic light phases
        self.phases = None
        self.current_phase = 0
        
        # State and action spaces
        self.state_size = None
        self.action_size = 2  # Extend or change phase
    
    def start_simulation(self):
        """Start SUMO simulation"""
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        sumo_cmd = [sumo_binary, "-c", self.sumo_cfg, "--start"]
        
        traci.start(sumo_cmd)
        
        # Get traffic light information
        self.phases = traci.trafficlight.getAllProgramLogics(self.junction_id)[0].phases
        
        # Calculate state size (queue lengths for all lanes + current phase + predicted CRS)
        lanes = traci.trafficlight.getControlledLanes(self.junction_id)
        self.state_size = len(set(lanes)) + 2  # unique lanes + phase + CRS
        
        print(f"✓ SUMO simulation started")
        print(f"  Junction: {self.junction_id}")
        print(f"  State size: {self.state_size}")
        print(f"  Action size: {self.action_size}")
    
    def get_state(self, predicted_crs=0.0):
        """
        Get current state of the junction
        
        Args:
            predicted_crs: Predicted congestion risk score from LSTM
            
        Returns:
            State vector
        """
        # Get controlled lanes
        lanes = traci.trafficlight.getControlledLanes(self.junction_id)
        unique_lanes = list(set(lanes))
        
        # Get queue lengths for each lane
        queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in unique_lanes]
        
        # Get current phase
        current_phase = traci.trafficlight.getPhase(self.junction_id)
        
        # Construct state vector
        state = queue_lengths + [current_phase, predicted_crs]
        
        return np.array(state)
    
    def get_reward(self):
        """
        Calculate reward based on current traffic conditions
        
        Returns:
            Reward value (negative of total waiting time and queue length)
        """
        # Get controlled lanes
        lanes = traci.trafficlight.getControlledLanes(self.junction_id)
        unique_lanes = list(set(lanes))
        
        # Calculate total queue length
        total_queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in unique_lanes)
        
        # Calculate total waiting time
        total_waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in unique_lanes)
        
        # Reward is negative of congestion (we want to minimize it)
        reward = -(total_queue + total_waiting_time * 0.1)
        
        return reward
    
    def step(self, action):
        """
        Execute action in the environment
        
        Args:
            action: 0 = extend current phase, 1 = change to next phase
            
        Returns:
            next_state, reward, done
        """
        if action == 0:
            # Extend current phase by 5 seconds
            extension_steps = 5
            for _ in range(extension_steps):
                traci.simulationStep()
        else:
            # Change to next phase
            current_phase = traci.trafficlight.getPhase(self.junction_id)
            next_phase = (current_phase + 1) % len(self.phases)
            traci.trafficlight.setPhase(self.junction_id, next_phase)
            traci.simulationStep()
        
        # Get next state and reward
        next_state = self.get_state()
        reward = self.get_reward()
        
        # Check if simulation is done
        done = traci.simulation.getMinExpectedNumber() <= 0
        
        return next_state, reward, done
    
    def reset(self):
        """
        Reset the simulation
        
        Returns:
            Initial state
        """
        traci.load(["-c", self.sumo_cfg, "--start"])
        return self.get_state()
    
    def close(self):
        """Close SUMO simulation"""
        traci.close()


def train_dqn_agent(sumo_cfg, junction_id, episodes=100, gui=False, model_output_dir="../models/dqn_lja"):
    """
    Train DQN agent for traffic light control
    
    Args:
        sumo_cfg: Path to SUMO configuration file
        junction_id: ID of the junction to control
        episodes: Number of training episodes
        gui: Whether to use SUMO GUI
        model_output_dir: Directory to save trained model
    """
    # Initialize environment
    env = SUMOEnvironment(sumo_cfg, junction_id, gui)
    env.start_simulation()
    
    # Initialize agent
    agent = DQNJunctionAgent(state_size=env.state_size, action_size=env.action_size)
    
    # Training loop
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Select and perform action
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            agent.replay()
            
            # Update state and metrics
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Update target network every 10 episodes
        if episode % 10 == 0:
            agent.update_target_model()
        
        scores.append(total_reward)
        
        print(f"Episode {episode + 1}/{episodes} - Score: {total_reward:.2f} - "
              f"Epsilon: {agent.epsilon:.4f} - Steps: {steps}")
    
    # Save trained model
    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, "dqn_model.h5")
    agent.save(model_path)
    
    # Close environment
    env.close()
    
    print(f"\n✓ Training completed. Model saved to {model_output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train DQN Junction Agent')
    parser.add_argument('--sumo-cfg', required=True, help='Path to SUMO configuration file')
    parser.add_argument('--junction-id', required=True, help='ID of the junction to control')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--output', default='../models/dqn_lja', help='Output directory for model')
    
    args = parser.parse_args()
    
    train_dqn_agent(
        sumo_cfg=args.sumo_cfg,
        junction_id=args.junction_id,
        episodes=args.episodes,
        gui=args.gui,
        model_output_dir=args.output
    )


if __name__ == "__main__":
    main()
