"""
Data Extraction Module for MorrisTrafik
Provides vehicle detection, tracking, and traffic data extraction from CCTV videos
"""

from .vehicle_detector import VehicleDetector, Detection
from .trajectory_tracker import TrajectoryTracker, Track
from .video_processor import TrafficDataExtractor, VirtualLine

__all__ = [
    'VehicleDetector',
    'Detection',
    'TrajectoryTracker',
    'Track',
    'TrafficDataExtractor',
    'VirtualLine'
]
