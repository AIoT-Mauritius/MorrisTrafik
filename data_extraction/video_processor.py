"""
Video Processor for Traffic Data Extraction
Processes CCTV videos and extracts traffic flow data
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

from vehicle_detector import VehicleDetector
from trajectory_tracker import TrajectoryTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VirtualLine:
    """Virtual counting line for traffic flow measurement"""
    
    def __init__(self, name: str, start: Tuple[int, int], end: Tuple[int, int], 
                 direction: str = "both"):
        """
        Initialize virtual line
        
        Args:
            name: Line identifier
            start: Start point (x, y)
            end: End point (x, y)
            direction: Counting direction ("up", "down", "left", "right", "both")
        """
        self.name = name
        self.start = start
        self.end = end
        self.direction = direction
        self.crossed_tracks = set()
        self.count = 0
        self.count_by_type = defaultdict(int)
    
    def check_crossing(self, track_id: int, prev_center: Tuple[int, int], 
                      curr_center: Tuple[int, int], vehicle_type: str) -> bool:
        """
        Check if a track crossed the line
        
        Returns:
            True if crossing detected
        """
        if track_id in self.crossed_tracks:
            return False
        
        # Check line crossing using cross product
        crossed = self._line_intersection(prev_center, curr_center, self.start, self.end)
        
        if crossed:
            self.crossed_tracks.add(track_id)
            self.count += 1
            self.count_by_type[vehicle_type] += 1
            return True
        
        return False
    
    def _line_intersection(self, p1: Tuple[int, int], p2: Tuple[int, int],
                          p3: Tuple[int, int], p4: Tuple[int, int]) -> bool:
        """Check if line segment p1-p2 intersects with p3-p4"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw the virtual line on frame"""
        cv2.line(frame, self.start, self.end, (0, 255, 255), 3)
        
        # Draw label
        mid_x = (self.start[0] + self.end[0]) // 2
        mid_y = (self.start[1] + self.end[1]) // 2
        label = f"{self.name}: {self.count}"
        cv2.putText(frame, label, (mid_x, mid_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def get_stats(self) -> Dict:
        """Get counting statistics"""
        return {
            'name': self.name,
            'total_count': self.count,
            'count_by_type': dict(self.count_by_type)
        }


class TrafficDataExtractor:
    """
    Main class for extracting traffic data from CCTV videos
    """
    
    def __init__(self, video_path: str, output_dir: str = "output"):
        """
        Initialize traffic data extractor
        
        Args:
            video_path: Path to input video
            output_dir: Directory for output files
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize detector and tracker
        self.detector = VehicleDetector(confidence_threshold=0.5)
        self.tracker = TrajectoryTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        
        # Virtual counting lines
        self.virtual_lines: List[VirtualLine] = []
        
        # Video properties
        self.cap = None
        self.fps = 0
        self.frame_width = 0
        self.frame_height = 0
        self.total_frames = 0
        
        # Data storage
        self.traffic_data = []
        self.frame_data = []
        
        # Processing state
        self.prev_tracks = {}
    
    def open_video(self) -> bool:
        """Open video file and get properties"""
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            logger.error(f"Cannot open video: {self.video_path}")
            return False
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video opened: {self.frame_width}x{self.frame_height} @ {self.fps} fps")
        logger.info(f"Total frames: {self.total_frames}")
        
        return True
    
    def add_virtual_line(self, name: str, start: Tuple[int, int], 
                        end: Tuple[int, int], direction: str = "both"):
        """Add a virtual counting line"""
        line = VirtualLine(name, start, end, direction)
        self.virtual_lines.append(line)
        logger.info(f"Added virtual line: {name}")
    
    def setup_default_lines(self):
        """Setup default counting lines based on frame size"""
        # Horizontal line in the middle
        mid_y = self.frame_height // 2
        self.add_virtual_line("horizontal_mid", (0, mid_y), (self.frame_width, mid_y))
        
        # Vertical line in the middle
        mid_x = self.frame_width // 2
        self.add_virtual_line("vertical_mid", (mid_x, 0), (mid_x, self.frame_height))
    
    def process_video(self, save_video: bool = True, show_display: bool = False) -> Dict:
        """
        Process entire video and extract traffic data
        
        Args:
            save_video: Save annotated video
            show_display: Show real-time display
            
        Returns:
            Processing statistics
        """
        if not self.open_video():
            return {}
        
        # Setup virtual lines if none defined
        if len(self.virtual_lines) == 0:
            self.setup_default_lines()
        
        # Setup video writer
        video_writer = None
        if save_video:
            output_video_path = self.output_dir / "annotated_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_video_path), fourcc, self.fps,
                                          (self.frame_width, self.frame_height))
        
        frame_count = 0
        start_time = datetime.now()
        
        logger.info("Starting video processing...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            annotated_frame = self._process_frame(frame, frame_count)
            
            # Save frame data
            if frame_count % int(self.fps) == 0:  # Every second
                self._save_frame_data(frame_count)
            
            # Write to output video
            if video_writer:
                video_writer.write(annotated_frame)
            
            # Display
            if show_display:
                cv2.imshow('Traffic Analysis', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / self.total_frames) * 100
                logger.info(f"Progress: {progress:.1f}% ({frame_count}/{self.total_frames})")
        
        # Cleanup
        self.cap.release()
        if video_writer:
            video_writer.release()
        if show_display:
            cv2.destroyAllWindows()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Save final data
        self._save_traffic_data()
        
        # Generate statistics
        stats = self._generate_statistics(frame_count, processing_time)
        
        logger.info("Processing complete!")
        logger.info(f"Processed {frame_count} frames in {processing_time:.2f} seconds")
        
        return stats
    
    def _process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Process a single frame"""
        # Detect vehicles
        detections = self.detector.detect(frame)
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        # Check line crossings
        for track in tracks:
            if track.track_id in self.prev_tracks:
                prev_center = self.prev_tracks[track.track_id]
                curr_center = track.center
                
                for line in self.virtual_lines:
                    line.check_crossing(track.track_id, prev_center, curr_center, 
                                      track.class_name)
            
            self.prev_tracks[track.track_id] = track.center
        
        # Draw annotations
        annotated_frame = self.tracker.draw_tracks(frame, tracks)
        
        # Draw virtual lines
        for line in self.virtual_lines:
            annotated_frame = line.draw(annotated_frame)
        
        # Draw statistics
        annotated_frame = self._draw_statistics(annotated_frame, frame_number, tracks)
        
        return annotated_frame
    
    def _draw_statistics(self, frame: np.ndarray, frame_number: int, 
                        tracks: List) -> np.ndarray:
        """Draw statistics overlay on frame"""
        # Frame info
        timestamp = frame_number / self.fps
        time_str = str(timedelta(seconds=int(timestamp)))
        cv2.putText(frame, f"Time: {time_str}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Active tracks
        cv2.putText(frame, f"Active Vehicles: {len(tracks)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Total tracks
        stats = self.tracker.get_statistics()
        cv2.putText(frame, f"Total Detected: {stats['total_tracks']}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def _save_frame_data(self, frame_number: int):
        """Save data for current frame"""
        timestamp = frame_number / self.fps
        stats = self.tracker.get_statistics()
        
        frame_data = {
            'frame': frame_number,
            'timestamp': timestamp,
            'active_tracks': stats['active_tracks'],
            'line_counts': [line.get_stats() for line in self.virtual_lines]
        }
        
        self.frame_data.append(frame_data)
    
    def _save_traffic_data(self):
        """Save all traffic data to JSON file"""
        output_file = self.output_dir / "traffic_data.json"
        
        data = {
            'video_info': {
                'path': self.video_path,
                'fps': self.fps,
                'resolution': f"{self.frame_width}x{self.frame_height}",
                'total_frames': self.total_frames
            },
            'virtual_lines': [line.get_stats() for line in self.virtual_lines],
            'tracker_stats': self.tracker.get_statistics(),
            'frame_data': self.frame_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Traffic data saved to: {output_file}")
    
    def _generate_statistics(self, frame_count: int, processing_time: float) -> Dict:
        """Generate final statistics"""
        stats = self.tracker.get_statistics()
        
        return {
            'frames_processed': frame_count,
            'processing_time': processing_time,
            'fps_processed': frame_count / processing_time if processing_time > 0 else 0,
            'total_vehicles_detected': stats['total_tracks'],
            'line_counts': [line.get_stats() for line in self.virtual_lines]
        }


def main():
    parser = argparse.ArgumentParser(description='Extract traffic data from CCTV video')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--save-video', action='store_true', help='Save annotated video')
    parser.add_argument('--display', action='store_true', help='Show real-time display')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = TrafficDataExtractor(args.input, args.output)
    
    # Process video
    stats = extractor.process_video(save_video=args.save_video, show_display=args.display)
    
    # Print statistics
    print("\n" + "="*50)
    print("TRAFFIC DATA EXTRACTION COMPLETE")
    print("="*50)
    print(f"Frames processed: {stats['frames_processed']}")
    print(f"Processing time: {stats['processing_time']:.2f} seconds")
    print(f"Processing FPS: {stats['fps_processed']:.2f}")
    print(f"Total vehicles detected: {stats['total_vehicles_detected']}")
    print("\nLine Counts:")
    for line_stats in stats['line_counts']:
        print(f"  {line_stats['name']}: {line_stats['total_count']}")
        for vtype, count in line_stats['count_by_type'].items():
            print(f"    {vtype}: {count}")


if __name__ == "__main__":
    main()
