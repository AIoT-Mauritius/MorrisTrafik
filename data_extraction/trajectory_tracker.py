"""
Trajectory Tracker using DeepSort-like algorithm
Tracks vehicles across frames and maintains trajectory history
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging


@dataclass
class Track:
    """Data class for vehicle track"""
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    trajectory: List[Tuple[int, int]] = field(default_factory=list)
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)
    
    def update(self, bbox: Tuple[int, int, int, int], center: Tuple[int, int]):
        """Update track with new detection"""
        # Calculate velocity
        if len(self.trajectory) > 0:
            prev_center = self.trajectory[-1]
            self.velocity = (
                center[0] - prev_center[0],
                center[1] - prev_center[1]
            )
        
        self.bbox = bbox
        self.center = center
        self.trajectory.append(center)
        self.hits += 1
        self.time_since_update = 0
        self.age += 1
    
    def predict(self):
        """Predict next position based on velocity"""
        self.time_since_update += 1
        self.age += 1
        
        # Simple linear prediction
        predicted_center = (
            int(self.center[0] + self.velocity[0]),
            int(self.center[1] + self.velocity[1])
        )
        return predicted_center
    
    def to_dict(self) -> Dict:
        """Convert track to dictionary"""
        return {
            'track_id': self.track_id,
            'class_name': self.class_name,
            'bbox': self.bbox,
            'center': self.center,
            'trajectory': self.trajectory,
            'age': self.age,
            'hits': self.hits,
            'velocity': self.velocity
        }


class TrajectoryTracker:
    """
    Multi-object tracker for vehicles
    Uses IoU-based matching and Kalman-like prediction
    """
    
    def __init__(self, 
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3):
        """
        Initialize trajectory tracker
        
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum hits before track is confirmed
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks: List[Track] = []
        self.next_id = 1
        self.frame_count = 0
        
        # Statistics
        self.total_tracks = 0
        self.active_tracks = 0
    
    def update(self, detections: List) -> List[Track]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of Detection objects from vehicle_detector
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to tracks
        matched, unmatched_dets, unmatched_tracks = self._match_detections(detections)
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            detection = detections[det_idx]
            self.tracks[track_idx].update(detection.bbox, detection.center)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            self._create_track(detection)
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
        
        # Return confirmed tracks
        confirmed_tracks = [t for t in self.tracks if t.hits >= self.min_hits]
        self.active_tracks = len(confirmed_tracks)
        
        return confirmed_tracks
    
    def _match_detections(self, detections: List) -> Tuple[List, List, List]:
        """
        Match detections to existing tracks using IoU
        
        Returns:
            matched: List of (track_idx, detection_idx) pairs
            unmatched_detections: List of unmatched detection indices
            unmatched_tracks: List of unmatched track indices
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for t, track in enumerate(self.tracks):
            for d, detection in enumerate(detections):
                iou_matrix[t, d] = self._calculate_iou(track.bbox, detection.bbox)
        
        # Greedy matching
        matched = []
        unmatched_tracks = []
        unmatched_detections = list(range(len(detections)))
        
        for t in range(len(self.tracks)):
            if len(unmatched_detections) == 0:
                unmatched_tracks.append(t)
                continue
            
            # Find best matching detection
            best_iou = 0
            best_det = -1
            
            for d in unmatched_detections:
                if iou_matrix[t, d] > best_iou:
                    best_iou = iou_matrix[t, d]
                    best_det = d
            
            if best_iou >= self.iou_threshold:
                matched.append((t, best_det))
                unmatched_detections.remove(best_det)
            else:
                unmatched_tracks.append(t)
        
        return matched, unmatched_detections, unmatched_tracks
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                       bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _create_track(self, detection):
        """Create new track from detection"""
        track = Track(
            track_id=self.next_id,
            class_name=detection.class_name,
            bbox=detection.bbox,
            center=detection.center,
            trajectory=[detection.center]
        )
        
        self.tracks.append(track)
        self.next_id += 1
        self.total_tracks += 1
    
    def get_statistics(self) -> Dict:
        """Get tracker statistics"""
        return {
            'frame_count': self.frame_count,
            'total_tracks': self.total_tracks,
            'active_tracks': self.active_tracks,
            'tracks': [t.to_dict() for t in self.tracks if t.hits >= self.min_hits]
        }
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """
        Draw tracks on frame
        
        Args:
            frame: Input frame
            tracks: List of tracks to draw
            
        Returns:
            Annotated frame
        """
        import cv2
        
        annotated_frame = frame.copy()
        
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            
            # Draw bounding box
            color = self._get_color_for_id(track.track_id)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            label = f"ID:{track.track_id} {track.class_name}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw trajectory
            if len(track.trajectory) > 1:
                for i in range(1, len(track.trajectory)):
                    pt1 = track.trajectory[i - 1]
                    pt2 = track.trajectory[i]
                    cv2.line(annotated_frame, pt1, pt2, color, 2)
            
            # Draw velocity vector
            if track.velocity != (0.0, 0.0):
                end_point = (
                    int(track.center[0] + track.velocity[0] * 3),
                    int(track.center[1] + track.velocity[1] * 3)
                )
                cv2.arrowedLine(annotated_frame, track.center, end_point, 
                              color, 2, tipLength=0.3)
        
        return annotated_frame
    
    def _get_color_for_id(self, track_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for track ID"""
        np.random.seed(track_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color


def main():
    """Test the trajectory tracker"""
    import sys
    import cv2
    from vehicle_detector import VehicleDetector
    
    if len(sys.argv) < 2:
        print("Usage: python trajectory_tracker.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Initialize detector and tracker
    detector = VehicleDetector(confidence_threshold=0.5)
    tracker = TrajectoryTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        sys.exit(1)
    
    frame_count = 0
    
    print("Processing video with tracking... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect vehicles
        detections = detector.detect(frame)
        
        # Update tracker
        tracks = tracker.update(detections)
        
        # Draw tracks
        annotated_frame = tracker.draw_tracks(frame, tracks)
        
        # Display statistics
        stats = tracker.get_statistics()
        cv2.putText(annotated_frame, f"Active Tracks: {stats['active_tracks']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Total Tracks: {stats['total_tracks']}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Vehicle Tracking', annotated_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    stats = tracker.get_statistics()
    print(f"\nTracking complete!")
    print(f"Total frames: {frame_count}")
    print(f"Total tracks created: {stats['total_tracks']}")
    print(f"Active tracks at end: {stats['active_tracks']}")


if __name__ == "__main__":
    main()
