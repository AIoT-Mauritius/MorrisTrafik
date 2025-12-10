"""
Vehicle Detection Module using YOLO
Detects vehicles, pedestrians, and other road users in video frames
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging

# Try to import ultralytics YOLO, fallback to basic implementation
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics YOLO not available. Install with: pip install ultralytics")


@dataclass
class Detection:
    """Data class for vehicle detection"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    
    def to_dict(self) -> Dict:
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'center': self.center
        }


class VehicleDetector:
    """
    Vehicle detector using YOLO for real-time object detection
    Supports multiple vehicle types: car, truck, bus, motorcycle, bicycle
    """
    
    # COCO dataset class IDs for vehicles
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
        1: 'bicycle',
        0: 'person'  # For pedestrian detection
    }
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        """
        Initialize vehicle detector
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                logging.info(f"Loaded YOLO model: {model_path}")
            except Exception as e:
                logging.error(f"Failed to load YOLO model: {e}")
        else:
            logging.warning("YOLO not available. Using dummy detector.")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect vehicles in a single frame
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of Detection objects
        """
        if self.model is None:
            return self._dummy_detect(frame)
        
        detections = []
        
        try:
            # Run YOLO inference
            results = self.model(frame, verbose=False)[0]
            
            # Process detections
            for box in results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Filter by vehicle classes and confidence
                if class_id in self.VEHICLE_CLASSES and confidence >= self.confidence_threshold:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Calculate center point
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    detection = Detection(
                        class_id=class_id,
                        class_name=self.VEHICLE_CLASSES[class_id],
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        center=(center_x, center_y)
                    )
                    
                    detections.append(detection)
        
        except Exception as e:
            logging.error(f"Detection error: {e}")
        
        return detections
    
    def _dummy_detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Dummy detector for testing when YOLO is not available
        Returns empty list
        """
        return []
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Detect vehicles in multiple frames (batch processing)
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detection lists (one per frame)
        """
        return [self.detect(frame) for frame in frames]
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Color map for different vehicle types
        color_map = {
            'car': (0, 255, 0),      # Green
            'truck': (0, 0, 255),    # Red
            'bus': (255, 0, 0),      # Blue
            'motorcycle': (255, 255, 0),  # Cyan
            'bicycle': (255, 0, 255),     # Magenta
            'person': (0, 255, 255)       # Yellow
        }
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = color_map.get(det.class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw center point
            cv2.circle(annotated_frame, det.center, 3, color, -1)
        
        return annotated_frame
    
    def count_by_type(self, detections: List[Detection]) -> Dict[str, int]:
        """
        Count detections by vehicle type
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary with counts per vehicle type
        """
        counts = {}
        for det in detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts


def main():
    """Test the vehicle detector"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vehicle_detector.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Initialize detector
    detector = VehicleDetector(confidence_threshold=0.5)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        sys.exit(1)
    
    frame_count = 0
    total_detections = 0
    
    print("Processing video... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect vehicles
        detections = detector.detect(frame)
        total_detections += len(detections)
        
        # Draw detections
        annotated_frame = detector.draw_detections(frame, detections)
        
        # Display counts
        counts = detector.count_by_type(detections)
        y_offset = 30
        for vehicle_type, count in counts.items():
            text = f"{vehicle_type}: {count}"
            cv2.putText(annotated_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        # Display frame info
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Vehicle Detection', annotated_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessing complete!")
    print(f"Total frames: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per frame: {total_detections / frame_count:.2f}")


if __name__ == "__main__":
    main()
