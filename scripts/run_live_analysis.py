#!/usr/bin/env python3
"""
End-to-End Live Traffic Analysis Script
This script automates the complete workflow:
1. Capture video from my.t streams
2. Process with OpenDataCam (or direct ffmpeg if ODC not available)
3. Extract traffic data
4. Generate visualizations
5. Optionally train AI models
"""

import subprocess
import argparse
import os
import time
import json
from datetime import datetime
import sys


class LiveTrafficAnalysis:
    """
    Orchestrates the complete live traffic analysis pipeline
    """
    
    def __init__(self, stream_name, output_base_dir='../data'):
        """
        Initialize the analysis pipeline
        
        Args:
            stream_name: Name of the stream to analyze
            output_base_dir: Base directory for all outputs
        """
        self.stream_name = stream_name
        self.output_base_dir = output_base_dir
        
        # Create directory structure
        self.raw_dir = os.path.join(output_base_dir, 'raw')
        self.processed_dir = os.path.join(output_base_dir, 'processed')
        self.figures_dir = os.path.join(output_base_dir, '../docs/figures')
        
        for d in [self.raw_dir, self.processed_dir, self.figures_dir]:
            os.makedirs(d, exist_ok=True)
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_id = f"{stream_name}_{self.timestamp}"
    
    def step1_capture_video(self, duration_minutes=60):
        """
        Step 1: Capture video from live stream
        
        Args:
            duration_minutes: Duration to capture
            
        Returns:
            Path to captured video file
        """
        print(f"\n{'='*70}")
        print("STEP 1: Capturing Live Video")
        print(f"{'='*70}")
        
        cmd = [
            'python3',
            'capture_myt_streams.py',
            '--stream', self.stream_name,
            '--duration', str(duration_minutes),
            '--output-dir', self.raw_dir
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Find the most recent video file
            video_files = [f for f in os.listdir(self.raw_dir) if f.startswith(self.stream_name) and f.endswith('.mp4')]
            if video_files:
                video_files.sort(reverse=True)
                video_path = os.path.join(self.raw_dir, video_files[0])
                print(f"✓ Video captured: {video_path}")
                return video_path
            else:
                print("✗ No video file found")
                return None
        else:
            print(f"✗ Capture failed: {result.stderr}")
            return None
    
    def step2_check_opendatacam(self):
        """
        Step 2: Check if OpenDataCam is available
        
        Returns:
            True if OpenDataCam is running
        """
        print(f"\n{'='*70}")
        print("STEP 2: Checking OpenDataCam Status")
        print(f"{'='*70}")
        
        try:
            result = subprocess.run(
                ['docker', 'ps'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if 'opendatacam' in result.stdout:
                print("✓ OpenDataCam is running")
                return True
            else:
                print("⚠ OpenDataCam is not running")
                print("  You can start it with: docker start opendatacam")
                return False
                
        except Exception as e:
            print(f"⚠ Could not check OpenDataCam status: {e}")
            return False
    
    def step3_process_video(self, video_path, use_opendatacam=True):
        """
        Step 3: Process video to extract traffic data
        
        Args:
            video_path: Path to video file
            use_opendatacam: Whether to use OpenDataCam (if False, uses alternative method)
            
        Returns:
            Paths to flow_rate and speed CSV files
        """
        print(f"\n{'='*70}")
        print("STEP 3: Processing Video")
        print(f"{'='*70}")
        
        if use_opendatacam:
            print("Using OpenDataCam for processing...")
            print("⚠ Manual step required:")
            print("  1. Open http://localhost:8080")
            print("  2. Upload the video file")
            print("  3. Define counting lines")
            print("  4. Run analysis")
            print("  5. Export data")
            print("\nAfter exporting data, run extract_traffic_data.py")
            
            input("\nPress Enter when OpenDataCam processing is complete...")
            
            # Run extraction script
            cmd = [
                'python3',
                'extract_traffic_data.py',
                '--url', 'http://localhost:8080',
                '--output', self.processed_dir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Data extracted from OpenDataCam")
            else:
                print(f"⚠ Extraction had issues: {result.stderr}")
        
        else:
            print("⚠ OpenDataCam not available")
            print("Alternative: Use pre-recorded data or manual processing")
        
        # Find generated CSV files
        csv_files = [f for f in os.listdir(self.processed_dir) if f.endswith('.csv')]
        flow_rate_file = None
        speed_file = None
        
        for f in csv_files:
            if 'flow_rate' in f:
                flow_rate_file = os.path.join(self.processed_dir, f)
            elif 'speed' in f:
                speed_file = os.path.join(self.processed_dir, f)
        
        return flow_rate_file, speed_file
    
    def step4_visualize_data(self, flow_rate_file, speed_file):
        """
        Step 4: Generate visualizations
        
        Args:
            flow_rate_file: Path to flow rate CSV
            speed_file: Path to speed CSV
        """
        print(f"\n{'='*70}")
        print("STEP 4: Generating Visualizations")
        print(f"{'='*70}")
        
        if not flow_rate_file or not speed_file:
            print("⚠ Missing data files, skipping visualization")
            return
        
        cmd = [
            'python3',
            'visualize_traffic_data.py',
            '--flow-rate', flow_rate_file,
            '--speed', speed_file,
            '--output', self.figures_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Visualizations generated")
            print(f"  Output directory: {self.figures_dir}")
        else:
            print(f"✗ Visualization failed: {result.stderr}")
    
    def step5_generate_sumo_data(self, flow_rate_file, speed_file):
        """
        Step 5: Generate SUMO simulation data
        
        Args:
            flow_rate_file: Path to flow rate CSV
            speed_file: Path to speed CSV
        """
        print(f"\n{'='*70}")
        print("STEP 5: Generating SUMO Simulation Data")
        print(f"{'='*70}")
        
        if not flow_rate_file or not speed_file:
            print("⚠ Missing data files, skipping SUMO data generation")
            return
        
        sumo_dir = os.path.join(self.output_base_dir, 'sumo')
        os.makedirs(sumo_dir, exist_ok=True)
        
        cmd = [
            'python3',
            'transform_data_for_sumo.py',
            '--flow-rate', flow_rate_file,
            '--speed', speed_file,
            '--output', sumo_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ SUMO data generated")
            print(f"  Output directory: {sumo_dir}")
        else:
            print(f"✗ SUMO data generation failed: {result.stderr}")
    
    def generate_report(self, video_path, flow_rate_file, speed_file):
        """
        Generate a summary report of the analysis session
        
        Args:
            video_path: Path to captured video
            flow_rate_file: Path to flow rate CSV
            speed_file: Path to speed CSV
        """
        print(f"\n{'='*70}")
        print("Generating Analysis Report")
        print(f"{'='*70}")
        
        report = {
            'session_id': self.session_id,
            'stream_name': self.stream_name,
            'timestamp': self.timestamp,
            'video_file': video_path,
            'flow_rate_file': flow_rate_file,
            'speed_file': speed_file,
            'figures_dir': self.figures_dir
        }
        
        report_path = os.path.join(self.processed_dir, f'analysis_report_{self.session_id}.json')
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Report saved: {report_path}")
        
        # Print summary
        print(f"\n{'='*70}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"Session ID: {self.session_id}")
        print(f"Stream: {self.stream_name}")
        print(f"Video: {video_path}")
        if flow_rate_file:
            print(f"Flow Rate Data: {flow_rate_file}")
        if speed_file:
            print(f"Speed Data: {speed_file}")
        print(f"Figures: {self.figures_dir}")
        print(f"{'='*70}\n")
    
    def run_full_pipeline(self, duration_minutes=60, skip_capture=False, video_path=None):
        """
        Run the complete analysis pipeline
        
        Args:
            duration_minutes: Duration to capture video
            skip_capture: Skip video capture and use existing video
            video_path: Path to existing video (if skip_capture=True)
        """
        print(f"\n{'#'*70}")
        print(f"# LIVE TRAFFIC ANALYSIS PIPELINE")
        print(f"# Stream: {self.stream_name}")
        print(f"# Session: {self.session_id}")
        print(f"{'#'*70}\n")
        
        # Step 1: Capture video
        if not skip_capture:
            video_path = self.step1_capture_video(duration_minutes)
            if not video_path:
                print("\n✗ Pipeline failed at Step 1: Video capture")
                return
        else:
            if not video_path or not os.path.exists(video_path):
                print(f"\n✗ Video file not found: {video_path}")
                return
            print(f"\n✓ Using existing video: {video_path}")
        
        # Step 2: Check OpenDataCam
        odc_available = self.step2_check_opendatacam()
        
        # Step 3: Process video
        flow_rate_file, speed_file = self.step3_process_video(video_path, use_opendatacam=odc_available)
        
        # Step 4: Visualize
        if flow_rate_file and speed_file:
            self.step4_visualize_data(flow_rate_file, speed_file)
            
            # Step 5: Generate SUMO data
            self.step5_generate_sumo_data(flow_rate_file, speed_file)
        
        # Generate report
        self.generate_report(video_path, flow_rate_file, speed_file)


def main():
    parser = argparse.ArgumentParser(description='Run live traffic analysis pipeline')
    parser.add_argument('--stream', required=True,
                       choices=['caudan_north', 'caudan_south', 'la_chaussee', 
                               'place_darmes', 'to_city_center', 'casernes_police'],
                       help='Stream to analyze')
    parser.add_argument('--duration', type=int, default=60,
                       help='Capture duration in minutes (default: 60)')
    parser.add_argument('--skip-capture', action='store_true',
                       help='Skip video capture and use existing video')
    parser.add_argument('--video', help='Path to existing video file (if --skip-capture)')
    parser.add_argument('--output-dir', default='../data',
                       help='Base output directory')
    
    args = parser.parse_args()
    
    # Change to scripts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Initialize pipeline
    pipeline = LiveTrafficAnalysis(args.stream, args.output_dir)
    
    # Run pipeline
    pipeline.run_full_pipeline(
        duration_minutes=args.duration,
        skip_capture=args.skip_capture,
        video_path=args.video
    )


if __name__ == "__main__":
    main()
