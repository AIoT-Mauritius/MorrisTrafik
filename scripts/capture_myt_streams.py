#!/usr/bin/env python3
"""
Automated Video Capture Script for my.t Traffic Watch Streams
This script captures live video from my.t CCTV feeds for specified durations
"""

import subprocess
import argparse
import os
from datetime import datetime
import time
import signal
import sys


class StreamCapture:
    """
    Class to handle video stream capture from my.t Traffic Watch
    """
    
    # Available stream URLs
    STREAMS = {
        'caudan_north': 'https://stream.myt.mu/prod/CAUDAN_NORTH.stream_720p/playlist.m3u8',
        'caudan_south': 'https://stream.myt.mu/prod/CAUDAN_SOUTH.stream_720p/playlist.m3u8',
        'la_chaussee': 'https://stream.myt.mu/prod/LA_CHAUSSEE_STREET.stream_720p/playlist.m3u8',
        'place_darmes': 'https://stream.myt.mu/prod/PLACE_DARMES.stream_720p/playlist.m3u8',
        'to_city_center': 'https://stream.myt.mu/prod/CASERNES_CITY_CENTRE.stream_720p/playlist.m3u8',
        'casernes_police': 'https://stream.myt.mu/prod/CASERNES_BRABANT_STREET.stream_720p/playlist.m3u8'
    }
    
    def __init__(self, output_dir='../data/raw'):
        """
        Initialize the stream capture
        
        Args:
            output_dir: Directory to save captured videos
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.processes = []
        
        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully"""
        print("\n\n⚠️  Received shutdown signal. Stopping captures...")
        self.stop_all()
        sys.exit(0)
    
    def capture_stream(self, stream_name, duration_minutes=60, output_filename=None):
        """
        Capture a single stream for specified duration
        
        Args:
            stream_name: Name of the stream (from STREAMS dict)
            duration_minutes: Duration to capture in minutes
            output_filename: Custom output filename (optional)
            
        Returns:
            Path to captured video file
        """
        if stream_name not in self.STREAMS:
            raise ValueError(f"Unknown stream: {stream_name}. Available: {list(self.STREAMS.keys())}")
        
        stream_url = self.STREAMS[stream_name]
        
        # Generate output filename
        if output_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"{stream_name}_{timestamp}.mp4"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Calculate duration in seconds
        duration_seconds = duration_minutes * 60
        
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-i', stream_url,
            '-t', str(duration_seconds),
            '-c', 'copy',
            '-y',  # Overwrite output file if exists
            output_path
        ]
        
        print(f"\n{'='*60}")
        print(f"Starting capture: {stream_name}")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Output: {output_path}")
        print(f"{'='*60}\n")
        
        # Start capture
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        self.processes.append(process)
        
        # Wait for completion
        try:
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                print(f"\n✓ Capture completed: {output_path}")
                
                # Get file size
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"  File size: {file_size_mb:.2f} MB")
                
                return output_path
            else:
                print(f"\n✗ Capture failed for {stream_name}")
                print(f"Error: {stderr}")
                return None
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Capture interrupted for {stream_name}")
            process.terminate()
            process.wait()
            return output_path if os.path.exists(output_path) else None
    
    def capture_multiple_streams(self, stream_names, duration_minutes=60):
        """
        Capture multiple streams simultaneously
        
        Args:
            stream_names: List of stream names to capture
            duration_minutes: Duration to capture in minutes
            
        Returns:
            Dictionary of stream_name -> output_path
        """
        results = {}
        processes = []
        
        # Start all captures
        for stream_name in stream_names:
            if stream_name not in self.STREAMS:
                print(f"⚠️  Skipping unknown stream: {stream_name}")
                continue
            
            stream_url = self.STREAMS[stream_name]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"{stream_name}_{timestamp}.mp4"
            output_path = os.path.join(self.output_dir, output_filename)
            
            duration_seconds = duration_minutes * 60
            
            cmd = [
                'ffmpeg',
                '-i', stream_url,
                '-t', str(duration_seconds),
                '-c', 'copy',
                '-y',
                output_path
            ]
            
            print(f"Starting capture: {stream_name} -> {output_path}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            processes.append({
                'name': stream_name,
                'process': process,
                'output_path': output_path
            })
        
        # Wait for all to complete
        print(f"\n⏳ Capturing {len(processes)} streams for {duration_minutes} minutes...")
        print("Press Ctrl+C to stop early\n")
        
        try:
            for p in processes:
                stdout, stderr = p['process'].communicate()
                
                if p['process'].returncode == 0:
                    file_size_mb = os.path.getsize(p['output_path']) / (1024 * 1024)
                    print(f"✓ {p['name']}: {file_size_mb:.2f} MB")
                    results[p['name']] = p['output_path']
                else:
                    print(f"✗ {p['name']}: Failed")
                    results[p['name']] = None
                    
        except KeyboardInterrupt:
            print("\n⚠️  Stopping all captures...")
            for p in processes:
                p['process'].terminate()
                p['process'].wait()
                if os.path.exists(p['output_path']):
                    results[p['name']] = p['output_path']
        
        return results
    
    def capture_scheduled(self, stream_name, duration_minutes=60, interval_minutes=None, repeat_count=1):
        """
        Capture stream on a schedule
        
        Args:
            stream_name: Name of the stream to capture
            duration_minutes: Duration of each capture in minutes
            interval_minutes: Interval between captures (None = continuous)
            repeat_count: Number of times to repeat (None = infinite)
            
        Returns:
            List of captured video paths
        """
        captured_files = []
        count = 0
        
        while repeat_count is None or count < repeat_count:
            print(f"\n{'='*60}")
            print(f"Scheduled Capture {count + 1}" + (f"/{repeat_count}" if repeat_count else ""))
            print(f"{'='*60}")
            
            output_path = self.capture_stream(stream_name, duration_minutes)
            
            if output_path:
                captured_files.append(output_path)
            
            count += 1
            
            if interval_minutes and (repeat_count is None or count < repeat_count):
                wait_time = interval_minutes * 60
                print(f"\n⏳ Waiting {interval_minutes} minutes until next capture...")
                time.sleep(wait_time)
        
        return captured_files
    
    def stop_all(self):
        """Stop all running capture processes"""
        for process in self.processes:
            if process.poll() is None:  # Process still running
                process.terminate()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description='Capture video from my.t Traffic Watch streams')
    parser.add_argument('--stream', choices=list(StreamCapture.STREAMS.keys()), 
                       help='Stream to capture')
    parser.add_argument('--streams', nargs='+', choices=list(StreamCapture.STREAMS.keys()),
                       help='Multiple streams to capture simultaneously')
    parser.add_argument('--duration', type=int, default=60,
                       help='Capture duration in minutes (default: 60)')
    parser.add_argument('--output-dir', default='../data/raw',
                       help='Output directory for captured videos')
    parser.add_argument('--list-streams', action='store_true',
                       help='List available streams')
    parser.add_argument('--scheduled', action='store_true',
                       help='Enable scheduled capture mode')
    parser.add_argument('--interval', type=int,
                       help='Interval between scheduled captures in minutes')
    parser.add_argument('--repeat', type=int,
                       help='Number of times to repeat scheduled capture')
    
    args = parser.parse_args()
    
    # List streams
    if args.list_streams:
        print("\nAvailable streams:")
        for name, url in StreamCapture.STREAMS.items():
            print(f"  {name}: {url}")
        return
    
    # Initialize capture
    capture = StreamCapture(output_dir=args.output_dir)
    
    # Capture multiple streams
    if args.streams:
        results = capture.capture_multiple_streams(args.streams, args.duration)
        print(f"\n{'='*60}")
        print("Capture Summary:")
        for name, path in results.items():
            status = "✓" if path else "✗"
            print(f"  {status} {name}: {path}")
        print(f"{'='*60}\n")
    
    # Capture single stream
    elif args.stream:
        if args.scheduled:
            captured_files = capture.capture_scheduled(
                args.stream,
                duration_minutes=args.duration,
                interval_minutes=args.interval,
                repeat_count=args.repeat
            )
            print(f"\n✓ Scheduled capture completed. {len(captured_files)} files captured.")
        else:
            output_path = capture.capture_stream(args.stream, args.duration)
            if output_path:
                print(f"\n✓ Video saved to: {output_path}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
