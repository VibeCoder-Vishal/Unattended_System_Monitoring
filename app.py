import cv2
import argparse
import os
import time
from monitoring import UnattendedMonitorDetector

def process_video(input_path, output_path, show_preview=True):
    """
    Process video for unattended monitor detection
    
    Args:
        input_path: Path to input video file
        output_path: Path to save output video
        show_preview: Whether to show preview window
    """
    # Check if input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file: {input_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize detector
    detector = UnattendedMonitorDetector()
    
    # Track statistics
    frame_count = 0
    unattended_frames = 0
    max_unattended = 0
    processing_times = []
    
    print(f"Processing video: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    
    try:
        while cap.isOpened():
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            start_time = time.time()
            unattended_monitors, total_active, annotated_frame = detector.detect_unattended_monitors(frame)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Update statistics
            if unattended_monitors:
                unattended_frames += 1
                max_unattended = max(max_unattended, len(unattended_monitors))
            
            # Write output frame
            out.write(annotated_frame)
            
            # Show preview
            if show_preview:
                cv2.imshow('Unattended Monitor Detection', annotated_frame)
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Print progress every 50 frames
            if frame_count % 50 == 0:
                progress = frame_count / total_frames * 100
                print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    # Print final statistics
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    unattended_percentage = (unattended_frames / frame_count) * 100 if frame_count > 0 else 0
    
    print("\nProcessing complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Average processing time per frame: {avg_processing_time:.3f} seconds")
    print(f"Frames with unattended monitors: {unattended_frames} ({unattended_percentage:.1f}%)")
    print(f"Maximum unattended monitors detected: {max_unattended}")
    print(f"Output saved to: {output_path}")

def main():
    """Main function to parse arguments and start processing"""
    parser = argparse.ArgumentParser(description='Unattended Monitor Detection with YOLOv8')
    parser.add_argument('--input', '-i', required=True, help='Path to input video file')
    parser.add_argument('--output', '-o', required=True, help='Path to save output video')
    parser.add_argument('--no-preview', action='store_true', help='Disable preview window')
    
    args = parser.parse_args()
    
    try:
        process_video(args.input, args.output, not args.no_preview)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())