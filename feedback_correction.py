"""
Feedback Correction System
--------------------------
Allows you to correct wrong predictions and retrain the model.

Usage:
    1. Save wrong predictions to feedback folder
    2. Manually correct the labels
    3. Retrain model with corrected data
"""

import os
import json
import numpy as np
import shutil
from datetime import datetime
from pathlib import Path

class FeedbackSystem:
    """System to collect and manage prediction feedback."""
    
    def __init__(self, feedback_dir="feedback_data"):
        """
        Initialize feedback system.
        
        Args:
            feedback_dir: Directory to store feedback data
        """
        self.feedback_dir = feedback_dir
        self.corrections_file = os.path.join(feedback_dir, "corrections.json")
        
        # Create directories
        os.makedirs(feedback_dir, exist_ok=True)
        os.makedirs(os.path.join(feedback_dir, "videos"), exist_ok=True)
        
        # Load existing corrections
        self.corrections = self._load_corrections()
    
    def _load_corrections(self):
        """Load existing corrections from file."""
        if os.path.exists(self.corrections_file):
            with open(self.corrections_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_corrections(self):
        """Save corrections to file."""
        with open(self.corrections_file, 'w') as f:
            json.dump(self.corrections, f, indent=2)
    
    def save_wrong_prediction(self, video_path, predicted_label, confidence):
        """
        Save a video with wrong prediction for later correction.
        
        Args:
            video_path: Path to the video file
            predicted_label: What the model predicted (0=normal, 1=distress)
            confidence: Prediction confidence
        
        Returns:
            feedback_id: Unique ID for this feedback entry
        """
        # Generate unique ID
        feedback_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Copy video to feedback directory
        video_name = os.path.basename(video_path)
        feedback_video_path = os.path.join(
            self.feedback_dir, 
            "videos", 
            f"{feedback_id}_{video_name}"
        )
        shutil.copy2(video_path, feedback_video_path)
        
        # Save metadata
        self.corrections[feedback_id] = {
            "video_path": feedback_video_path,
            "predicted_label": int(predicted_label),
            "predicted_class": "distress" if predicted_label == 1 else "normal",
            "confidence": float(confidence),
            "correct_label": None,  # To be filled manually
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        self._save_corrections()
        
        print(f"✓ Saved wrong prediction: {feedback_id}")
        print(f"  Video: {video_name}")
        print(f"  Predicted: {self.corrections[feedback_id]['predicted_class']}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"\nTo correct: Edit {self.corrections_file}")
        
        return feedback_id
    
    def mark_correction(self, feedback_id, correct_label):
        """
        Mark the correct label for a feedback entry.
        
        Args:
            feedback_id: ID of the feedback entry
            correct_label: Correct label (0=normal, 1=distress)
        """
        if feedback_id not in self.corrections:
            print(f"Error: Feedback ID {feedback_id} not found")
            return False
        
        self.corrections[feedback_id]["correct_label"] = int(correct_label)
        self.corrections[feedback_id]["correct_class"] = "distress" if correct_label == 1 else "normal"
        self.corrections[feedback_id]["status"] = "corrected"
        
        self._save_corrections()
        
        print(f"✓ Marked correction for {feedback_id}")
        print(f"  Correct label: {self.corrections[feedback_id]['correct_class']}")
        
        return True
    
    def get_pending_corrections(self):
        """Get list of pending corrections."""
        return {
            fid: data for fid, data in self.corrections.items()
            if data["status"] == "pending"
        }
    
    def get_corrected_data(self):
        """Get list of corrected data ready for retraining."""
        return {
            fid: data for fid, data in self.corrections.items()
            if data["status"] == "corrected"
        }
    
    def export_corrected_videos(self, output_dir="corrected_videos"):
        """
        Export corrected videos to organized folders for retraining.
        
        Args:
            output_dir: Output directory
        """
        os.makedirs(os.path.join(output_dir, "normal"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "distress"), exist_ok=True)
        
        corrected = self.get_corrected_data()
        
        if not corrected:
            print("No corrected data to export")
            return
        
        for fid, data in corrected.items():
            correct_class = data["correct_class"]
            video_path = data["video_path"]
            
            if os.path.exists(video_path):
                dest_dir = os.path.join(output_dir, correct_class)
                dest_path = os.path.join(dest_dir, os.path.basename(video_path))
                shutil.copy2(video_path, dest_path)
                print(f"✓ Exported {fid} to {correct_class}/")
        
        print(f"\n✓ Exported {len(corrected)} corrected videos to {output_dir}/")
        print("\nNext steps:")
        print(f"1. Add these videos to your training data:")
        print(f"   - Copy from {output_dir}/normal/ to videos/normal/")
        print(f"   - Copy from {output_dir}/distress/ to videos/distress/")
        print(f"2. Retrain the model:")
        print(f"   python prepare_training_data.py --normal_videos_dir ./videos/normal --distress_videos_dir ./videos/distress --output training_data_v2.npz")
        print(f"   python train_lstm.py --data_path training_data_v2.npz --epochs 50")
    
    def show_summary(self):
        """Show summary of feedback data."""
        total = len(self.corrections)
        pending = len(self.get_pending_corrections())
        corrected = len(self.get_corrected_data())
        
        print("=" * 60)
        print("FEEDBACK SYSTEM SUMMARY")
        print("=" * 60)
        print(f"Total feedback entries: {total}")
        print(f"  - Pending correction: {pending}")
        print(f"  - Corrected: {corrected}")
        print(f"\nFeedback directory: {self.feedback_dir}")
        print(f"Corrections file: {self.corrections_file}")


def main():
    """Interactive feedback correction tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Feedback correction system')
    parser.add_argument('--action', type=str, required=True,
                       choices=['save', 'correct', 'export', 'summary'],
                       help='Action to perform')
    parser.add_argument('--video', type=str, help='Video file path (for save)')
    parser.add_argument('--predicted', type=int, choices=[0, 1], help='Predicted label (for save)')
    parser.add_argument('--confidence', type=float, help='Prediction confidence (for save)')
    parser.add_argument('--feedback_id', type=str, help='Feedback ID (for correct)')
    parser.add_argument('--correct_label', type=int, choices=[0, 1], help='Correct label (for correct)')
    
    args = parser.parse_args()
    
    feedback = FeedbackSystem()
    
    if args.action == 'save':
        if not all([args.video, args.predicted is not None, args.confidence]):
            print("Error: --video, --predicted, and --confidence required for save")
            return
        feedback.save_wrong_prediction(args.video, args.predicted, args.confidence)
    
    elif args.action == 'correct':
        if not all([args.feedback_id, args.correct_label is not None]):
            print("Error: --feedback_id and --correct_label required for correct")
            return
        feedback.mark_correction(args.feedback_id, args.correct_label)
    
    elif args.action == 'export':
        feedback.export_corrected_videos()
    
    elif args.action == 'summary':
        feedback.show_summary()
        
        # Show pending corrections
        pending = feedback.get_pending_corrections()
        if pending:
            print("\n" + "=" * 60)
            print("PENDING CORRECTIONS")
            print("=" * 60)
            for fid, data in pending.items():
                print(f"\nID: {fid}")
                print(f"  Video: {data['video_path']}")
                print(f"  Predicted: {data['predicted_class']} ({data['confidence']:.2%})")
                print(f"  To correct, run:")
                print(f"    python feedback_correction.py --action correct --feedback_id {fid} --correct_label [0 or 1]")


if __name__ == "__main__":
    main()
