#!/usr/bin/env python3
"""Quick start script for semi-supervised image classification."""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print("Error:", e.stderr)
        return False


def main():
    """Main function to run the project."""
    print("🚀 Semi-supervised Image Classification - Quick Start")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Tests failed, but continuing...")
    
    # Train model
    print("\n🎯 Starting training...")
    print("This will train a semi-supervised model on CIFAR-10")
    print("Training will take some time depending on your hardware...")
    
    if not run_command("python scripts/train.py", "Training model"):
        print("❌ Training failed")
        sys.exit(1)
    
    # Evaluate model
    if not run_command("python scripts/train.py --eval-only --resume checkpoints/best_model.pth", "Evaluating model"):
        print("⚠️  Evaluation failed, but continuing...")
    
    # Launch demo
    print("\n🎉 Training completed successfully!")
    print("\n📱 To launch the interactive demo, run:")
    print("   streamlit run demo/streamlit_app.py")
    print("\n📊 To view analysis notebook, run:")
    print("   jupyter notebook notebooks/analysis.ipynb")
    
    print("\n✨ Project setup complete!")
    print("Check the 'logs/' directory for training logs and 'assets/' for visualizations.")


if __name__ == "__main__":
    main()
