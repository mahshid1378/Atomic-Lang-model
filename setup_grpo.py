#!/usr/bin/env python3
"""
GRPO Setup Script
================

Install dependencies and prepare environment for GRPO testing.
"""

import subprocess
import sys
import platform
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required Python packages."""
    print("📦 Installing dependencies...")
    
    # Core dependencies
    core_packages = [
        "torch>=2.0.0",
        "transformers>=4.30.0", 
        "accelerate>=0.20.0",
        "peft>=0.4.0",
        "bitsandbytes>=0.39.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "datasets>=2.0.0",
        "evaluate>=0.4.0"
    ]
    
    # Optional packages
    optional_packages = [
        "jupyter",
        "tensorboard",
        "wandb"
    ]
    
    try:
        # Install core packages
        for package in core_packages:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
        
        print("✅ Core dependencies installed")
        
        # Try to install optional packages
        for package in optional_packages:
            try:
                print(f"Installing optional {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ])
            except subprocess.CalledProcessError:
                print(f"⚠️  Optional package {package} failed to install")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_gpu_availability():
    """Check GPU and CUDA availability."""
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            memory_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            
            print(f"✅ CUDA available")
            print(f"   GPU count: {gpu_count}")
            print(f"   Current GPU: {gpu_name}")
            print(f"   GPU memory: {memory_gb:.1f} GB")
            
            if memory_gb < 4:
                print("⚠️  Warning: Low GPU memory. Consider using CPU or smaller model.")
            
            return True
        else:
            print("⚠️  CUDA not available. Will use CPU (slower training).")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def setup_directories():
    """Create necessary directories."""
    directories = [
        "evaluation_results",
        "checkpoints",
        "logs"
    ]
    
    base_path = Path("atomic-lang-model/python")
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def create_requirements_file():
    """Create requirements.txt file."""
    requirements = """# GRPO Integration Dependencies
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
peft>=0.4.0
bitsandbytes>=0.39.0
numpy>=1.21.0
matplotlib>=3.5.0
datasets>=2.0.0
evaluate>=0.4.0

# Optional dependencies
jupyter
tensorboard
wandb
"""
    
    req_path = Path("requirements.txt")
    with open(req_path, 'w') as f:
        f.write(requirements)
    
    print(f"✅ Created {req_path}")

def main():
    """Main setup function."""
    print("🚀 GRPO Setup Script")
    print("=" * 30)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create requirements file
    create_requirements_file()
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Check GPU
    gpu_available = check_gpu_availability()
    
    # Setup directories
    setup_directories()
    
    print("\n" + "=" * 30)
    print("✅ Setup completed!")
    
    if gpu_available:
        print("\nNext steps:")
        print("1. cd atomic-lang-model/python")
        print("2. python test_grpo_integration.py")
    else:
        print("\nNext steps (CPU mode):")
        print("1. cd atomic-lang-model/python")
        print("2. python test_grpo_integration.py")
        print("   (Note: Training will be slower on CPU)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)