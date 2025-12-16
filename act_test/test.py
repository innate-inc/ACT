#!/usr/bin/env python3
"""
Test script for HDF5 to WebDataset conversion pipeline.
"""

import os
import sys
from pathlib import Path

# Add the data_tools directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_tools'))

from data_tools.webdataset import convert_hdf5_to_webdataset

def test_conversion_pipeline():
    """Test the HDF5 to WebDataset conversion pipeline."""
    
    # Configuration
    HDF5_DIR = "/home/vignesh/raid/PaperMulti2"
    OUTPUT_DIR = "/home/vignesh/raid/PaperMulti2_webd"
    SHARD_SIZE = 1000  # Samples per shard
    
    print("🧪 TESTING HDF5 TO WEBDATASET CONVERSION PIPELINE")
    print("=" * 60)
    print(f"📁 Input HDF5 directory: {HDF5_DIR}")
    print(f"📁 Output WebDataset directory: {OUTPUT_DIR}")
    print(f"📦 Shard size: {SHARD_SIZE}")
    print()
    
    # Check if input directory exists
    if not os.path.exists(HDF5_DIR):
        print(f"❌ Error: Input directory does not exist: {HDF5_DIR}")
        return False
    
    # Check if dataset_metadata.json exists
    metadata_path = os.path.join(HDF5_DIR, "dataset_metadata.json")
    if not os.path.exists(metadata_path):
        print(f"❌ Error: dataset_metadata.json not found in {HDF5_DIR}")
        print("Please ensure your HDF5 directory contains a dataset_metadata.json file")
        return False
    
    print(f"✅ Found dataset_metadata.json: {metadata_path}")
    
    # List some HDF5 files to verify structure
    h5_files = [f for f in os.listdir(HDF5_DIR) if f.endswith('.h5')]
    print(f"📊 Found {len(h5_files)} HDF5 files in directory")
    if h5_files:
        print(f"📄 Example files: {h5_files[:5]}")  # Show first 5 files
    print()
    
    # Run the conversion
    try:
        success = convert_hdf5_to_webdataset(
            hdf5_directory=HDF5_DIR,
            webd_directory=OUTPUT_DIR,
            shard_size=SHARD_SIZE
        )
        
        if success:
            print("\n🎉 CONVERSION TEST SUCCESSFUL!")
            
            # Verify output
            if os.path.exists(OUTPUT_DIR):
                tar_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.tar')]
                json_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')]
                
                print(f"📦 Created {len(tar_files)} WebDataset shard files")
                print(f"📄 Created {len(json_files)} metadata files")
                
                if tar_files:
                    print(f"🗂️  Example shards: {tar_files[:3]}")
                if json_files:
                    print(f"📋 Metadata files: {json_files}")
                
                # Check total size
                total_size = 0
                for f in os.listdir(OUTPUT_DIR):
                    file_path = os.path.join(OUTPUT_DIR, f)
                    if os.path.isfile(file_path):
                        total_size += os.path.getsize(file_path)
                
                total_size_gb = total_size / (1024**3)
                print(f"💾 Total output size: {total_size_gb:.2f} GB")
            
            return True
        else:
            print("\n❌ CONVERSION TEST FAILED!")
            return False
            
    except Exception as e:
        print(f"\n💥 CONVERSION TEST CRASHED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_webdataset_loading():
    """Test loading the converted WebDataset."""
    
    OUTPUT_DIR = "/home/vignesh/raid/PaperMulti2_webd"
    
    print("\n🔍 TESTING WEBDATASET LOADING")
    print("=" * 40)
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"❌ WebDataset directory not found: {OUTPUT_DIR}")
        print("Please run conversion test first.")
        return False
    
    try:
        # Import webdataset and test basic loading
        import webdataset as wds
        import torch
        
        # Find tar files
        tar_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.tar')]
        if not tar_files:
            print(f"❌ No .tar files found in {OUTPUT_DIR}")
            return False
        
        # Create a simple dataset
        dataset_pattern = os.path.join(OUTPUT_DIR, "train-*.tar")
        print(f"📂 Dataset pattern: {dataset_pattern}")
        
        dataset = (
            wds.WebDataset(dataset_pattern)
            .decode("pil")
            .to_tuple("cam1.png", "cam2.png", "qpos.npy", "actions.npy")
        )
        
        # Test loading a few samples
        print("🔄 Loading test samples...")
        sample_count = 0
        for sample in dataset:
            cam1, cam2, qpos, actions = sample
            
            print(f"Sample {sample_count + 1}:")
            print(f"  📷 Camera 1: {cam1.size if hasattr(cam1, 'size') else type(cam1)}")
            print(f"  📷 Camera 2: {cam2.size if hasattr(cam2, 'size') else type(cam2)}")
            print(f"  🎯 QPos shape: {qpos.shape}")
            print(f"  🎮 Actions shape: {actions.shape}")
            
            sample_count += 1
            if sample_count >= 3:  # Test first 3 samples
                break
        
        print(f"✅ Successfully loaded {sample_count} samples from WebDataset")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install webdataset: pip install webdataset")
        return False
    except Exception as e:
        print(f"❌ Loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 STARTING DATA CONVERSION PIPELINE TESTS")
    print("=" * 60)
    
    # Test 1: Conversion
    conversion_success = test_conversion_pipeline()
    
    if conversion_success:
        # Test 2: Loading
        loading_success = test_webdataset_loading()
        
        if loading_success:
            print("\n🎉 ALL TESTS PASSED!")
            print("Your data conversion pipeline is working correctly.")
        else:
            print("\n⚠️  CONVERSION PASSED BUT LOADING FAILED")
            print("Check WebDataset installation and file integrity.")
    else:
        print("\n❌ CONVERSION TEST FAILED")
        print("Please check your input data and fix any issues.")

if __name__ == "__main__":
    main()
