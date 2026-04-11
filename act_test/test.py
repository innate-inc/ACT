#!/usr/bin/env python3
"""
Test script for HDF5 to WebDataset conversion pipeline.

Designed to run on Lambda Labs instances for debugging.
"""

import os
import sys
import argparse

# Fix import path - use absolute import
from act_test.data_tools.webdataset import convert_to_webdataset



def test_conversion_pipeline(hdf5_dir: str, output_dir: str, shard_size: int = 500):
    """Test the HDF5 to WebDataset conversion pipeline."""
    
    print("🧪 TESTING HDF5 TO WEBDATASET CONVERSION PIPELINE")
    print("=" * 60)
    print(f"📁 Input HDF5 directory: {hdf5_dir}")
    print(f"📁 Output WebDataset directory: {output_dir}")
    print(f"📦 Shard size: {shard_size}")
    print()
    
    # Check if input directory exists
    if not os.path.exists(hdf5_dir):
        print(f"❌ Error: Input directory does not exist: {hdf5_dir}")
        return False
    
    # Check for metadata file (support both names)
    metadata_path = os.path.join(hdf5_dir, "metadata.json")
    dataset_metadata_path = os.path.join(hdf5_dir, "dataset_metadata.json")
    
    if os.path.exists(metadata_path):
        print(f"✅ Found metadata.json: {metadata_path}")
    elif os.path.exists(dataset_metadata_path):
        print(f"✅ Found dataset_metadata.json: {dataset_metadata_path}")
    else:
        print(f"❌ Error: No metadata file found in {hdf5_dir}")
        print("   Expected: metadata.json or dataset_metadata.json")
        return False
    
    # List some HDF5 files to verify structure
    h5_files = [f for f in os.listdir(hdf5_dir) if f.endswith('.h5')]
    print(f"📊 Found {len(h5_files)} HDF5 files in directory")
    if h5_files:
        print(f"📄 Example files: {h5_files[:5]}")  # Show first 5 files
    print()
    
    # Run the conversion
    try:
        success = convert_to_webdataset(
            data_directory=hdf5_dir,
            webd_directory=output_dir,
            shard_size=shard_size
        )
        
        if success:
            print("\n🎉 CONVERSION TEST SUCCESSFUL!")
            
            # Verify output
            if os.path.exists(output_dir):
                tar_files = [f for f in os.listdir(output_dir) if f.endswith('.tar')]
                json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
                
                print(f"📦 Created {len(tar_files)} WebDataset shard files")
                print(f"📄 Created {len(json_files)} metadata files")
                
                if tar_files:
                    print(f"🗂️  Example shards: {tar_files[:3]}")
                if json_files:
                    print(f"📋 Metadata files: {json_files}")
                
                # Check total size
                total_size = 0
                for f in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, f)
                    if os.path.isfile(file_path):
                        total_size += os.path.getsize(file_path)
                
                total_size_mb = total_size / (1024**2)
                print(f"💾 Total output size: {total_size_mb:.2f} MB")
            
            return True
        else:
            print("\n❌ CONVERSION TEST FAILED!")
            return False
            
    except Exception as e:
        print(f"\n💥 CONVERSION TEST CRASHED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_webdataset_loading(output_dir: str):
    """Test loading the converted WebDataset."""
    
    print("\n🔍 TESTING WEBDATASET LOADING")
    print("=" * 40)
    
    if not os.path.exists(output_dir):
        print(f"❌ WebDataset directory not found: {output_dir}")
        print("Please run conversion test first.")
        return False
    
    try:
        # Import webdataset and test basic loading
        import webdataset as wds
        import torch
        
        # Find tar files
        tar_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.tar')])
        if not tar_files:
            print(f"❌ No .tar files found in {output_dir}")
            return False
        
        print(f"📦 Found {len(tar_files)} shard files")
        
        # Create dataset pattern
        first_file = tar_files[0]
        last_file = tar_files[-1]
        
        import re
        match_first = re.search(r'train-(\d+)\.tar', first_file)
        match_last = re.search(r'train-(\d+)\.tar', last_file)
        
        if match_first and match_last:
            start_num = int(match_first.group(1))
            end_num = int(match_last.group(1))
            dataset_pattern = os.path.join(output_dir, f"train-{{{start_num:05d}..{end_num:05d}}}.tar")
        else:
            dataset_pattern = os.path.join(output_dir, "train-*.tar")
        
        print(f"📂 Dataset pattern: {dataset_pattern}")
        
        # Create dataset - note: images are stored as .pth (PyTorch tensors)
        dataset = (
            wds.WebDataset(dataset_pattern)
            .decode("torch")
            .to_tuple("cam1.pth", "cam2.pth", "qpos.pth", "actions.pth")
        )
        
        # Test loading a few samples
        print("🔄 Loading test samples...")
        sample_count = 0
        for sample in dataset:
            cam1, cam2, qpos, actions = sample
            
            print(f"Sample {sample_count + 1}:")
            print(f"  📷 Camera 1: {cam1.shape if hasattr(cam1, 'shape') else type(cam1)}")
            print(f"  📷 Camera 2: {cam2.shape if hasattr(cam2, 'shape') else type(cam2)}")
            print(f"  🎯 QPos shape: {qpos.shape}")
            print(f"  🎮 Actions shape: {actions.shape}")
            
            # Check for NaNs
            if torch.isnan(cam1).any():
                print(f"  ⚠️  WARNING: cam1 contains NaN values!")
            if torch.isnan(cam2).any():
                print(f"  ⚠️  WARNING: cam2 contains NaN values!")
            if torch.isnan(qpos).any():
                print(f"  ⚠️  WARNING: qpos contains NaN values!")
            if torch.isnan(actions).any():
                print(f"  ⚠️  WARNING: actions contains NaN values!")
            
            sample_count += 1
            if sample_count >= 5:  # Test first 5 samples
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
    parser = argparse.ArgumentParser(description="Test HDF5 to WebDataset conversion pipeline")
    parser.add_argument(
        "--local-dir",
        type=str,
        default="/data/test_dataset",
        help="Local directory to store downloaded data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/test_dataset/webdataset",
        help="Output directory for WebDataset files"
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=500,
        help="Number of samples per shard"
    )
    args = parser.parse_args()
    
    print("🚀 STARTING DATA CONVERSION PIPELINE TESTS")
    print("=" * 60)
    print(f"📁 Local Dir: {args.local_dir}")
    print(f"📁 Output Dir: {args.output_dir}")
    print(f"📦 Shard Size: {args.shard_size}")
    print()
    
    # Step 1: Test conversion
    
    conversion_success = test_conversion_pipeline(
        hdf5_dir=args.local_dir,
        output_dir=args.output_dir,
        shard_size=args.shard_size
    )
    
    if conversion_success:
        # Step 3: Test loading
        loading_success = test_webdataset_loading(args.output_dir)
        
        if loading_success:
            print("\n🎉 ALL TESTS PASSED!")
            print("Your data conversion pipeline is working correctly.")
            sys.exit(0)
        else:
            print("\n⚠️  CONVERSION PASSED BUT LOADING FAILED")
            print("Check WebDataset installation and file integrity.")
            sys.exit(1)
    else:
        print("\n❌ CONVERSION TEST FAILED")
        print("Please check your input data and fix any issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
