# Data Loading Test Script

This script tests data loading performance and verifies data shuffling/uniqueness between GPUs, similar to how `train_dist.py` loads data.

## Features

- **Timing Analysis**: Measures batch loading, data transfer, and total processing times
- **Data Verification**: Checks data uniqueness and shuffling between GPUs
- **Multi-GPU Support**: Tests with different numbers of GPUs
- **Comprehensive Reporting**: Detailed timing metrics and data analysis
- **Multiple Test Types**: Basic, multi-GPU, performance, and custom tests

## Usage

### Basic Usage

```bash
# Run all tests
python test_data_loading.py

# Run with custom data directory
python test_data_loading.py --data_dir /path/to/your/data

# Run a single test
python test_data_loading.py --single_test --chunk_size 30 --num_batches 50
```

### Test Types

```bash
# Run basic tests (quick, standard, extended)
python test_data_loading.py --test_type basic

# Run multi-GPU tests
python test_data_loading.py --test_type multi_gpu

# Run performance tests
python test_data_loading.py --test_type performance

# Run custom test
python test_data_loading.py --test_type custom --chunk_size 50 --num_batches 100 --world_size 2
```

### Command Line Options

- `--data_dir`: Path to the dataset directory (default: `/home/vignesh/raid/PaperMulti_1_2_Filtered`)
- `--test_type`: Type of tests to run (`basic`, `multi_gpu`, `performance`, `custom`, `all`)
- `--chunk_size`: Action sequence length / chunk size (default: 30)
- `--num_batches`: Number of batches to test (default: 100)
- `--world_size`: Number of GPUs to use (default: all available)
- `--single_test`: Run a single test instead of a test suite

## What It Tests

### Timing Metrics
- **Batch Load Time**: Time to load a batch from the dataset
- **Data Transfer Time**: Time to move data to GPU
- **Total Batch Time**: End-to-end batch processing time
- **Throughput**: Samples processed per second

### Data Verification
- **Batch Uniqueness**: Ensures each batch is unique within and across GPUs
- **Sample Uniqueness**: Verifies individual samples are unique
- **Data Shuffling**: Confirms different GPUs see different data
- **Sequence Analysis**: Analyzes batch ordering and shuffling patterns

### Multi-GPU Analysis
- **Cross-GPU Uniqueness**: Ensures no data overlap between GPUs
- **Shuffling Verification**: Confirms proper data distribution
- **Performance Scaling**: Measures performance across different GPU counts

## Output

The script provides:
1. **Real-time Progress**: Progress bars and timing information during testing
2. **Detailed Analysis**: Comprehensive timing and data verification results
3. **Summary Report**: Test success/failure summary
4. **JSON Results**: Detailed results saved to timestamped JSON file

## Example Output

```
🚀 Running Data Loading Test Suite
============================================================

🧪 Running: Quick Test
  Chunk size: 10
  Number of batches: 20
  World size: 1
----------------------------------------

📈 DATA LOADING PERFORMANCE ANALYSIS
================================================================================

⏱️  TIMING METRICS:
----------------------------------------

Rank 0:
  Average batch load time: 45.23 ± 12.34 ms
  Average data transfer time: 8.76 ± 2.11 ms
  Average total batch time: 54.12 ± 14.22 ms
  Average throughput: 18.47 ± 4.85 samples/sec

🔍 DATA UNIQUENESS VERIFICATION:
----------------------------------------
Rank 0: 20/20 unique batches (100.0%)
Across all ranks: 20/20 unique batches (100.0%)

🔀 DATA SHUFFLING VERIFICATION:
----------------------------------------
Rank 0 vs Rank 1: 0/40 overlapping batches (0.0% overlap)

📊 TEST SUMMARY
================================================================================
✅ Quick Test
✅ Standard Test
❌ Extended Test
----------------------------------------
Total tests: 3
Passed: 2
Failed: 1
Success rate: 66.7%
================================================================================
```

## Requirements

- PyTorch with CUDA support
- WebDataset
- NumPy
- tqdm
- The ACT codebase (ACT.py, data_utils.py, etc.)

## Notes

- The script automatically converts HDF5 data to WebDataset format if needed
- Data conversion is done once and cached for subsequent runs
- The script uses the same data loading configuration as `train_dist.py`
- Results are saved to timestamped JSON files for further analysis
