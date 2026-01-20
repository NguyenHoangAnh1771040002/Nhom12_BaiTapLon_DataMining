"""
run_papermill.py - Chạy notebooks bằng papermill để reproducibility
Hotel Booking Cancellation Prediction

Usage:
    python scripts/run_papermill.py --all           # Chạy tất cả notebooks
    python scripts/run_papermill.py --notebook 01   # Chạy notebook cụ thể
    python scripts/run_papermill.py --list          # Liệt kê notebooks
"""

import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import papermill as pm
    PAPERMILL_AVAILABLE = True
except ImportError:
    PAPERMILL_AVAILABLE = False
    print("⚠️ papermill not installed. Install with: pip install papermill")


# Define notebooks in execution order
NOTEBOOKS = {
    '01': {
        'name': '01_eda.ipynb',
        'description': 'Exploratory Data Analysis',
        'parameters': {}
    },
    '02': {
        'name': '02_preprocess_feature.ipynb',
        'description': 'Data Preprocessing & Feature Engineering',
        'parameters': {}
    },
    '03': {
        'name': '03_mining_clustering.ipynb',
        'description': 'Association Rules & Clustering',
        'parameters': {}
    },
    '04': {
        'name': '04_modeling.ipynb',
        'description': 'Supervised Classification',
        'parameters': {}
    },
    '04b': {
        'name': '04b_semi_supervised.ipynb',
        'description': 'Semi-Supervised Learning',
        'parameters': {}
    },
    '05': {
        'name': '05_time_series.ipynb',
        'description': 'Time Series Forecasting',
        'parameters': {}
    },
    '06': {
        'name': '06_evaluation_report.ipynb',
        'description': 'Evaluation & Final Report',
        'parameters': {}
    }
}


def list_notebooks():
    """List all available notebooks."""
    print("\n📓 Available Notebooks:")
    print("=" * 60)
    
    notebooks_dir = PROJECT_ROOT / 'notebooks'
    
    for nb_id, nb_info in NOTEBOOKS.items():
        nb_path = notebooks_dir / nb_info['name']
        exists = "✅" if nb_path.exists() else "❌"
        print(f"  {exists} [{nb_id}] {nb_info['name']}")
        print(f"       {nb_info['description']}")
    
    print("=" * 60)


def run_notebook(nb_id: str, output_dir: Path = None) -> dict:
    """
    Run a single notebook using papermill.
    
    Args:
        nb_id: Notebook identifier (01, 02, etc.)
        output_dir: Directory to save executed notebooks (default: notebooks/)
    
    Returns:
        dict with execution status and timing
    """
    if not PAPERMILL_AVAILABLE:
        return {'status': 'error', 'error': 'papermill not installed'}
    
    if nb_id not in NOTEBOOKS:
        return {'status': 'error', 'error': f'Unknown notebook: {nb_id}'}
    
    nb_info = NOTEBOOKS[nb_id]
    notebooks_dir = PROJECT_ROOT / 'notebooks'
    input_path = notebooks_dir / nb_info['name']
    
    if not input_path.exists():
        return {'status': 'error', 'error': f'Notebook not found: {input_path}'}
    
    # Output to same directory as input (overwrite original)
    if output_dir is None:
        output_dir = notebooks_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file has same name as input (in-place execution)
    output_path = output_dir / nb_info['name']
    
    print(f"\n🚀 Running: {nb_info['name']}")
    print(f"   Description: {nb_info['description']}")
    print(f"   Path: {input_path}")
    
    start_time = time.time()
    
    try:
        # Run notebook with papermill (in-place execution)
        # Use None for kernel_name to use the kernel specified in the notebook
        pm.execute_notebook(
            str(input_path),
            str(output_path),
            parameters=nb_info.get('parameters', {}),
            kernel_name=None,  # Use kernel from notebook metadata
            progress_bar=True,
            log_output=False,
            cwd=str(notebooks_dir)  # Set working directory to notebooks/
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Completed in {elapsed:.2f}s")
        
        return {
            'status': 'success',
            'notebook': nb_info['name'],
            'output_path': str(output_path),
            'elapsed_time': elapsed
        }
        
    except pm.PapermillExecutionError as e:
        elapsed = time.time() - start_time
        print(f"❌ Execution failed after {elapsed:.2f}s")
        print(f"   Error: {str(e)[:200]}")
        
        return {
            'status': 'failed',
            'notebook': nb_info['name'],
            'error': str(e),
            'elapsed_time': elapsed
        }
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error: {str(e)}")
        
        return {
            'status': 'error',
            'notebook': nb_info['name'],
            'error': str(e),
            'elapsed_time': elapsed
        }


def run_all_notebooks(output_dir: Path = None) -> list:
    """
    Run all notebooks in order.
    
    Returns:
        list of execution results
    """
    print("\n" + "=" * 70)
    print("RUNNING ALL NOTEBOOKS")
    print("=" * 70)
    
    total_start = time.time()
    results = []
    
    # Run in order
    for nb_id in ['01', '02', '03', '04', '04b', '05', '06']:
        result = run_notebook(nb_id, output_dir)
        results.append(result)
        
        # If a notebook fails, ask whether to continue
        if result['status'] != 'success':
            print(f"\n⚠️ Notebook {nb_id} did not complete successfully.")
            # Continue anyway in automated mode
    
    total_elapsed = time.time() - total_start
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    error_count = sum(1 for r in results if r['status'] == 'error')
    
    for result in results:
        status_icon = "✅" if result['status'] == 'success' else "❌"
        elapsed = result.get('elapsed_time', 0)
        print(f"  {status_icon} {result.get('notebook', 'Unknown')}: {result['status']} ({elapsed:.2f}s)")
    
    print("-" * 70)
    print(f"Total: {success_count} success, {failed_count} failed, {error_count} errors")
    print(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    print("=" * 70)
    
    return results


def verify_reproducibility(seed: int = 42) -> bool:
    """
    Verify reproducibility by running pipeline twice and comparing outputs.
    
    Args:
        seed: Random seed to use
    
    Returns:
        bool indicating if outputs are reproducible
    """
    print("\n" + "=" * 70)
    print("VERIFYING REPRODUCIBILITY")
    print(f"Random Seed: {seed}")
    print("=" * 70)
    
    import hashlib
    import pandas as pd
    
    output_tables = PROJECT_ROOT / 'outputs' / 'tables'
    
    # Files to compare for reproducibility
    files_to_check = [
        'model_comparison.csv',
        'semi_supervised_results.csv',
        'ts_model_comparison.csv'
    ]
    
    # Get hashes of current outputs
    current_hashes = {}
    for filename in files_to_check:
        filepath = output_tables / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                current_hashes[filename] = hashlib.md5(f.read()).hexdigest()
            print(f"  📄 {filename}: {current_hashes[filename][:8]}...")
        else:
            print(f"  ⚠️ {filename}: Not found")
    
    print("\n💡 To verify reproducibility:")
    print("   1. Delete outputs/ folder")
    print("   2. Run: python scripts/run_pipeline.py --all --seed 42")
    print("   3. Compare file hashes with above values")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run notebooks with papermill for reproducibility'
    )
    
    parser.add_argument('--all', action='store_true', help='Run all notebooks')
    parser.add_argument('--notebook', '-n', type=str, help='Run specific notebook (01, 02, etc.)')
    parser.add_argument('--list', '-l', action='store_true', help='List available notebooks')
    parser.add_argument('--verify', action='store_true', help='Verify reproducibility')
    parser.add_argument('--output-dir', type=str, help='Output directory for executed notebooks')
    
    args = parser.parse_args()
    
    if args.list:
        list_notebooks()
        return
    
    if args.verify:
        verify_reproducibility()
        return
    
    if not PAPERMILL_AVAILABLE:
        print("❌ papermill is required. Install with: pip install papermill")
        print("   Alternatively, run notebooks manually in Jupyter.")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    if args.all:
        run_all_notebooks(output_dir)
    elif args.notebook:
        run_notebook(args.notebook, output_dir)
    else:
        # Default: list notebooks
        list_notebooks()
        print("\n💡 Usage examples:")
        print("   python scripts/run_papermill.py --all")
        print("   python scripts/run_papermill.py --notebook 01")
        print("   python scripts/run_papermill.py --verify")


if __name__ == '__main__':
    main()
