"""
verify_reproducibility.py - Kiểm tra tính reproducibility của pipeline
Hotel Booking Cancellation Prediction

Usage:
    python scripts/verify_reproducibility.py              # Kiểm tra với seed mặc định
    python scripts/verify_reproducibility.py --seed 42    # Kiểm tra với seed cụ thể
    python scripts/verify_reproducibility.py --full       # Chạy full verification
"""

import os
import sys
import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Try to set other library seeds
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    print(f"✅ All random seeds set to: {seed}")


def compute_file_hash(filepath: Path) -> str:
    """Compute MD5 hash of a file."""
    if not filepath.exists():
        return None
    
    hash_md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """Compute hash of DataFrame contents (ignoring order)."""
    # Sort by all columns for consistency
    df_sorted = df.sort_values(by=df.columns.tolist()).reset_index(drop=True)
    
    # Convert to string and hash
    df_str = df_sorted.to_csv(index=False)
    return hashlib.md5(df_str.encode()).hexdigest()


def verify_random_operations(seed: int, n_trials: int = 3) -> dict:
    """
    Verify that random operations produce same results.
    
    Args:
        seed: Random seed to test
        n_trials: Number of trials to run
    
    Returns:
        dict with verification results
    """
    print("\n📊 Verifying Random Operations...")
    print("-" * 50)
    
    results = {
        'numpy_random': [],
        'sklearn_split': [],
        'pandas_sample': []
    }
    
    for trial in range(n_trials):
        # Reset seed
        set_all_seeds(seed)
        
        # NumPy random
        np_result = np.random.rand(10).sum()
        results['numpy_random'].append(np_result)
        
        # Sklearn train_test_split
        from sklearn.model_selection import train_test_split
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        results['sklearn_split'].append(y_test.sum())
        
        # Pandas sample
        df = pd.DataFrame({'a': range(100)})
        sample = df.sample(n=10, random_state=seed)
        results['pandas_sample'].append(sample['a'].sum())
    
    # Check consistency
    all_consistent = True
    for op_name, values in results.items():
        is_consistent = len(set(values)) == 1
        status = "✅" if is_consistent else "❌"
        print(f"  {status} {op_name}: {values[0]:.6f} (consistent: {is_consistent})")
        if not is_consistent:
            all_consistent = False
    
    return {
        'consistent': all_consistent,
        'results': results
    }


def verify_model_training(seed: int) -> dict:
    """
    Verify that model training is reproducible.
    
    Args:
        seed: Random seed to test
    
    Returns:
        dict with verification results
    """
    print("\n🤖 Verifying Model Training...")
    print("-" * 50)
    
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    results = {'scores': [], 'feature_importances': []}
    
    for trial in range(3):
        # Reset seed
        set_all_seeds(seed)
        
        # Generate data
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=10,
            random_state=seed
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=1)
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        importance = model.feature_importances_[:5].sum()
        
        results['scores'].append(score)
        results['feature_importances'].append(importance)
    
    # Check consistency
    scores_consistent = len(set([f"{s:.10f}" for s in results['scores']])) == 1
    importance_consistent = len(set([f"{i:.10f}" for i in results['feature_importances']])) == 1
    
    print(f"  {'✅' if scores_consistent else '❌'} Accuracy: {results['scores'][0]:.6f} (consistent: {scores_consistent})")
    print(f"  {'✅' if importance_consistent else '❌'} Feature Importance: {results['feature_importances'][0]:.6f} (consistent: {importance_consistent})")
    
    return {
        'consistent': scores_consistent and importance_consistent,
        'results': results
    }


def verify_output_files(output_dir: Path) -> dict:
    """
    Verify output files and compute their hashes.
    
    Args:
        output_dir: Path to outputs directory
    
    Returns:
        dict with file hashes
    """
    print("\n📁 Verifying Output Files...")
    print("-" * 50)
    
    files_to_check = {
        'tables': [
            'model_comparison.csv',
            'semi_supervised_results.csv', 
            'ts_model_comparison.csv',
            'project_summary.csv',
            'association_rules.csv',
            'cluster_profiles.csv'
        ],
        'models': [
            'best_model.pkl',
            'random_forest_model.pkl',
            'random_forest_tuned_model.pkl'
        ],
        'reports': [
            'full_report.md',
            'summary_report.md',
            'business_insights.json'
        ]
    }
    
    hashes = {}
    
    for subdir, files in files_to_check.items():
        subdir_path = output_dir / subdir
        for filename in files:
            filepath = subdir_path / filename
            file_hash = compute_file_hash(filepath)
            
            if file_hash:
                hashes[f"{subdir}/{filename}"] = file_hash
                print(f"  ✅ {subdir}/{filename}: {file_hash[:16]}...")
            else:
                print(f"  ⚠️ {subdir}/{filename}: Not found")
    
    return hashes


def run_mini_pipeline(seed: int) -> dict:
    """
    Run a mini version of the pipeline to verify reproducibility.
    
    Args:
        seed: Random seed
    
    Returns:
        dict with results
    """
    print("\n🔄 Running Mini Pipeline...")
    print("-" * 50)
    
    set_all_seeds(seed)
    
    results = {}
    
    try:
        # Load data
        from src.data.loader import load_raw_data
        
        data_path = PROJECT_ROOT / 'data' / 'raw' / 'hotel_bookings.csv'
        df = load_raw_data(str(data_path))
        
        # Basic preprocessing
        target_col = 'is_canceled'
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c != target_col]
        
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        # Train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        # Train model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=1)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        from sklearn.metrics import f1_score, accuracy_score
        
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        
        results['f1_score'] = f1
        results['accuracy'] = acc
        results['feature_importance_sum'] = model.feature_importances_[:5].sum()
        
        print(f"  ✅ F1-Score: {f1:.6f}")
        print(f"  ✅ Accuracy: {acc:.6f}")
        print(f"  ✅ Top-5 Feature Importance Sum: {results['feature_importance_sum']:.6f}")
        
        results['status'] = 'success'
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        results['status'] = 'failed'
        results['error'] = str(e)
    
    return results


def compare_runs(results1: dict, results2: dict) -> bool:
    """
    Compare results from two pipeline runs.
    
    Args:
        results1: Results from first run
        results2: Results from second run
    
    Returns:
        bool indicating if results match
    """
    print("\n🔍 Comparing Results...")
    print("-" * 50)
    
    all_match = True
    
    for key in results1:
        if key in ['status', 'error']:
            continue
        
        val1 = results1.get(key)
        val2 = results2.get(key)
        
        if isinstance(val1, float):
            match = abs(val1 - val2) < 1e-10
        else:
            match = val1 == val2
        
        status = "✅" if match else "❌"
        print(f"  {status} {key}: {val1} vs {val2}")
        
        if not match:
            all_match = False
    
    return all_match


def full_verification(seed: int) -> dict:
    """
    Run full reproducibility verification.
    
    Args:
        seed: Random seed to use
    
    Returns:
        dict with all verification results
    """
    print("=" * 70)
    print("FULL REPRODUCIBILITY VERIFICATION")
    print(f"Seed: {seed}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    all_results = {}
    
    # 1. Verify random operations
    random_results = verify_random_operations(seed)
    all_results['random_operations'] = random_results
    
    # 2. Verify model training
    model_results = verify_model_training(seed)
    all_results['model_training'] = model_results
    
    # 3. Run mini pipeline twice
    print("\n🔄 Running Pipeline - Trial 1...")
    run1 = run_mini_pipeline(seed)
    
    print("\n🔄 Running Pipeline - Trial 2...")
    run2 = run_mini_pipeline(seed)
    
    # 4. Compare runs
    runs_match = compare_runs(run1, run2)
    all_results['pipeline_runs'] = {
        'run1': run1,
        'run2': run2,
        'match': runs_match
    }
    
    # 5. Check output files (if they exist)
    output_dir = PROJECT_ROOT / 'outputs'
    if output_dir.exists():
        file_hashes = verify_output_files(output_dir)
        all_results['file_hashes'] = file_hashes
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    checks = [
        ('Random Operations', random_results['consistent']),
        ('Model Training', model_results['consistent']),
        ('Pipeline Runs Match', runs_match)
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {check_name}")
        if not passed:
            all_passed = False
    
    print("-" * 70)
    if all_passed:
        print("🎉 ALL CHECKS PASSED - Pipeline is REPRODUCIBLE!")
    else:
        print("⚠️ SOME CHECKS FAILED - Please review the issues above")
    print("=" * 70)
    
    all_results['all_passed'] = all_passed
    
    # Save verification report
    report_path = PROJECT_ROOT / 'outputs' / 'reports' / 'reproducibility_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert non-serializable types
    def make_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(make_serializable(all_results), f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Report saved to: {report_path}")
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Verify reproducibility of the data mining pipeline'
    )
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed to test')
    parser.add_argument('--full', action='store_true', help='Run full verification')
    parser.add_argument('--quick', action='store_true', help='Run quick check only')
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick check
        print("🔍 Quick Reproducibility Check")
        print("-" * 40)
        verify_random_operations(args.seed)
    else:
        # Full verification
        full_verification(args.seed)


if __name__ == '__main__':
    main()
