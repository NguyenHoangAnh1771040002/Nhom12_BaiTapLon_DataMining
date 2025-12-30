"""
run_pipeline.py - Chạy toàn bộ pipeline của dự án Data Mining
Hotel Booking Cancellation Prediction

Usage:
    python scripts/run_pipeline.py --all          # Chạy toàn bộ pipeline
    python scripts/run_pipeline.py --eda          # Chỉ chạy EDA
    python scripts/run_pipeline.py --preprocess   # Chỉ chạy tiền xử lý
    python scripts/run_pipeline.py --mining       # Chỉ chạy data mining
    python scripts/run_pipeline.py --modeling     # Chỉ chạy modeling
    python scripts/run_pipeline.py --timeseries   # Chỉ chạy time series
    python scripts/run_pipeline.py --report       # Chỉ chạy báo cáo
"""

import os
import sys
import argparse
import yaml
import time
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Setup logging
def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    return logger


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except:
        pass
    
    # Set environment variable for other libraries
    os.environ['PYTHONHASHSEED'] = str(seed)


class PipelineRunner:
    """Class to run the complete data mining pipeline."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.seed = config.get('seed', 42)
        self.project_root = PROJECT_ROOT
        
        # Set paths
        self.data_raw = self.project_root / config['paths']['data_raw']
        self.data_processed = self.project_root / config['paths']['data_processed']
        self.output_figures = self.project_root / config['paths']['outputs_figures']
        self.output_tables = self.project_root / config['paths']['outputs_tables']
        self.output_models = self.project_root / config['paths']['outputs_models']
        self.output_reports = self.project_root / config['paths']['outputs_reports']
        
        # Ensure directories exist
        for dir_path in [self.data_processed, self.output_figures, 
                         self.output_tables, self.output_models, self.output_reports]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.results = {
            'eda': {},
            'preprocess': {},
            'mining': {},
            'modeling': {},
            'semi_supervised': {},
            'time_series': {},
            'report': {}
        }
    
    def run_eda(self) -> dict:
        """Phase 2: Exploratory Data Analysis"""
        self.logger.info("="*60)
        self.logger.info("PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # Load raw data
            from src.data.loader import load_raw_data, get_data_info
            
            self.logger.info(f"Loading data from: {self.data_raw}")
            df = load_raw_data(str(self.data_raw))
            
            # Get data info
            info = get_data_info(df)
            self.logger.info(f"Dataset shape: ({info['n_rows']}, {info['n_columns']})")
            self.logger.info(f"Missing values: {sum(info['missing_values'].values())}")
            
            # Calculate target rate
            target_col = self.config.get('target', 'is_canceled')
            cancel_rate = 0
            if target_col in df.columns:
                cancel_rate = df[target_col].mean() * 100
                self.logger.info(f"Cancellation rate: {cancel_rate:.2f}%")
            
            # Basic statistics
            stats = df.describe()
            
            # Save EDA summary
            eda_summary_path = self.output_tables / 'eda_summary.csv'
            stats.to_csv(eda_summary_path)
            self.logger.info(f"EDA summary saved to: {eda_summary_path}")
            
            elapsed = time.time() - start_time
            self.results['eda'] = {
                'status': 'success',
                'shape': (info['n_rows'], info['n_columns']),
                'missing_total': sum(info['missing_values'].values()),
                'cancel_rate': cancel_rate,
                'elapsed_time': elapsed
            }
            
            self.logger.info(f"✅ EDA completed in {elapsed:.2f}s")
            return self.results['eda']
            
        except Exception as e:
            self.logger.error(f"❌ EDA failed: {str(e)}")
            self.results['eda'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def run_preprocessing(self) -> dict:
        """Phase 3: Data Preprocessing & Feature Engineering"""
        self.logger.info("="*60)
        self.logger.info("PHASE 3: PREPROCESSING & FEATURE ENGINEERING")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            from src.data.loader import load_raw_data
            from src.data.cleaner import clean_data, save_artifacts
            from src.features.builder import build_features
            
            # Load raw data
            self.logger.info("Loading raw data...")
            df = load_raw_data(str(self.data_raw))
            self.logger.info(f"Raw data shape: {df.shape}")
            
            # Clean data
            self.logger.info("Cleaning data...")
            df_cleaned, artifacts = clean_data(
                df,
                target_col=self.config.get('target', 'is_canceled'),
                drop_leakage=True,
                handle_missing=True,
                handle_outliers=True
            )
            self.logger.info(f"Cleaned data shape: {df_cleaned.shape}")
            
            # Build features
            self.logger.info("Building features...")
            df_featured = build_features(
                df_cleaned,
                create_time_features=True,
                create_guest_features=True,
                create_booking_features=True
            )
            self.logger.info(f"Featured data shape: {df_featured.shape}")
            
            # Save processed data
            processed_path = self.data_processed / 'hotel_bookings_processed.csv'
            df_featured.to_csv(processed_path, index=False)
            self.logger.info(f"Processed data saved to: {processed_path}")
            
            # Save artifacts
            artifacts_path = self.data_processed / 'preprocessing_artifacts.pkl'
            save_artifacts(artifacts, str(artifacts_path))
            self.logger.info(f"Artifacts saved to: {artifacts_path}")
            
            elapsed = time.time() - start_time
            self.results['preprocess'] = {
                'status': 'success',
                'raw_shape': df.shape,
                'cleaned_shape': df_cleaned.shape,
                'featured_shape': df_featured.shape,
                'elapsed_time': elapsed
            }
            
            self.logger.info(f"✅ Preprocessing completed in {elapsed:.2f}s")
            return self.results['preprocess']
            
        except Exception as e:
            self.logger.error(f"❌ Preprocessing failed: {str(e)}")
            self.results['preprocess'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def run_mining(self) -> dict:
        """Phase 4: Data Mining (Association Rules & Clustering)"""
        self.logger.info("="*60)
        self.logger.info("PHASE 4: DATA MINING (ASSOCIATION & CLUSTERING)")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            from src.data.loader import load_raw_data
            from src.mining.association import mine_association_rules
            from src.mining.clustering import cluster_bookings
            
            # Load processed data
            processed_path = self.data_processed / 'hotel_bookings_processed.csv'
            if processed_path.exists():
                df = pd.read_csv(processed_path)
            else:
                df = load_raw_data(str(self.data_raw))
            
            self.logger.info(f"Data shape for mining: {df.shape}")
            
            # Association Rules
            self.logger.info("Running Association Rule Mining...")
            try:
                rules_df, frequent_itemsets = mine_association_rules(
                    df,
                    target_col='is_canceled',
                    min_support=self.config['association']['min_support'],
                    min_confidence=self.config['association']['min_confidence']
                )
                n_rules = len(rules_df) if rules_df is not None else 0
                self.logger.info(f"Found {n_rules} association rules")
                
                # Save rules
                if rules_df is not None and len(rules_df) > 0:
                    rules_path = self.output_tables / 'association_rules.csv'
                    rules_df.to_csv(rules_path, index=False)
            except Exception as e:
                self.logger.warning(f"Association rules failed: {e}")
                n_rules = 0
            
            # Clustering
            self.logger.info("Running Clustering...")
            try:
                cluster_results = cluster_bookings(
                    df,
                    n_clusters=5,
                    method='kmeans',
                    random_state=self.seed
                )
                n_clusters = cluster_results.get('n_clusters', 0)
                silhouette = cluster_results.get('silhouette_score', 0)
                self.logger.info(f"Clustering: {n_clusters} clusters, Silhouette: {silhouette:.4f}")
            except Exception as e:
                self.logger.warning(f"Clustering failed: {e}")
                n_clusters = 0
                silhouette = 0
            
            elapsed = time.time() - start_time
            self.results['mining'] = {
                'status': 'success',
                'n_association_rules': n_rules,
                'n_clusters': n_clusters,
                'silhouette_score': silhouette,
                'elapsed_time': elapsed
            }
            
            self.logger.info(f"✅ Mining completed in {elapsed:.2f}s")
            return self.results['mining']
            
        except Exception as e:
            self.logger.error(f"❌ Mining failed: {str(e)}")
            self.results['mining'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def run_modeling(self) -> dict:
        """Phase 5: Classification Modeling"""
        self.logger.info("="*60)
        self.logger.info("PHASE 5: CLASSIFICATION MODELING")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            from src.data.loader import load_raw_data
            from src.models.supervised import (
                train_logistic_regression,
                train_decision_tree,
                train_random_forest,
                train_xgboost
            )
            from src.evaluation.metrics import calculate_metrics
            from sklearn.model_selection import train_test_split
            import joblib
            
            # Load processed data
            processed_path = self.data_processed / 'hotel_bookings_processed.csv'
            if processed_path.exists():
                df = pd.read_csv(processed_path)
            else:
                df = load_raw_data(str(self.data_raw))
            
            # Prepare features
            target_col = self.config.get('target', 'is_canceled')
            
            # Drop non-numeric and target columns
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if c != target_col]
            
            X = df[feature_cols].fillna(0)
            y = df[target_col]
            
            self.logger.info(f"Features: {len(feature_cols)}, Samples: {len(y)}")
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config['split']['test_size'],
                random_state=self.seed,
                stratify=y
            )
            
            self.logger.info(f"Train: {len(y_train)}, Test: {len(y_test)}")
            
            # Train models
            models_results = {}
            
            # Logistic Regression
            self.logger.info("Training Logistic Regression...")
            lr_model = train_logistic_regression(
                X_train, y_train,
                **self.config['models']['logistic_regression']
            )
            lr_metrics = calculate_metrics(y_test, lr_model.predict(X_test), 
                                                  lr_model.predict_proba(X_test)[:, 1])
            models_results['Logistic Regression'] = lr_metrics
            self.logger.info(f"  F1: {lr_metrics['f1']:.4f}, ROC-AUC: {lr_metrics['roc_auc']:.4f}")
            
            # Decision Tree
            self.logger.info("Training Decision Tree...")
            dt_model = train_decision_tree(
                X_train, y_train,
                **self.config['models']['decision_tree']
            )
            dt_metrics = calculate_metrics(y_test, dt_model.predict(X_test),
                                                  dt_model.predict_proba(X_test)[:, 1])
            models_results['Decision Tree'] = dt_metrics
            self.logger.info(f"  F1: {dt_metrics['f1']:.4f}, ROC-AUC: {dt_metrics['roc_auc']:.4f}")
            
            # Random Forest
            self.logger.info("Training Random Forest...")
            rf_model = train_random_forest(
                X_train, y_train,
                **self.config['models']['random_forest']
            )
            rf_metrics = calculate_metrics(y_test, rf_model.predict(X_test),
                                                  rf_model.predict_proba(X_test)[:, 1])
            models_results['Random Forest'] = rf_metrics
            self.logger.info(f"  F1: {rf_metrics['f1']:.4f}, ROC-AUC: {rf_metrics['roc_auc']:.4f}")
            
            # XGBoost (optional)
            try:
                self.logger.info("Training XGBoost...")
                xgb_model = train_xgboost(
                    X_train, y_train,
                    **self.config['models']['xgboost']
                )
                xgb_metrics = calculate_metrics(y_test, xgb_model.predict(X_test),
                                                      xgb_model.predict_proba(X_test)[:, 1])
                models_results['XGBoost'] = xgb_metrics
                self.logger.info(f"  F1: {xgb_metrics['f1']:.4f}, ROC-AUC: {xgb_metrics['roc_auc']:.4f}")
            except Exception as e:
                self.logger.warning(f"XGBoost failed: {e}")
            
            # Find best model
            best_model_name = max(models_results, key=lambda x: models_results[x]['f1'])
            best_f1 = models_results[best_model_name]['f1']
            
            self.logger.info(f"Best Model: {best_model_name} (F1: {best_f1:.4f})")
            
            # Save best model
            best_model = rf_model if best_model_name == 'Random Forest' else lr_model
            model_path = self.output_models / 'best_model.pkl'
            joblib.dump(best_model, model_path)
            self.logger.info(f"Best model saved to: {model_path}")
            
            # Save results
            results_df = pd.DataFrame(models_results).T
            results_path = self.output_tables / 'model_comparison.csv'
            results_df.to_csv(results_path)
            
            elapsed = time.time() - start_time
            self.results['modeling'] = {
                'status': 'success',
                'models_results': models_results,
                'best_model': best_model_name,
                'best_f1': best_f1,
                'elapsed_time': elapsed
            }
            
            self.logger.info(f"✅ Modeling completed in {elapsed:.2f}s")
            return self.results['modeling']
            
        except Exception as e:
            self.logger.error(f"❌ Modeling failed: {str(e)}")
            self.results['modeling'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def run_semi_supervised(self) -> dict:
        """Phase 6: Semi-Supervised Learning"""
        self.logger.info("="*60)
        self.logger.info("PHASE 6: SEMI-SUPERVISED LEARNING")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            from src.data.loader import load_raw_data
            from src.models.semi_supervised import (
                create_labeled_unlabeled_split,
                train_self_training,
                train_label_spreading
            )
            from src.evaluation.metrics import calculate_metrics
            from sklearn.model_selection import train_test_split
            
            # Load processed data
            processed_path = self.data_processed / 'hotel_bookings_processed.csv'
            if processed_path.exists():
                df = pd.read_csv(processed_path)
            else:
                df = load_raw_data(str(self.data_raw))
            
            # Prepare features
            target_col = self.config.get('target', 'is_canceled')
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if c != target_col]
            
            X = df[feature_cols].fillna(0)
            y = df[target_col]
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config['split']['test_size'],
                random_state=self.seed,
                stratify=y
            )
            
            semi_results = {}
            labeled_percentages = self.config['semi_supervised']['labeled_percentages']
            
            for pct in labeled_percentages:
                self.logger.info(f"Testing with {pct*100:.0f}% labeled data...")
                
                # Create labeled/unlabeled split
                X_labeled, y_labeled, X_unlabeled = create_labeled_unlabeled_split(
                    X_train, y_train, 
                    labeled_percentage=pct,
                    random_state=self.seed
                )
                
                # Self-training
                try:
                    st_model = train_self_training(
                        X_labeled, y_labeled, X_unlabeled,
                        threshold=self.config['semi_supervised']['self_training']['threshold']
                    )
                    st_pred = st_model.predict(X_test)
                    st_proba = st_model.predict_proba(X_test)[:, 1]
                    st_metrics = calculate_metrics(y_test, st_pred, st_proba)
                    semi_results[f'Self-Training ({pct*100:.0f}%)'] = st_metrics['f1']
                    self.logger.info(f"  Self-Training F1: {st_metrics['f1']:.4f}")
                except Exception as e:
                    self.logger.warning(f"  Self-Training failed: {e}")
                
                # Label Spreading
                try:
                    ls_model = train_label_spreading(X_labeled, y_labeled, X_unlabeled)
                    ls_pred = ls_model.predict(X_test)
                    ls_metrics = calculate_metrics(y_test, ls_pred)
                    semi_results[f'Label Spreading ({pct*100:.0f}%)'] = ls_metrics['f1']
                    self.logger.info(f"  Label Spreading F1: {ls_metrics['f1']:.4f}")
                except Exception as e:
                    self.logger.warning(f"  Label Spreading failed: {e}")
            
            # Save results
            results_df = pd.DataFrame([semi_results])
            results_path = self.output_tables / 'semi_supervised_results.csv'
            results_df.to_csv(results_path, index=False)
            
            elapsed = time.time() - start_time
            self.results['semi_supervised'] = {
                'status': 'success',
                'results': semi_results,
                'elapsed_time': elapsed
            }
            
            self.logger.info(f"✅ Semi-supervised completed in {elapsed:.2f}s")
            return self.results['semi_supervised']
            
        except Exception as e:
            self.logger.error(f"❌ Semi-supervised failed: {str(e)}")
            self.results['semi_supervised'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def run_time_series(self) -> dict:
        """Phase 7: Time Series Forecasting"""
        self.logger.info("="*60)
        self.logger.info("PHASE 7: TIME SERIES FORECASTING")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            from src.data.loader import load_raw_data
            from src.models.forecasting import (
                prepare_time_series,
                naive_forecast,
                moving_average_forecast,
                train_arima,
                train_exponential_smoothing,
                evaluate_forecast
            )
            
            # Load raw data (need date columns)
            df = load_raw_data(str(self.data_raw))
            
            # Prepare time series
            self.logger.info("Preparing time series data...")
            ts_df = prepare_time_series(
                df,
                year_col='arrival_date_year',
                month_col='arrival_date_month',
                day_col='arrival_date_day_of_month',
                target_col='is_canceled',
                period='M'
            )
            
            # Get cancellation rate series
            ts_data = ts_df['cancellation_rate']
            
            self.logger.info(f"Time series length: {len(ts_data)}")
            
            # Train/test split
            train_size = int(len(ts_data) * 0.8)
            train = ts_data[:train_size]
            test = ts_data[train_size:]
            
            self.logger.info(f"Train: {len(train)}, Test: {len(test)}")
            
            ts_results = {}
            n_forecast = len(test)
            
            # Naive forecast
            naive_pred = naive_forecast(train, forecast_periods=n_forecast)
            naive_metrics = evaluate_forecast(test.values, naive_pred)
            ts_results['Naive'] = naive_metrics
            self.logger.info(f"Naive: MAPE={naive_metrics['mape']:.2f}%")
            
            # Moving Average
            for window in [3, 6]:
                ma_pred = moving_average_forecast(train, window=window, forecast_periods=n_forecast)
                ma_metrics = evaluate_forecast(test.values, ma_pred)
                ts_results[f'MA({window})'] = ma_metrics
                self.logger.info(f"MA({window}): MAPE={ma_metrics['mape']:.2f}%")
            
            # ARIMA
            try:
                arima_model, arima_pred = train_arima(train, forecast_periods=n_forecast, order=(1,1,1))
                arima_metrics = evaluate_forecast(test.values, arima_pred)
                ts_results['ARIMA(1,1,1)'] = arima_metrics
                self.logger.info(f"ARIMA(1,1,1): MAPE={arima_metrics['mape']:.2f}%")
            except Exception as e:
                self.logger.warning(f"ARIMA failed: {e}")
            
            # Exponential Smoothing
            try:
                es_model, es_pred = train_exponential_smoothing(train, forecast_periods=n_forecast)
                es_metrics = evaluate_forecast(test.values, es_pred)
                ts_results['Exp. Smoothing'] = es_metrics
                self.logger.info(f"Exp. Smoothing: MAPE={es_metrics['mape']:.2f}%")
            except Exception as e:
                self.logger.warning(f"Exp. Smoothing failed: {e}")
            
            # Find best model
            best_ts_model = min(ts_results, key=lambda x: ts_results[x]['mape'])
            best_mape = ts_results[best_ts_model]['mape']
            
            self.logger.info(f"Best TS Model: {best_ts_model} (MAPE: {best_mape:.2f}%)")
            
            # Save results
            results_df = pd.DataFrame(ts_results).T
            results_path = self.output_tables / 'time_series_results.csv'
            results_df.to_csv(results_path)
            
            elapsed = time.time() - start_time
            self.results['time_series'] = {
                'status': 'success',
                'results': ts_results,
                'best_model': best_ts_model,
                'best_mape': best_mape,
                'elapsed_time': elapsed
            }
            
            self.logger.info(f"✅ Time series completed in {elapsed:.2f}s")
            return self.results['time_series']
            
        except Exception as e:
            self.logger.error(f"❌ Time series failed: {str(e)}")
            self.results['time_series'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def run_report(self) -> dict:
        """Phase 8: Generate Final Report"""
        self.logger.info("="*60)
        self.logger.info("PHASE 8: GENERATE FINAL REPORT")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            from src.evaluation.report import generate_full_report
            
            # Collect all results
            report_data = {
                'project_name': self.config['project']['name'],
                'seed': self.seed,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'phases': self.results
            }
            
            # Generate summary
            summary_lines = [
                "=" * 70,
                "PIPELINE EXECUTION SUMMARY",
                "=" * 70,
                f"Project: {self.config['project']['name']}",
                f"Random Seed: {self.seed}",
                f"Timestamp: {report_data['timestamp']}",
                "",
                "PHASE RESULTS:",
                "-" * 50
            ]
            
            for phase, result in self.results.items():
                status = result.get('status', 'not_run')
                elapsed = result.get('elapsed_time', 0)
                
                if status == 'success':
                    summary_lines.append(f"✅ {phase.upper()}: SUCCESS ({elapsed:.2f}s)")
                    
                    # Add key metrics
                    if phase == 'modeling' and 'best_model' in result:
                        summary_lines.append(f"   Best Model: {result['best_model']} (F1: {result['best_f1']:.4f})")
                    elif phase == 'time_series' and 'best_model' in result:
                        summary_lines.append(f"   Best TS Model: {result['best_model']} (MAPE: {result['best_mape']:.2f}%)")
                elif status == 'failed':
                    summary_lines.append(f"❌ {phase.upper()}: FAILED - {result.get('error', 'Unknown error')}")
                else:
                    summary_lines.append(f"⏭️ {phase.upper()}: SKIPPED")
            
            summary_lines.extend([
                "",
                "=" * 70,
                "PIPELINE COMPLETED",
                "=" * 70
            ])
            
            summary = "\n".join(summary_lines)
            self.logger.info("\n" + summary)
            
            # Save summary
            summary_path = self.output_reports / 'pipeline_summary.txt'
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            self.logger.info(f"Summary saved to: {summary_path}")
            
            elapsed = time.time() - start_time
            self.results['report'] = {
                'status': 'success',
                'summary_path': str(summary_path),
                'elapsed_time': elapsed
            }
            
            self.logger.info(f"✅ Report completed in {elapsed:.2f}s")
            return self.results['report']
            
        except Exception as e:
            self.logger.error(f"❌ Report failed: {str(e)}")
            self.results['report'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def run_all(self):
        """Run complete pipeline."""
        self.logger.info("=" * 70)
        self.logger.info("STARTING COMPLETE PIPELINE")
        self.logger.info(f"Random Seed: {self.seed}")
        self.logger.info("=" * 70)
        
        total_start = time.time()
        
        # Set random seed
        set_random_seed(self.seed)
        
        phases = [
            ('eda', self.run_eda),
            ('preprocess', self.run_preprocessing),
            ('mining', self.run_mining),
            ('modeling', self.run_modeling),
            ('semi_supervised', self.run_semi_supervised),
            ('time_series', self.run_time_series),
            ('report', self.run_report)
        ]
        
        for phase_name, phase_func in phases:
            try:
                phase_func()
            except Exception as e:
                self.logger.error(f"Phase {phase_name} failed, continuing...")
                continue
        
        total_elapsed = time.time() - total_start
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"COMPLETE PIPELINE FINISHED in {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
        self.logger.info(f"{'='*70}")
        
        return self.results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run Hotel Booking Cancellation Prediction Pipeline'
    )
    
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--eda', action='store_true', help='Run EDA only')
    parser.add_argument('--preprocess', action='store_true', help='Run preprocessing only')
    parser.add_argument('--mining', action='store_true', help='Run data mining only')
    parser.add_argument('--modeling', action='store_true', help='Run modeling only')
    parser.add_argument('--semi', action='store_true', help='Run semi-supervised only')
    parser.add_argument('--timeseries', action='store_true', help='Run time series only')
    parser.add_argument('--report', action='store_true', help='Generate report only')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    parser.add_argument('--config', type=str, default='configs/params.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Setup
    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)
    
    if args.seed is not None:
        config['seed'] = args.seed
    
    # Setup logging
    log_dir = PROJECT_ROOT / 'outputs' / 'logs'
    logger = setup_logging(log_dir)
    
    logger.info(f"Config loaded from: {config_path}")
    logger.info(f"Random seed: {config['seed']}")
    
    # Create pipeline runner
    runner = PipelineRunner(config, logger)
    
    # Run selected phases
    if args.all or not any([args.eda, args.preprocess, args.mining, 
                           args.modeling, args.semi, args.timeseries, args.report]):
        runner.run_all()
    else:
        set_random_seed(config['seed'])
        
        if args.eda:
            runner.run_eda()
        if args.preprocess:
            runner.run_preprocessing()
        if args.mining:
            runner.run_mining()
        if args.modeling:
            runner.run_modeling()
        if args.semi:
            runner.run_semi_supervised()
        if args.timeseries:
            runner.run_time_series()
        if args.report:
            runner.run_report()
    
    logger.info("Pipeline execution completed!")


if __name__ == '__main__':
    main()
