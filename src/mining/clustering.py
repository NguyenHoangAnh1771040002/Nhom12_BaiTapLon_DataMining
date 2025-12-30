"""
Clustering Module
=================

Clustering algorithms for hotel booking customer segmentation.

Functions:
----------
- prepare_clustering_data: Prepare and scale features for clustering
- find_optimal_k: Find optimal number of clusters using Elbow and Silhouette
- run_kmeans: Run KMeans clustering
- run_dbscan: Run DBSCAN clustering
- run_hierarchical: Run Hierarchical/Agglomerative clustering
- evaluate_clustering: Evaluate clustering results
- profile_clusters: Profile and interpret clusters
- visualize_clusters: Visualize clusters using PCA/t-SNE

Requirements:
-------------
- scikit-learn: pip install scikit-learn
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import warnings

# Sklearn imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')


# ====================
# DEFAULT FEATURES FOR CLUSTERING
# ====================

DEFAULT_CLUSTERING_FEATURES = [
    'lead_time',
    'total_nights',
    'total_guests',
    'adr',
    'total_of_special_requests',
    'booking_changes',
    'previous_cancellations',
    'previous_bookings_not_canceled',
    'days_in_waiting_list',
    'is_repeated_guest'
]


# ====================
# DATA PREPARATION
# ====================

def prepare_clustering_data(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    scaling_method: str = 'standard',
    handle_missing: str = 'median',
    verbose: bool = True
) -> Tuple[pd.DataFrame, object, List[str]]:
    """
    Prepare and scale data for clustering.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    features : list, optional
        Features to use for clustering. If None, uses defaults.
    scaling_method : str
        Scaling method: 'standard', 'minmax', 'robust'
    handle_missing : str
        How to handle missing values: 'median', 'mean', 'drop'
    verbose : bool
        Whether to print information
        
    Returns
    -------
    tuple
        (scaled_data, scaler, feature_names)
        
    Examples
    --------
    >>> X_scaled, scaler, features = prepare_clustering_data(df)
    """
    if features is None:
        features = [f for f in DEFAULT_CLUSTERING_FEATURES if f in df.columns]
    else:
        features = [f for f in features if f in df.columns]
    
    if verbose:
        print("=" * 60)
        print("PREPARING DATA FOR CLUSTERING")
        print("=" * 60)
        print(f"Selected features: {features}")
    
    # Extract features
    X = df[features].copy()
    
    # Handle missing values
    if handle_missing == 'median':
        X = X.fillna(X.median())
    elif handle_missing == 'mean':
        X = X.fillna(X.mean())
    elif handle_missing == 'drop':
        X = X.dropna()
    
    if verbose:
        print(f"Data shape: {X.shape}")
        print(f"Missing values after handling: {X.isnull().sum().sum()}")
    
    # Scale data
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {scaling_method}")
    
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=X.index)
    
    if verbose:
        print(f"Scaling method: {scaling_method}")
        print("=" * 60)
    
    return X_scaled_df, scaler, features


# ====================
# OPTIMAL K SELECTION
# ====================

def find_optimal_k(
    X: pd.DataFrame,
    k_range: Tuple[int, int] = (2, 11),
    method: str = 'both',
    random_state: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Find optimal number of clusters using Elbow and Silhouette methods.
    
    Parameters
    ----------
    X : pd.DataFrame
        Scaled feature data
    k_range : tuple
        Range of k values to try (min, max)
    method : str
        Method to use: 'elbow', 'silhouette', or 'both'
    random_state : int
        Random seed
    verbose : bool
        Whether to print information
        
    Returns
    -------
    dict
        Results containing inertias, silhouette scores, and optimal k
        
    Examples
    --------
    >>> results = find_optimal_k(X_scaled, k_range=(2, 10))
    >>> optimal_k = results['optimal_k_silhouette']
    """
    k_values = list(range(k_range[0], k_range[1]))
    inertias = []
    silhouette_scores = []
    
    if verbose:
        print("=" * 60)
        print("FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("=" * 60)
        print(f"Testing k from {k_range[0]} to {k_range[1]-1}")
    
    for k in k_values:
        # Fit KMeans
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Store inertia (for elbow method)
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette score
        sil_score = silhouette_score(X, labels)
        silhouette_scores.append(sil_score)
        
        if verbose:
            print(f"  k={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={sil_score:.4f}")
    
    # Find optimal k based on silhouette
    optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
    
    # Find elbow point using rate of change
    inertia_changes = np.diff(inertias)
    inertia_changes_2nd = np.diff(inertia_changes)
    optimal_k_elbow = k_values[np.argmax(inertia_changes_2nd) + 1] if len(inertia_changes_2nd) > 0 else k_values[0]
    
    results = {
        'k_values': k_values,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'optimal_k_silhouette': optimal_k_silhouette,
        'optimal_k_elbow': optimal_k_elbow,
        'best_silhouette_score': max(silhouette_scores)
    }
    
    if verbose:
        print("-" * 60)
        print(f"Optimal k (Silhouette): {optimal_k_silhouette} (score: {max(silhouette_scores):.4f})")
        print(f"Optimal k (Elbow): {optimal_k_elbow}")
        print("=" * 60)
    
    return results


def plot_elbow_silhouette(
    results: Dict,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
):
    """
    Plot Elbow and Silhouette curves.
    
    Parameters
    ----------
    results : dict
        Output from find_optimal_k
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Elbow plot
    axes[0].plot(results['k_values'], results['inertias'], 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(x=results['optimal_k_elbow'], color='r', linestyle='--', label=f"Elbow k={results['optimal_k_elbow']}")
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia (Within-cluster Sum of Squares)')
    axes[0].set_title('Elbow Method')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Silhouette plot
    axes[1].plot(results['k_values'], results['silhouette_scores'], 'go-', linewidth=2, markersize=8)
    axes[1].axvline(x=results['optimal_k_silhouette'], color='r', linestyle='--', label=f"Best k={results['optimal_k_silhouette']}")
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Method')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()
    return fig


# ====================
# CLUSTERING ALGORITHMS
# ====================

def run_kmeans(
    X: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 10,
    verbose: bool = True
) -> Tuple[np.ndarray, KMeans]:
    """
    Run KMeans clustering.
    
    Parameters
    ----------
    X : pd.DataFrame
        Scaled feature data
    n_clusters : int
        Number of clusters
    random_state : int
        Random seed
    n_init : int
        Number of initializations
    verbose : bool
        Whether to print information
        
    Returns
    -------
    tuple
        (cluster_labels, kmeans_model)
        
    Examples
    --------
    >>> labels, model = run_kmeans(X_scaled, n_clusters=5)
    """
    if verbose:
        print("=" * 60)
        print(f"RUNNING KMEANS (k={n_clusters})")
        print("=" * 60)
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init
    )
    
    labels = kmeans.fit_predict(X)
    
    if verbose:
        print(f"Inertia: {kmeans.inertia_:.2f}")
        print(f"Iterations: {kmeans.n_iter_}")
        print("\nCluster sizes:")
        unique, counts = np.unique(labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            pct = count / len(labels) * 100
            print(f"  Cluster {cluster}: {count:,} ({pct:.1f}%)")
        print("=" * 60)
    
    return labels, kmeans


def run_dbscan(
    X: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 5,
    verbose: bool = True
) -> Tuple[np.ndarray, DBSCAN]:
    """
    Run DBSCAN clustering.
    
    Parameters
    ----------
    X : pd.DataFrame
        Scaled feature data
    eps : float
        Maximum distance between samples
    min_samples : int
        Minimum samples in a neighborhood
    verbose : bool
        Whether to print information
        
    Returns
    -------
    tuple
        (cluster_labels, dbscan_model)
        
    Examples
    --------
    >>> labels, model = run_dbscan(X_scaled, eps=0.5, min_samples=5)
    """
    if verbose:
        print("=" * 60)
        print(f"RUNNING DBSCAN (eps={eps}, min_samples={min_samples})")
        print("=" * 60)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    
    if verbose:
        print(f"Number of clusters: {n_clusters}")
        print(f"Noise points: {n_noise:,} ({n_noise/len(labels)*100:.1f}%)")
        print("\nCluster sizes:")
        unique, counts = np.unique(labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            pct = count / len(labels) * 100
            label_name = "Noise" if cluster == -1 else f"Cluster {cluster}"
            print(f"  {label_name}: {count:,} ({pct:.1f}%)")
        print("=" * 60)
    
    return labels, dbscan


def run_hierarchical(
    X: pd.DataFrame,
    n_clusters: int,
    linkage: str = 'ward',
    verbose: bool = True
) -> Tuple[np.ndarray, AgglomerativeClustering]:
    """
    Run Hierarchical/Agglomerative clustering.
    
    Parameters
    ----------
    X : pd.DataFrame
        Scaled feature data
    n_clusters : int
        Number of clusters
    linkage : str
        Linkage criterion: 'ward', 'complete', 'average', 'single'
    verbose : bool
        Whether to print information
        
    Returns
    -------
    tuple
        (cluster_labels, model)
        
    Examples
    --------
    >>> labels, model = run_hierarchical(X_scaled, n_clusters=5)
    """
    if verbose:
        print("=" * 60)
        print(f"RUNNING HIERARCHICAL CLUSTERING (k={n_clusters}, linkage={linkage})")
        print("=" * 60)
    
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = hc.fit_predict(X)
    
    if verbose:
        print("\nCluster sizes:")
        unique, counts = np.unique(labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            pct = count / len(labels) * 100
            print(f"  Cluster {cluster}: {count:,} ({pct:.1f}%)")
        print("=" * 60)
    
    return labels, hc


# ====================
# EVALUATION
# ====================

def evaluate_clustering(
    X: pd.DataFrame,
    labels: np.ndarray,
    verbose: bool = True
) -> Dict:
    """
    Evaluate clustering results using multiple metrics.
    
    Parameters
    ----------
    X : pd.DataFrame
        Scaled feature data
    labels : np.ndarray
        Cluster labels
    verbose : bool
        Whether to print information
        
    Returns
    -------
    dict
        Evaluation metrics
        
    Examples
    --------
    >>> metrics = evaluate_clustering(X_scaled, labels)
    """
    # Remove noise points for evaluation (if DBSCAN)
    mask = labels != -1
    X_eval = X[mask] if isinstance(X, pd.DataFrame) else X[mask]
    labels_eval = labels[mask]
    
    n_clusters = len(set(labels_eval))
    
    if n_clusters < 2:
        if verbose:
            print("⚠️ Less than 2 clusters, cannot compute metrics")
        return {'n_clusters': n_clusters}
    
    metrics = {
        'n_clusters': n_clusters,
        'silhouette_score': silhouette_score(X_eval, labels_eval),
        'davies_bouldin_score': davies_bouldin_score(X_eval, labels_eval),
        'calinski_harabasz_score': calinski_harabasz_score(X_eval, labels_eval)
    }
    
    if verbose:
        print("=" * 60)
        print("CLUSTERING EVALUATION METRICS")
        print("=" * 60)
        print(f"Number of clusters: {metrics['n_clusters']}")
        print(f"Silhouette Score: {metrics['silhouette_score']:.4f} (higher is better, -1 to 1)")
        print(f"Davies-Bouldin Index: {metrics['davies_bouldin_score']:.4f} (lower is better)")
        print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.2f} (higher is better)")
        print("=" * 60)
    
    return metrics


# ====================
# CLUSTER PROFILING
# ====================

def profile_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    features: Optional[List[str]] = None,
    target_col: str = 'is_canceled',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Profile clusters by analyzing feature distributions and cancellation rates.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe (before scaling)
    labels : np.ndarray
        Cluster labels
    features : list, optional
        Features to include in profiling
    target_col : str
        Target column for cancellation analysis
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        Cluster profiles
        
    Examples
    --------
    >>> profiles = profile_clusters(df, labels)
    """
    if features is None:
        features = [f for f in DEFAULT_CLUSTERING_FEATURES if f in df.columns]
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = labels
    
    # Remove noise points if present
    df_clustered = df_clustered[df_clustered['cluster'] != -1]
    
    if verbose:
        print("=" * 60)
        print("CLUSTER PROFILING")
        print("=" * 60)
    
    profiles = []
    
    for cluster in sorted(df_clustered['cluster'].unique()):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster]
        
        profile = {'cluster': cluster}
        profile['size'] = len(cluster_data)
        profile['size_pct'] = len(cluster_data) / len(df_clustered) * 100
        
        # Cancellation rate
        if target_col in cluster_data.columns:
            profile['cancel_rate'] = cluster_data[target_col].mean() * 100
        
        # Feature statistics
        for feat in features:
            if feat in cluster_data.columns:
                profile[f'{feat}_mean'] = cluster_data[feat].mean()
                profile[f'{feat}_median'] = cluster_data[feat].median()
        
        profiles.append(profile)
    
    profiles_df = pd.DataFrame(profiles)
    
    if verbose:
        print("\nCluster Summary:")
        for _, row in profiles_df.iterrows():
            print(f"\n📊 Cluster {int(row['cluster'])}:")
            print(f"   Size: {int(row['size']):,} ({row['size_pct']:.1f}%)")
            if 'cancel_rate' in row:
                print(f"   Cancellation Rate: {row['cancel_rate']:.1f}%")
        print("=" * 60)
    
    return profiles_df


def identify_high_risk_clusters(
    profiles: pd.DataFrame,
    threshold_pct: float = 50.0,
    verbose: bool = True
) -> List[int]:
    """
    Identify clusters with high cancellation risk.
    
    Parameters
    ----------
    profiles : pd.DataFrame
        Cluster profiles from profile_clusters
    threshold_pct : float
        Cancellation rate threshold to consider high risk
    verbose : bool
        Whether to print information
        
    Returns
    -------
    list
        List of high-risk cluster IDs
    """
    if 'cancel_rate' not in profiles.columns:
        if verbose:
            print("⚠️ No cancellation rate in profiles")
        return []
    
    high_risk = profiles[profiles['cancel_rate'] >= threshold_pct]['cluster'].tolist()
    
    if verbose:
        print(f"\n🚨 High-risk clusters (cancel rate >= {threshold_pct}%):")
        for cluster in high_risk:
            rate = profiles[profiles['cluster'] == cluster]['cancel_rate'].values[0]
            print(f"   Cluster {int(cluster)}: {rate:.1f}% cancellation rate")
    
    return [int(c) for c in high_risk]


# ====================
# VISUALIZATION
# ====================

def reduce_dimensions(
    X: pd.DataFrame,
    method: str = 'pca',
    n_components: int = 2,
    random_state: int = 42,
    perplexity: int = 30
) -> np.ndarray:
    """
    Reduce dimensions for visualization.
    
    Parameters
    ----------
    X : pd.DataFrame
        Scaled feature data
    method : str
        Reduction method: 'pca' or 'tsne'
    n_components : int
        Number of components
    random_state : int
        Random seed
    perplexity : int
        Perplexity for t-SNE
        
    Returns
    -------
    np.ndarray
        Reduced data
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return reducer.fit_transform(X)


def plot_clusters_2d(
    X: pd.DataFrame,
    labels: np.ndarray,
    method: str = 'pca',
    title: str = 'Cluster Visualization',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    random_state: int = 42
):
    """
    Plot clusters in 2D using PCA or t-SNE.
    
    Parameters
    ----------
    X : pd.DataFrame
        Scaled feature data
    labels : np.ndarray
        Cluster labels
    method : str
        Reduction method: 'pca' or 'tsne'
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    random_state : int
        Random seed
    """
    import matplotlib.pyplot as plt
    
    # Reduce dimensions
    X_reduced = reduce_dimensions(X, method=method, random_state=random_state)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique labels
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        label_name = "Noise" if label == -1 else f"Cluster {label}"
        ax.scatter(
            X_reduced[mask, 0],
            X_reduced[mask, 1],
            c=[color],
            label=label_name,
            alpha=0.6,
            s=30
        )
    
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()
    return fig


def plot_cluster_profiles(
    profiles: pd.DataFrame,
    features: List[str],
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
):
    """
    Plot cluster profiles as a heatmap.
    
    Parameters
    ----------
    profiles : pd.DataFrame
        Cluster profiles
    features : list
        Features to include
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Extract mean columns
    mean_cols = [f'{f}_mean' for f in features if f'{f}_mean' in profiles.columns]
    
    if not mean_cols:
        print("⚠️ No feature means found in profiles")
        return
    
    # Create heatmap data
    heatmap_data = profiles[['cluster'] + mean_cols].set_index('cluster')
    heatmap_data.columns = [c.replace('_mean', '') for c in heatmap_data.columns]
    
    # Normalize for visualization
    heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        heatmap_normalized.T,
        annot=heatmap_data.T.round(2),
        fmt='.2f',
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Normalized Value'}
    )
    
    ax.set_title('Cluster Profiles (Feature Means)')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Feature')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()
    return fig


def plot_cancellation_by_cluster(
    profiles: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot cancellation rate by cluster.
    
    Parameters
    ----------
    profiles : pd.DataFrame
        Cluster profiles with cancel_rate column
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    
    if 'cancel_rate' not in profiles.columns:
        print("⚠️ No cancel_rate in profiles")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    clusters = profiles['cluster'].astype(int).astype(str)
    rates = profiles['cancel_rate']
    sizes = profiles['size_pct']
    
    # Color based on cancellation rate
    colors = plt.cm.RdYlGn_r(rates / 100)
    
    bars = ax.bar(clusters, rates, color=colors, edgecolor='black')
    
    # Add size labels on top
    for bar, size in zip(bars, sizes):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f'{size:.1f}%',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    # Add threshold line
    overall_rate = (profiles['cancel_rate'] * profiles['size_pct']).sum() / profiles['size_pct'].sum()
    ax.axhline(y=overall_rate, color='blue', linestyle='--', label=f'Overall: {overall_rate:.1f}%')
    
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Cancellation Rate (%)')
    ax.set_title('Cancellation Rate by Cluster\n(percentages on bars = cluster size)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()
    return fig


# ====================
# MAIN PIPELINE
# ====================

def cluster_bookings(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    n_clusters: Optional[int] = None,
    algorithm: str = 'kmeans',
    scaling_method: str = 'standard',
    target_col: str = 'is_canceled',
    random_state: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Complete clustering pipeline for hotel bookings.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    features : list, optional
        Features for clustering
    n_clusters : int, optional
        Number of clusters (if None, finds optimal)
    algorithm : str
        Clustering algorithm: 'kmeans', 'dbscan', 'hierarchical'
    scaling_method : str
        Scaling method
    target_col : str
        Target column for profiling
    random_state : int
        Random seed
    verbose : bool
        Whether to print information
        
    Returns
    -------
    dict
        Results including labels, model, profiles, metrics
        
    Examples
    --------
    >>> results = cluster_bookings(df, n_clusters=5)
    >>> labels = results['labels']
    >>> profiles = results['profiles']
    """
    if verbose:
        print("\n" + "=" * 70)
        print("🔍 CLUSTERING PIPELINE")
        print("=" * 70 + "\n")
    
    # Step 1: Prepare data
    X_scaled, scaler, feature_names = prepare_clustering_data(
        df, features=features, scaling_method=scaling_method, verbose=verbose
    )
    
    # Step 2: Find optimal k if not specified (only for kmeans/hierarchical)
    optimal_k_results = None
    if n_clusters is None and algorithm in ['kmeans', 'hierarchical']:
        optimal_k_results = find_optimal_k(X_scaled, random_state=random_state, verbose=verbose)
        n_clusters = optimal_k_results['optimal_k_silhouette']
    
    # Step 3: Run clustering
    if algorithm == 'kmeans':
        labels, model = run_kmeans(X_scaled, n_clusters, random_state=random_state, verbose=verbose)
    elif algorithm == 'dbscan':
        labels, model = run_dbscan(X_scaled, verbose=verbose)
    elif algorithm == 'hierarchical':
        labels, model = run_hierarchical(X_scaled, n_clusters, verbose=verbose)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Step 4: Evaluate
    metrics = evaluate_clustering(X_scaled, labels, verbose=verbose)
    
    # Step 5: Profile clusters
    profiles = profile_clusters(df.iloc[X_scaled.index], labels, feature_names, target_col, verbose=verbose)
    
    # Step 6: Identify high-risk clusters
    high_risk = identify_high_risk_clusters(profiles, verbose=verbose)
    
    results = {
        'labels': labels,
        'model': model,
        'scaler': scaler,
        'features': feature_names,
        'X_scaled': X_scaled,
        'metrics': metrics,
        'profiles': profiles,
        'high_risk_clusters': high_risk,
        'optimal_k_results': optimal_k_results
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print("✅ CLUSTERING PIPELINE COMPLETE")
        print("=" * 70 + "\n")
    
    return results


# ====================
# EXPORT FUNCTIONS
# ====================

__all__ = [
    # Constants
    'DEFAULT_CLUSTERING_FEATURES',
    
    # Data preparation
    'prepare_clustering_data',
    
    # Optimal k
    'find_optimal_k',
    'plot_elbow_silhouette',
    
    # Algorithms
    'run_kmeans',
    'run_dbscan',
    'run_hierarchical',
    
    # Evaluation
    'evaluate_clustering',
    
    # Profiling
    'profile_clusters',
    'identify_high_risk_clusters',
    
    # Visualization
    'reduce_dimensions',
    'plot_clusters_2d',
    'plot_cluster_profiles',
    'plot_cancellation_by_cluster',
    
    # Pipeline
    'cluster_bookings'
]


# ====================
# MAIN (Testing)
# ====================

if __name__ == "__main__":
    print("Testing clustering module...")
    print("=" * 70)
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    sample_data = pd.DataFrame({
        'lead_time': np.random.exponential(50, n),
        'total_nights': np.random.poisson(3, n),
        'total_guests': np.random.poisson(2, n) + 1,
        'adr': np.random.normal(100, 30, n),
        'is_canceled': np.random.choice([0, 1], n, p=[0.63, 0.37])
    })
    
    print(f"Sample data shape: {sample_data.shape}")
    
    # Run clustering
    results = cluster_bookings(
        sample_data,
        n_clusters=4,
        verbose=True
    )
    
    print(f"\nNumber of clusters: {len(set(results['labels']))}")
    print(f"High-risk clusters: {results['high_risk_clusters']}")
    
    print("\n✅ Clustering module test passed!")
