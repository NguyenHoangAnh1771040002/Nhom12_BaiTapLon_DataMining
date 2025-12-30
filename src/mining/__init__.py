"""
Mining Module
=============

Data mining algorithms including association rules and clustering.

Submodules:
-----------
- association: Apriori, FP-Growth algorithms for association rule mining
- clustering: KMeans, DBSCAN, Hierarchical clustering (Phase 4.2)
"""

from .association import (
    # Preparation
    prepare_transactions,
    convert_to_transaction_list,
    
    # Algorithms
    run_apriori,
    run_fpgrowth,
    
    # Rules extraction
    extract_rules,
    filter_rules_by_consequent,
    filter_rules_by_antecedent,
    
    # Analysis
    compare_rules_by_group,
    get_top_rules,
    summarize_rules,
    
    # Visualization
    plot_rules_heatmap,
    plot_support_confidence_scatter,
    rules_to_network_data,
    
    # Pipeline
    mine_association_rules,
    
    # Constants
    MLXTEND_AVAILABLE
)

# Clustering imports
from .clustering import (
    # Constants
    DEFAULT_CLUSTERING_FEATURES,
    
    # Data preparation
    prepare_clustering_data,
    
    # Optimal k
    find_optimal_k,
    plot_elbow_silhouette,
    
    # Algorithms
    run_kmeans,
    run_dbscan,
    run_hierarchical,
    
    # Evaluation
    evaluate_clustering,
    
    # Profiling
    profile_clusters,
    identify_high_risk_clusters,
    
    # Visualization
    reduce_dimensions,
    plot_clusters_2d,
    plot_cluster_profiles,
    plot_cancellation_by_cluster,
    
    # Pipeline
    cluster_bookings
)

__all__ = [
    # Association Rules
    'prepare_transactions',
    'convert_to_transaction_list',
    'run_apriori',
    'run_fpgrowth',
    'extract_rules',
    'filter_rules_by_consequent',
    'filter_rules_by_antecedent',
    'compare_rules_by_group',
    'get_top_rules',
    'summarize_rules',
    'plot_rules_heatmap',
    'plot_support_confidence_scatter',
    'rules_to_network_data',
    'mine_association_rules',
    'MLXTEND_AVAILABLE',
    
    # Clustering
    'DEFAULT_CLUSTERING_FEATURES',
    'prepare_clustering_data',
    'find_optimal_k',
    'plot_elbow_silhouette',
    'run_kmeans',
    'run_dbscan',
    'run_hierarchical',
    'evaluate_clustering',
    'profile_clusters',
    'identify_high_risk_clusters',
    'reduce_dimensions',
    'plot_clusters_2d',
    'plot_cluster_profiles',
    'plot_cancellation_by_cluster',
    'cluster_bookings'
]
