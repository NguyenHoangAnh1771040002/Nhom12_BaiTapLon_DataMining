"""
Module Khai Phá Luật Kết Hợp (Association Rules Mining)
======================================================

Các hàm khai phá luật kết hợp từ dữ liệu đặt phòng khách sạn.

Các hàm chính:
--------------
- prepare_transactions: Chuyển dataframe sang định dạng giao dịch
- run_apriori: Chạy thuật toán Apriori
- run_fpgrowth: Chạy thuật toán FP-Growth
- extract_rules: Trích xuất luật với các chỉ số đánh giá
- filter_rules_by_consequent: Lọc luật theo vế phải
- compare_rules_by_group: So sánh luật giữa các nhóm
- visualize_rules: Trực quan hóa luật kết hợp
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import warnings

# Import mlxtend for association rules
try:
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    warnings.warn("mlxtend not installed. Run: pip install mlxtend")

warnings.filterwarnings('ignore')


# ====================
# DATA PREPARATION
# ====================

def prepare_transactions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    include_target: bool = True,
    target_col: str = 'is_canceled',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Prepare data for association rule mining.
    
    Converts categorical columns to one-hot encoded boolean format
    suitable for Apriori/FP-Growth algorithms.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with categorical features
    columns : list, optional
        Columns to include. If None, uses default columns.
    include_target : bool
        Whether to include target column (is_canceled)
    target_col : str
        Name of target column
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        Boolean DataFrame for association rules mining
        
    Examples
    --------
    >>> df_trans = prepare_transactions(df)
    >>> # Run apriori on df_trans
    """
    df_work = df.copy()
    
    # Default columns for association rules
    if columns is None:
        columns = [
            'hotel', 'arrival_season', 'meal', 'country_grouped',
            'market_segment', 'deposit_type', 'customer_type',
            'lead_time_category', 'is_family', 'is_returning_customer',
            'has_special_requests', 'room_type_changed'
        ]
    
    # Filter to existing columns
    columns = [col for col in columns if col in df_work.columns]
    
    if verbose:
        print("=" * 60)
        print("PREPARING DATA FOR ASSOCIATION RULES")
        print("=" * 60)
        print(f"Input shape: {df_work.shape}")
        print(f"Selected columns: {columns}")
    
    # Add target if requested
    if include_target and target_col in df_work.columns:
        # Convert target to categorical
        df_work['canceled'] = df_work[target_col].map({0: 'Not_Canceled', 1: 'Canceled'})
        columns.append('canceled')
    
    # Select only these columns
    df_subset = df_work[columns].copy()
    
    # Convert numeric columns to categorical strings
    for col in df_subset.columns:
        if df_subset[col].dtype in ['int64', 'float64']:
            # For binary columns
            if df_subset[col].nunique() <= 2:
                df_subset[col] = df_subset[col].map({0: f'{col}=No', 1: f'{col}=Yes'})
            else:
                df_subset[col] = df_subset[col].astype(str)
    
    # One-hot encode all columns
    df_encoded = pd.get_dummies(df_subset, prefix_sep='=')
    
    # Convert to boolean
    df_encoded = df_encoded.astype(bool)
    
    if verbose:
        print(f"Output shape: {df_encoded.shape}")
        print(f"Number of items: {len(df_encoded.columns)}")
        print("=" * 60)
    
    return df_encoded


def convert_to_transaction_list(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> List[List[str]]:
    """
    Convert dataframe to list of transactions.
    
    Each transaction is a list of items present in that row.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        Columns to include
        
    Returns
    -------
    list
        List of transactions (list of items)
    """
    if columns is None:
        columns = df.columns.tolist()
    
    transactions = []
    for _, row in df[columns].iterrows():
        # Get items that are True or non-null
        items = []
        for col in columns:
            val = row[col]
            if isinstance(val, bool):
                if val:
                    items.append(col)
            elif pd.notna(val):
                items.append(f"{col}={val}")
        transactions.append(items)
    
    return transactions


# ====================
# FREQUENT ITEMSETS
# ====================

def run_apriori(
    df: pd.DataFrame,
    min_support: float = 0.05,
    use_colnames: bool = True,
    max_len: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run Apriori algorithm to find frequent itemsets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Boolean DataFrame (output of prepare_transactions)
    min_support : float
        Minimum support threshold (0 to 1)
    use_colnames : bool
        Use column names instead of indices
    max_len : int, optional
        Maximum length of itemsets
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        Frequent itemsets with support values
        
    Examples
    --------
    >>> frequent = run_apriori(df_trans, min_support=0.05)
    """
    if not MLXTEND_AVAILABLE:
        raise ImportError("mlxtend not installed. Run: pip install mlxtend")
    
    if verbose:
        print("=" * 60)
        print("RUNNING APRIORI ALGORITHM")
        print("=" * 60)
        print(f"Min support: {min_support}")
        print(f"Max itemset length: {max_len or 'None'}")
    
    # Run Apriori
    frequent_itemsets = apriori(
        df, 
        min_support=min_support, 
        use_colnames=use_colnames,
        max_len=max_len
    )
    
    # Add itemset length
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
    
    # Sort by support
    frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)
    
    if verbose:
        print(f"Found {len(frequent_itemsets)} frequent itemsets")
        print(f"Support range: [{frequent_itemsets['support'].min():.4f}, {frequent_itemsets['support'].max():.4f}]")
        print("=" * 60)
    
    return frequent_itemsets


def run_fpgrowth(
    df: pd.DataFrame,
    min_support: float = 0.05,
    use_colnames: bool = True,
    max_len: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run FP-Growth algorithm to find frequent itemsets.
    
    FP-Growth is generally faster than Apriori for large datasets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Boolean DataFrame (output of prepare_transactions)
    min_support : float
        Minimum support threshold (0 to 1)
    use_colnames : bool
        Use column names instead of indices
    max_len : int, optional
        Maximum length of itemsets
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        Frequent itemsets with support values
        
    Examples
    --------
    >>> frequent = run_fpgrowth(df_trans, min_support=0.05)
    """
    if not MLXTEND_AVAILABLE:
        raise ImportError("mlxtend not installed. Run: pip install mlxtend")
    
    if verbose:
        print("=" * 60)
        print("RUNNING FP-GROWTH ALGORITHM")
        print("=" * 60)
        print(f"Min support: {min_support}")
        print(f"Max itemset length: {max_len or 'None'}")
    
    # Run FP-Growth
    frequent_itemsets = fpgrowth(
        df, 
        min_support=min_support, 
        use_colnames=use_colnames,
        max_len=max_len
    )
    
    # Add itemset length
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
    
    # Sort by support
    frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)
    
    if verbose:
        print(f"Found {len(frequent_itemsets)} frequent itemsets")
        print(f"Support range: [{frequent_itemsets['support'].min():.4f}, {frequent_itemsets['support'].max():.4f}]")
        print("=" * 60)
    
    return frequent_itemsets


# ====================
# ASSOCIATION RULES
# ====================

def extract_rules(
    frequent_itemsets: pd.DataFrame,
    metric: str = 'confidence',
    min_threshold: float = 0.5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract association rules from frequent itemsets.
    
    Parameters
    ----------
    frequent_itemsets : pd.DataFrame
        Output from run_apriori or run_fpgrowth
    metric : str
        Metric to evaluate rules: 'confidence', 'lift', 'leverage', 'conviction'
    min_threshold : float
        Minimum threshold for the metric
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        Association rules with metrics
        
    Examples
    --------
    >>> rules = extract_rules(frequent, metric='confidence', min_threshold=0.6)
    """
    if not MLXTEND_AVAILABLE:
        raise ImportError("mlxtend not installed. Run: pip install mlxtend")
    
    if verbose:
        print("=" * 60)
        print("EXTRACTING ASSOCIATION RULES")
        print("=" * 60)
        print(f"Metric: {metric}")
        print(f"Min threshold: {min_threshold}")
    
    # Generate rules
    rules = association_rules(
        frequent_itemsets, 
        metric=metric, 
        min_threshold=min_threshold
    )
    
    if len(rules) == 0:
        if verbose:
            print("⚠️ No rules found! Try lowering the threshold.")
        return rules
    
    # Convert frozensets to strings for readability
    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    # Add rule string
    rules['rule'] = rules['antecedents_str'] + ' → ' + rules['consequents_str']
    
    # Sort by lift (most interesting rules first)
    rules = rules.sort_values('lift', ascending=False)
    
    if verbose:
        print(f"Found {len(rules)} rules")
        print(f"Lift range: [{rules['lift'].min():.2f}, {rules['lift'].max():.2f}]")
        print(f"Confidence range: [{rules['confidence'].min():.2f}, {rules['confidence'].max():.2f}]")
        print("=" * 60)
    
    return rules


def filter_rules_by_consequent(
    rules: pd.DataFrame,
    consequent_items: List[str],
    match_any: bool = True
) -> pd.DataFrame:
    """
    Filter rules by consequent items.
    
    Useful for finding rules that lead to cancellation.
    
    Parameters
    ----------
    rules : pd.DataFrame
        Association rules DataFrame
    consequent_items : list
        Items to filter by (e.g., ['canceled=Canceled'])
    match_any : bool
        If True, match any item. If False, match all items.
        
    Returns
    -------
    pd.DataFrame
        Filtered rules
        
    Examples
    --------
    >>> cancel_rules = filter_rules_by_consequent(rules, ['canceled=Canceled'])
    """
    if match_any:
        mask = rules['consequents'].apply(
            lambda x: any(item in x for item in consequent_items)
        )
    else:
        mask = rules['consequents'].apply(
            lambda x: all(item in x for item in consequent_items)
        )
    
    return rules[mask].copy()


def filter_rules_by_antecedent(
    rules: pd.DataFrame,
    antecedent_items: List[str],
    match_any: bool = True
) -> pd.DataFrame:
    """
    Filter rules by antecedent items.
    
    Parameters
    ----------
    rules : pd.DataFrame
        Association rules DataFrame
    antecedent_items : list
        Items to filter by
    match_any : bool
        If True, match any item. If False, match all items.
        
    Returns
    -------
    pd.DataFrame
        Filtered rules
    """
    if match_any:
        mask = rules['antecedents'].apply(
            lambda x: any(item in x for item in antecedent_items)
        )
    else:
        mask = rules['antecedents'].apply(
            lambda x: all(item in x for item in antecedent_items)
        )
    
    return rules[mask].copy()


# ====================
# COMPARISON & ANALYSIS
# ====================

def compare_rules_by_group(
    df: pd.DataFrame,
    group_col: str,
    group_values: List,
    columns: Optional[List[str]] = None,
    min_support: float = 0.05,
    min_confidence: float = 0.5,
    consequent_filter: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Compare association rules across different groups.
    
    Useful for comparing rules in summer vs winter, or across countries.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    group_col : str
        Column to group by (e.g., 'arrival_season')
    group_values : list
        Values to compare (e.g., ['Summer', 'Winter'])
    columns : list, optional
        Columns for transaction encoding
    min_support : float
        Minimum support for frequent itemsets
    min_confidence : float
        Minimum confidence for rules
    consequent_filter : list, optional
        Filter rules by consequent (e.g., ['canceled=Canceled'])
    verbose : bool
        Whether to print information
        
    Returns
    -------
    dict
        Dictionary mapping group value to rules DataFrame
        
    Examples
    --------
    >>> rules_by_season = compare_rules_by_group(
    ...     df, 'arrival_season', ['Summer', 'Winter']
    ... )
    """
    results = {}
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"COMPARING RULES BY {group_col.upper()}")
        print("=" * 70)
    
    for value in group_values:
        if verbose:
            print(f"\n📊 Processing: {group_col} = {value}")
            print("-" * 50)
        
        # Filter data for this group
        df_group = df[df[group_col] == value].copy()
        
        if len(df_group) == 0:
            if verbose:
                print(f"⚠️ No data for {value}")
            continue
        
        if verbose:
            print(f"   Rows: {len(df_group)}")
        
        # Remove the group column from transactions
        cols_to_use = columns.copy() if columns else None
        if cols_to_use and group_col in cols_to_use:
            cols_to_use.remove(group_col)
        
        # Prepare transactions
        df_trans = prepare_transactions(
            df_group, 
            columns=cols_to_use,
            verbose=False
        )
        
        # Run FP-Growth (faster than Apriori)
        try:
            frequent = run_fpgrowth(df_trans, min_support=min_support, verbose=False)
            
            if len(frequent) == 0:
                if verbose:
                    print(f"   ⚠️ No frequent itemsets found")
                continue
            
            # Extract rules
            rules = extract_rules(frequent, min_threshold=min_confidence, verbose=False)
            
            if len(rules) == 0:
                if verbose:
                    print(f"   ⚠️ No rules found")
                continue
            
            # Filter by consequent if specified
            if consequent_filter:
                rules = filter_rules_by_consequent(rules, consequent_filter)
            
            results[value] = rules
            
            if verbose:
                print(f"   ✓ Found {len(rules)} rules")
                if len(rules) > 0:
                    print(f"   Top rule by lift: {rules.iloc[0]['rule']}")
                    print(f"   Lift: {rules.iloc[0]['lift']:.2f}, Confidence: {rules.iloc[0]['confidence']:.2f}")
                    
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Error: {str(e)}")
    
    return results


def get_top_rules(
    rules: pd.DataFrame,
    n: int = 10,
    sort_by: str = 'lift'
) -> pd.DataFrame:
    """
    Get top N rules sorted by a metric.
    
    Parameters
    ----------
    rules : pd.DataFrame
        Association rules DataFrame
    n : int
        Number of top rules
    sort_by : str
        Metric to sort by: 'lift', 'confidence', 'support'
        
    Returns
    -------
    pd.DataFrame
        Top N rules
    """
    return rules.nlargest(n, sort_by)[
        ['rule', 'support', 'confidence', 'lift', 'leverage', 'conviction']
    ].reset_index(drop=True)


def summarize_rules(
    rules: pd.DataFrame,
    name: str = "Rules"
) -> pd.DataFrame:
    """
    Get summary statistics of association rules.
    
    Parameters
    ----------
    rules : pd.DataFrame
        Association rules DataFrame
    name : str
        Name for the summary
        
    Returns
    -------
    pd.DataFrame
        Summary statistics
    """
    if len(rules) == 0:
        return pd.DataFrame()
    
    summary = {
        'Name': name,
        'Total Rules': len(rules),
        'Avg Support': rules['support'].mean(),
        'Avg Confidence': rules['confidence'].mean(),
        'Avg Lift': rules['lift'].mean(),
        'Max Lift': rules['lift'].max(),
        'Rules with Lift > 1': (rules['lift'] > 1).sum()
    }
    
    return pd.DataFrame([summary])


# ====================
# VISUALIZATION
# ====================

def plot_rules_heatmap(
    rules: pd.DataFrame,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
):
    """
    Plot heatmap of top association rules.
    
    Parameters
    ----------
    rules : pd.DataFrame
        Association rules DataFrame
    top_n : int
        Number of top rules to plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get top rules
    top_rules = rules.nlargest(top_n, 'lift').copy()
    
    # Create pivot for heatmap
    # Use shortened rule names
    top_rules['short_rule'] = top_rules.apply(
        lambda x: f"R{x.name}: {str(list(x['antecedents']))[:30]}...", 
        axis=1
    )
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    metrics = ['support', 'confidence', 'lift']
    titles = ['Support', 'Confidence', 'Lift']
    
    for ax, metric, title in zip(axes, metrics, titles):
        data = top_rules[['rule', metric]].set_index('rule')
        
        # Truncate rule names for display
        data.index = [r[:40] + '...' if len(r) > 40 else r for r in data.index]
        
        sns.barplot(x=metric, y=data.index, data=data.reset_index(), ax=ax, palette='viridis')
        ax.set_title(title)
        ax.set_xlabel(metric.capitalize())
        ax.set_ylabel('')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()
    
    return fig


def plot_support_confidence_scatter(
    rules: pd.DataFrame,
    color_by: str = 'lift',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Scatter plot of support vs confidence, colored by lift.
    
    Parameters
    ----------
    rules : pd.DataFrame
        Association rules DataFrame
    color_by : str
        Metric to color by
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(
        rules['support'],
        rules['confidence'],
        c=rules[color_by],
        cmap='viridis',
        alpha=0.6,
        s=50
    )
    
    plt.colorbar(scatter, label=color_by.capitalize())
    ax.set_xlabel('Support')
    ax.set_ylabel('Confidence')
    ax.set_title('Association Rules: Support vs Confidence')
    
    # Add threshold lines
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Confidence = 0.5')
    ax.axvline(x=0.05, color='b', linestyle='--', alpha=0.5, label='Support = 0.05')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()
    
    return fig


def rules_to_network_data(
    rules: pd.DataFrame,
    top_n: int = 50
) -> Tuple[List[Dict], List[Dict]]:
    """
    Convert rules to network graph data format.
    
    Parameters
    ----------
    rules : pd.DataFrame
        Association rules DataFrame
    top_n : int
        Number of top rules
        
    Returns
    -------
    tuple
        (nodes, edges) for network visualization
    """
    top_rules = rules.nlargest(top_n, 'lift')
    
    nodes = set()
    edges = []
    
    for _, row in top_rules.iterrows():
        # Add antecedent nodes
        for item in row['antecedents']:
            nodes.add(item)
        
        # Add consequent nodes
        for item in row['consequents']:
            nodes.add(item)
        
        # Add edges
        for ant in row['antecedents']:
            for con in row['consequents']:
                edges.append({
                    'source': ant,
                    'target': con,
                    'weight': row['lift'],
                    'confidence': row['confidence']
                })
    
    nodes_list = [{'id': n, 'label': n} for n in nodes]
    
    return nodes_list, edges


# ====================
# MAIN PIPELINE
# ====================

def mine_association_rules(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    min_support: float = 0.05,
    min_confidence: float = 0.5,
    algorithm: str = 'fpgrowth',
    filter_canceled: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete pipeline for mining association rules.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with features
    columns : list, optional
        Columns for transaction encoding
    min_support : float
        Minimum support threshold
    min_confidence : float
        Minimum confidence threshold
    algorithm : str
        Algorithm: 'fpgrowth' or 'apriori'
    filter_canceled : bool
        Whether to filter rules related to cancellation
    verbose : bool
        Whether to print information
        
    Returns
    -------
    tuple
        (all_rules, canceled_rules)
        
    Examples
    --------
    >>> all_rules, cancel_rules = mine_association_rules(df)
    """
    if verbose:
        print("\n" + "=" * 70)
        print("🔍 ASSOCIATION RULES MINING PIPELINE")
        print("=" * 70 + "\n")
    
    # Step 1: Prepare transactions
    df_trans = prepare_transactions(df, columns=columns, verbose=verbose)
    
    # Step 2: Find frequent itemsets
    if algorithm == 'fpgrowth':
        frequent = run_fpgrowth(df_trans, min_support=min_support, verbose=verbose)
    else:
        frequent = run_apriori(df_trans, min_support=min_support, verbose=verbose)
    
    if len(frequent) == 0:
        print("⚠️ No frequent itemsets found! Try lowering min_support.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Step 3: Extract rules
    all_rules = extract_rules(frequent, min_threshold=min_confidence, verbose=verbose)
    
    if len(all_rules) == 0:
        print("⚠️ No rules found! Try lowering min_confidence.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Step 4: Filter rules for cancellation
    canceled_rules = pd.DataFrame()
    if filter_canceled:
        canceled_rules = filter_rules_by_consequent(
            all_rules, 
            ['canceled=Canceled']
        )
        
        if verbose:
            print("\n" + "=" * 60)
            print("RULES LEADING TO CANCELLATION")
            print("=" * 60)
            print(f"Found {len(canceled_rules)} rules where consequent = Canceled")
            
            if len(canceled_rules) > 0:
                print("\nTop 5 rules by lift:")
                top5 = get_top_rules(canceled_rules, n=5, sort_by='lift')
                for i, row in top5.iterrows():
                    print(f"  {i+1}. {row['rule']}")
                    print(f"     Lift: {row['lift']:.2f}, Confidence: {row['confidence']:.2f}")
    
    if verbose:
        print("\n" + "=" * 70)
        print("✅ ASSOCIATION RULES MINING COMPLETE")
        print("=" * 70 + "\n")
    
    return all_rules, canceled_rules


# ====================
# EXPORT FUNCTIONS
# ====================

__all__ = [
    # Preparation
    'prepare_transactions',
    'convert_to_transaction_list',
    
    # Algorithms
    'run_apriori',
    'run_fpgrowth',
    
    # Rules extraction
    'extract_rules',
    'filter_rules_by_consequent',
    'filter_rules_by_antecedent',
    
    # Analysis
    'compare_rules_by_group',
    'get_top_rules',
    'summarize_rules',
    
    # Visualization
    'plot_rules_heatmap',
    'plot_support_confidence_scatter',
    'rules_to_network_data',
    
    # Pipeline
    'mine_association_rules',
    
    # Constants
    'MLXTEND_AVAILABLE'
]


# ====================
# MAIN (Testing)
# ====================

if __name__ == "__main__":
    print("Testing association rules module...")
    print("=" * 70)
    
    # Check mlxtend availability
    if not MLXTEND_AVAILABLE:
        print("⚠️ mlxtend not installed. Run: pip install mlxtend")
        exit(1)
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    sample_data = pd.DataFrame({
        'hotel': np.random.choice(['City Hotel', 'Resort Hotel'], n),
        'arrival_season': np.random.choice(['Summer', 'Winter', 'Spring', 'Fall'], n),
        'deposit_type': np.random.choice(['No Deposit', 'Refundable', 'Non Refund'], n),
        'is_canceled': np.random.choice([0, 1], n, p=[0.63, 0.37])
    })
    
    print(f"Sample data shape: {sample_data.shape}")
    
    # Run pipeline
    all_rules, cancel_rules = mine_association_rules(
        sample_data,
        min_support=0.05,
        min_confidence=0.3,
        verbose=True
    )
    
    print(f"\nTotal rules: {len(all_rules)}")
    print(f"Cancellation rules: {len(cancel_rules)}")
    
    print("\n✅ Association rules module test passed!")
