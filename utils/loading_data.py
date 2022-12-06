import pandas as pd
import numpy as np

def load_to_df_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df

def get_all_feature_combinations(feature_set):
    """A function that takes a set and produces all subsets"""
    from itertools import chain, combinations
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(feature_set)
    
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

def test_get_all_feature_combinations(feature_set):
    """A function that takes a feature set and produces all feature combinations"""

def get_all_feature_combinations(data_columns):
    from itertools import chain, combinations
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    feature_combinations = list(chain.from_iterable(combinations(data_columns, r) for r in range(len(data_columns)+1)))

    feature_combinations_set = []
    for feature_combination in feature_combinations:
        feature_combination_set = []
        for feature in feature_combination:
            feature_combination_set.append(feature)
        
        feature_combinations_set.append(feature_combination_set)

    return feature_combinations_set

