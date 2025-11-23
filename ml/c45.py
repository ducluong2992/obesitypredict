# c45.py
import pandas as pd
from collections import Counter
import math

class Node:
    def __init__(self, attribute=None, threshold=None, leaf_class=None):
        self.attribute = attribute       # Thuộc tính phân nhánh
        self.threshold = threshold       # Nếu numeric
        self.leaf_class = leaf_class     # Nếu là lá
        self.children = {}               # Nếu nominal: value -> Node

# Entropy
def entropy(y):
    counts = Counter(y)
    total = len(y)
    return -sum((c/total) * math.log2(c/total) for c in counts.values() if c>0)

# Gain Ratio cho nominal
def gain_ratio_nominal(X_column, y):
    total_entropy = entropy(y)
    n = len(y)
    values = set(X_column)
    
    subset_entropy = 0
    split_info = 0
    for v in values:
        subset_y = [y[i] for i in range(n) if X_column[i]==v]
        p = len(subset_y)/n
        subset_entropy += p * entropy(subset_y)
        split_info -= p * math.log2(p) if p>0 else 0
    
    info_gain = total_entropy - subset_entropy
    if split_info==0:
        return 0
    return info_gain / split_info

# Gain Ratio cho numeric
def gain_ratio_numeric(X_column, y):
    sorted_idx = sorted(range(len(X_column)), key=lambda i: X_column[i])
    X_sorted = [X_column[i] for i in sorted_idx]
    y_sorted = [y[i] for i in sorted_idx]
    
    best_gain_ratio = 0
    best_threshold = None
    total_entropy = entropy(y)
    
    for i in range(1, len(X_sorted)):
        if X_sorted[i] == X_sorted[i-1]:
            continue
        threshold = (X_sorted[i] + X_sorted[i-1])/2
        left_y = [y_sorted[j] for j in range(len(y_sorted)) if X_sorted[j]<=threshold]
        right_y = [y_sorted[j] for j in range(len(y_sorted)) if X_sorted[j]>threshold]
        p_left = len(left_y)/len(y)
        p_right = len(right_y)/len(y)
        subset_entropy = p_left*entropy(left_y) + p_right*entropy(right_y)
        info_gain = total_entropy - subset_entropy
        split_info = 0
        for p in [p_left, p_right]:
            if p>0:
                split_info -= p*math.log2(p)
        gain_ratio = info_gain / split_info if split_info>0 else 0
        if gain_ratio > best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_threshold = threshold
    return best_gain_ratio, best_threshold

# Build tree
def build_tree(Xy, target_col, features):
    y = Xy[target_col].tolist()
    if len(set(y))==1:
        return Node(leaf_class=y[0])
    if len(features)==0 or Xy.empty:
        majority_class = Counter(y).most_common(1)[0][0]
        return Node(leaf_class=majority_class)
    
    best_gain_ratio = -1
    best_attr = None
    best_threshold = None
    
    for attr in features:
        X_col = Xy[attr].tolist()
        if pd.api.types.is_numeric_dtype(Xy[attr]):
            gain, threshold = gain_ratio_numeric(X_col, y)
        else:
            gain = gain_ratio_nominal(X_col, y)
            threshold = None
        if gain > best_gain_ratio:
            best_gain_ratio = gain
            best_attr = attr
            best_threshold = threshold

    if best_gain_ratio <= 0:
        majority_class = Counter(y).most_common(1)[0][0]
        return Node(leaf_class=majority_class)
    
    node = Node(attribute=best_attr, threshold=best_threshold)

    if best_threshold is None:
        values = set(Xy[best_attr])
        for v in values:
            subset = Xy[Xy[best_attr]==v].drop(columns=[best_attr])
            node.children[v] = build_tree(pd.concat([subset, Xy.loc[subset.index, [target_col]]], axis=1), target_col, subset.columns.tolist())
    else:
        left = Xy[Xy[best_attr]<=best_threshold]
        right = Xy[Xy[best_attr]>best_threshold]
        node.children['le'] = build_tree(left, target_col, left.drop(columns=[target_col]).columns.tolist())
        node.children['gt'] = build_tree(right, target_col, right.drop(columns=[target_col]).columns.tolist())

    return node

# Predict 1 row
def predict(node, x_dict):
    if node.leaf_class is not None:
        return node.leaf_class
    if node.threshold is None:
        value = x_dict[node.attribute]
        if value in node.children:
            return predict(node.children[value], x_dict)
        else:
            child_classes = [child.leaf_class for child in node.children.values() if child.leaf_class is not None]
            return Counter(child_classes).most_common(1)[0][0]
    else:
        if x_dict[node.attribute] <= node.threshold:
            return predict(node.children['le'], x_dict)
        else:
            return predict(node.children['gt'], x_dict)
