import numpy as np
import unittest
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

class Node:
    """A node in the decision tree."""
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeClassifier:
    """A simplified Decision Tree Classifier with detailed TODOs."""
    def __init__(self, max_depth=1, min_samples_split=2):
        """
        Parameters
        ----------
        max_depth : int, default=1
            The maximum depth the tree can grow to.
        min_samples_split : int, default=2
            The minimum number of samples a node must have to be considered for splitting.
        Attributes
        ----------
        root : Node
            The root node of the fitted decision tree.
        n_classes_ : int
            The number of classes discovered in the target variable `y`.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_classes_ = None
    def fit(self, X, y, sample_weight=None):
        """Build a decision tree classifier from the training set (X, y).
        Parameters
        ----------
        X : np.ndarray
            The training input samples.
        y : np.ndarray
            The target values (class labels).
        sample_weight : np.ndarray, default=None
            Sample weights. If None, then samples are equally weighted.
        """
        self.n_classes_ = len(np.unique(y))
        self.root = self._grow_tree(X, y)
    def _grow_tree(self, X, y, depth=0):
        """Recursively grow the decision tree.
        Parameters
        ----------
        X : np.ndarray
            The training input samples for the current node.
        y : np.ndarray
            The target values (class labels) for the current node.
        depth : int, default=0
            The current depth of the node in the tree.
        Returns
        -------
        Node
            The root node of the built subtree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        
        is_max_depth_reached = depth >= self.max_depth
        is_one_class = n_labels == 1
        is_min_samples_not_met = n_samples < self.min_samples_split
        
        if is_max_depth_reached or is_one_class or is_min_samples_not_met:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is not None:
            
            left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
            
            left_child = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
            right_child = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
            return Node(best_feature, best_threshold, left_child, right_child)
        
        leaf_value = self._most_common_label(y)
        
        return Node(value=leaf_value)
    def _find_best_split(self, X, y):
        """Find the best split for a node.
        Iterates through all features and their unique values to find the
        split that results in the highest information gain.
        Parameters
        ----------
        X : np.ndarray
            The input samples for the current node.
        y : np.ndarray
            The class labels for the current node.
        Returns
        -------
        tuple
            A tuple containing the index of the best feature and the best
            threshold value. Returns (None, None) if no split improves gain.
        """
        best_gain = 0.0
        best_feature, best_threshold = None, None
        n_samples, n_features = X.shape
        
        
        for feat_idx in range(n_features):
            thresholds = np.unique(X[:, feat_idx])
            for thr in thresholds:
                gain = self._information_gain(y, X[:, feat_idx], thr)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = thr
        
        return best_feature, best_threshold
    def _information_gain(self, y, X_column, threshold):
        """Calculate the information gain of a split.
        Information Gain = Gini(parent) - [weighted average Gini(children)]
        Parameters
        ----------
        y : np.ndarray
            The class labels for the parent node.
        X_column : np.ndarray
            The feature column to be split.
        threshold : float
            The threshold value to split the feature column on.
        Returns
        -------
        float
            The information gain from the split.
        """
        
        parent_impurity = self._gini(y)
        
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0.0
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        gini_left = self._gini(y[left_idxs])
        gini_right = self._gini(y[right_idxs])
        child_impurity = (n_l / n) * gini_left + (n_r / n) * gini_right
        information_gain = parent_impurity - child_impurity
        
        return information_gain
    def _split(self, X_column, split_thresh):
        """Split a feature column by a threshold into left and right indices.
        Parameters
        ----------
        X_column : np.ndarray
            A single feature column from the dataset.
        split_thresh : float
            The threshold at which to split the column.
        Returns
        -------
        tuple
            A tuple containing two numpy arrays: the indices for the left
            child (<= threshold) and the indices for the right child (> threshold).
        """
        
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
        
    def _gini(self, y):
        """Calculate the Gini impurity of a set of labels.
        The Gini impurity is a measure of how often a randomly chosen element
        from the set would be incorrectly labeled.
        Parameters
        ----------
        y : np.ndarray
            An array of class labels.
        Returns
        -------
        float
            The Gini impurity of the labels.
        """
        
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes_)
        probabilities = counts / counts.sum()
        gini = 1.0 - np.sum(probabilities ** 2)
        
        return gini
    def _most_common_label(self, y):
        """Find the most common label in a set of labels.
        Parameters
        ----------
        y : np.ndarray
            An array of class labels.
        Returns
        -------
        int
            The most frequently occurring label.
        """
        
        counts = np.bincount(y, minlength=self.n_classes_)
        most_common = int(np.argmax(counts))
        
        return most_common
    def predict(self, X):
        """Predict class labels for samples in X.
        Parameters
        ----------
        X : np.ndarray
            The input samples to predict.
        Returns
        -------
        np.ndarray
            An array of predicted class labels.
        """
        
        return np.array([self._traverse_tree(x, self.root) for x in X])
    def _traverse_tree(self, x, node):
        """Recursively traverse the tree to predict the label for a single sample.
        Parameters
        ----------
        x : np.ndarray
            A single sample from the input data.
        node : Node
            The current node in the tree from which to traverse.
        Returns
        -------
        int
            The predicted class label from the leaf node.
        """
        
        
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            predicted_label = self._traverse_tree(x, node.left)
        else:
            predicted_label = self._traverse_tree(x, node.right)
        
        return predicted_label

if __name__ == "__main__":
    print("--- 1. Simple 1D Smoke Test ---")
    
    X_simple = np.array([[1], [2], [3], [4]])
    y_simple = np.array([0, 0, 1, 1])
    print(f"Simple X:\n{X_simple}")
    print(f"Simple y:\n{y_simple}")
    
    
    clf_simple = DecisionTreeClassifier(max_depth=1)
    clf_simple.fit(X_simple, y_simple)
    
    if clf_simple.root and not clf_simple.root.is_leaf_node():
        print("\nFitted Tree (Root Node):")
        print(f"  Split Feature: {clf_simple.root.feature}")
        print(f"  Split Threshold: {clf_simple.root.threshold}")
        print(f"  Left Child Value: {clf_simple.root.left.value}")
        print(f"  Right Child Value: {clf_simple.root.right.value}")
    else:
        print("\nTree has no splits (root is a leaf). Check implementation.")
    
    X_test_simple = np.array([[1.5], [2.0], [2.5], [5.0]])
    y_pred_simple = clf_simple.predict(X_test_simple)
    print(f"\nTest Data:\n{X_test_simple}")
    print(f"Predictions: {y_pred_simple}")
    print(f"Expected: [0 0 1 1]")  
    print("\n\n--- 2. Sklearn make_classification Test ---")
    X_complex, y_complex = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=0,
    )
    
    clf_complex = DecisionTreeClassifier(max_depth=1)
    clf_complex.fit(X_complex, y_complex)
    
    y_pred_complex = clf_complex.predict(X_complex)
    
    
    accuracy = accuracy_score(y_complex, y_pred_complex)
    print(f"Complex Test - Weak Learner | Training Accuracy: {accuracy:.4f}")
    print("A correct implementation should achieve moderate accuracy here.")
    
    clf_complex = DecisionTreeClassifier(max_depth=5)
    clf_complex.fit(X_complex, y_complex)
    
    y_pred_complex = clf_complex.predict(X_complex)
    
    
    accuracy = accuracy_score(y_complex, y_pred_complex)
    print(f"Complex Test - Better Learner | Training Accuracy: {accuracy:.4f}")
    print("A correct implementation should achieve better accuracy here.")
