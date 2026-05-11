"""
SCRIPT AUXILIAR QUE CONTIENE EL MODELO DE MACHINE LEARNING DE BOOSTING + TREE REGRESSOR
"""

import numpy as np


class Node:
    """
    Node of the decision tree.

    A node can represent either:
    - an internal decision node, which stores a feature index, a threshold,
      the impurity reduction achieved by the split, and references to its
      left and right child nodes, or
    - a leaf node, which stores the predicted continuous value.

    Attributes:
        feature_index : int, default=None
            Index of the feature used to split the data at this node.

        threshold : float, default=None
            Threshold value used to divide the data into left and right branches.

        left : Node, default=None
            Left child node. It contains the subtree for samples satisfying
            the split condition.

        right : Node, default=None
            Right child node. It contains the subtree for samples not satisfying
            the split condition.

        value : float or None, default=None
            Predicted value assigned to the node if it is a leaf.
            If None, the node is considered an internal decision node.

        impurity_reduction : float, default=0.0
            Reduction in impurity obtained by the split at this node.
            It is used later to compute feature importances.
    """

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None, impurity_reduction=0.0):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.impurity_reduction = impurity_reduction


class DecisionTreeRegressor:
    """
    Decision Tree regressor.

    This class implements a decision tree algorithm for regression tasks.
    The model recursively partitions the feature space into smaller regions
    by selecting the best feature and threshold at each step, with the goal
    of reducing the target variance.

    The tree grows until one of the stopping criteria is reached, such as:
    - the maximum depth,
    - the minimum number of samples required to split,
    - or nodes with constant target values.

    Attributes:
        max_depth : int or None, default=None
            Maximum depth allowed for the tree. If None, the tree grows until
            no further valid splits can be made.

        criterion : {"squared_error"}, default="squared_error"
            Criterion used to evaluate the quality of a split:
            - "squared_error": reduction of variance / MSE

        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node.

        root : Node or None
            Root node of the trained decision tree.

        n_features_in_ : int or None
            Number of features seen during fit.

        feature_importances_ : np.ndarray or None
            Normalized importance of each feature based on the accumulated
            impurity reduction over the whole tree.
    """

    def __init__(self, max_depth=None, criterion='squared_error', min_samples_split=2):
        """
        Initialize the Decision Tree regressor.

        Parameters:
            max_depth : int or None, default=None
                Maximum depth allowed for the tree.

            criterion : {"squared_error"}, default="squared_error"
                Function used to measure the quality of a split.

            min_samples_split : int, default=2
                Minimum number of samples required to split a node.
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_features_in_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        """
        Train the decision tree regressor.

        This method builds the tree recursively from the training data.

        Parameters:
            X : array-like of shape (n_samples, n_features)
                Training input samples.

            y : array-like of shape (n_samples,)
                Continuous target values.

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.n_features_in_ = X.shape[1]
        self.root = self._build_tree(X, y, depth=0)
        self.feature_importances_ = self._compute_feature_importances()

        return self

    def predict(self, X):
        """
        Predict continuous values for the input samples.

        Parameters:
            X : array-like of shape (n_samples, n_features)

        Returns:
            np.ndarray
                Predicted continuous values.
        """
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.

        Stopping conditions:
        - all samples have the same target value
        - minimum number of samples to split is not met
        - maximum depth is reached
        - no valid split is found
        """
        n_samples, n_features = X.shape

        # Si solamente hay un valor objetivo
        if len(np.unique(y)) == 1:
            return Node(value=float(y[0]))

        # Si no tenemos suficientes samples
        if n_samples < self.min_samples_split:
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value)

        # Si hemos alcanzado la profundidad máxima
        if self.max_depth is not None and depth >= self.max_depth:
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value)

        # Buscamos el mejor split
        best_feature, best_threshold, best_gain = self._best_split(X, y, n_features)

        # Si no encontramos el mejor split creamos un nodo hoja
        if best_feature is None:
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value)

        # Construimos el subarbol
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)

        # Seguridad extra por si el split resulta degenerado
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value)

        left_subtree = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_subtree = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
            impurity_reduction=best_gain
        )

    def _best_split(self, X, y, n_features):
        """
        Find the best feature and threshold for splitting the data.

        Returns:
            best_feature : int or None
            best_threshold : float or None
            best_gain : float
        """
        best_gain = -1.0
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            thresholds = np.unique(np.linspace(X_column.min(), X_column.max(), 10)) # en vez de np.unique(y) para reducir coste computacional

            for threshold in thresholds:
                gain = self._impurity_reduction(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _impurity_reduction(self, y, X_column, threshold):
        """
        Compute the impurity reduction of a split.

        For regression with squared error, this corresponds to the reduction
        in variance (equivalently, weighted MSE reduction).
        """
        parent_impurity = self._impurity(y)

        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0.0

        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)

        left_impurity = self._impurity(y[left_idxs])
        right_impurity = self._impurity(y[right_idxs])

        child_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity

        gain = parent_impurity - child_impurity
        return gain

    def _impurity(self, y):
        """
        Compute node impurity according to the selected criterion.
        """
        if self.criterion == 'squared_error':
            return self._variance(y)
        else:
            raise ValueError("criterion must be 'squared_error'")

    def _variance(self, y):
        """
        Compute variance of the target values in a node.
        """
        if len(y) == 0:
            return 0.0
        return np.var(y)

    def _split(self, X_column, threshold):
        """
        Split indices into left and right subsets.

        Left: values <= threshold
        Right: values > threshold
        """
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs

    def _leaf_value(self, y):
        """
        Return the prediction value of a leaf.

        In regression, a leaf predicts the mean target value of the samples
        contained in that node.
        """
        return float(np.mean(y))

    def _traverse_tree(self, x, node):
        """
        Returns the tree's prediction for a single sample.
        """
        # Nodo hoja: devolver el valor directamente
        if node.value is not None:
            return node.value

        # Nodo interno: bajar por la rama correspondiente
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _compute_feature_importances(self):
        """
        Compute feature importances as accumulated impurity reduction.

        The importance of each feature is the total reduction in impurity
        brought by all splits that use that feature, normalized so that
        the importances sum to 1.
        """
        importances = np.zeros(self.n_features_in_, dtype=float)
        self._accumulate_importances(self.root, importances)

        total = importances.sum()
        if total > 0:
            importances /= total

        return importances

    def _accumulate_importances(self, node, importances):
        """
        Traverse the tree recursively and accumulate split gains.
        """
        if node is None:
            return

        if node.value is not None:
            return

        importances[node.feature_index] += node.impurity_reduction

        self._accumulate_importances(node.left, importances)
        self._accumulate_importances(node.right, importances)


class GradientBoostingRegressor:
    """
    Gradient Boosting Regressor with squared error loss.

    The algorithm works as follows:
        y_hat_0 = mean(y)
        for t in 1..n_estimators:
            r_i = y_i - y_hat_i              # residuals / negative gradients
            h_t = tree.fit(X, r)             # fit a regression tree on residuals
            y_hat += learning_rate * h_t.predict(X)

    Attributes:
        n_estimators : int
            Number of boosting iterations / trees.

        learning_rate : float
            Shrinkage factor applied to the prediction of each tree.

        max_depth : int
            Maximum depth of each regression tree.

        min_samples_split : int
            Minimum number of samples needed to split a node.

        trees_ : list
            List of fitted regression trees.

        initial_prediction_ : float
            Initial constant prediction of the ensemble.

        n_features_in_ : int or None
            Number of features seen during fit.

        feature_importances_ : np.ndarray or None
            Global feature importances obtained by aggregating the importances
            of all trees in the ensemble.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        """
        Initialize the Gradient Boosting regressor.

        Parameters:
            n_estimators : int, default=100
                Number of trees (boosting iterations).

            learning_rate : float, default=0.1
                Shrinkage factor of each tree.

            max_depth : int, default=3
                Maximum depth of each regression tree.

            min_samples_split : int, default=2
                Minimum number of samples required to split a node.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.trees_ = []
        self.initial_prediction_ = 0.0
        self.n_features_in_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        """
        Train the Gradient Boosting model.

        Parameters:
            X : array-like of shape (n_samples, n_features)
                Training input samples.

            y : array-like of shape (n_samples,)
                Continuous target values.

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.n_features_in_ = X.shape[1]

        # Paso 0 — predicción base (media global)
        self.initial_prediction_ = float(np.mean(y))
        y_pred = np.full(len(y), self.initial_prediction_, dtype=float)

        self.trees_ = []

        for _ in range(self.n_estimators):
            # Residuos del MSE
            residuals = y - y_pred

            # Árbol entrenado sobre los residuos
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                criterion='squared_error',
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, residuals)

            # Actualización de la predicción
            y_pred += self.learning_rate * tree.predict(X)

            self.trees_.append(tree)

        self.feature_importances_ = self._compute_feature_importances()

        return self

    def predict(self, X):
        """
        Predict continuous values for the samples in X.

        Parameters:
            X : array-like of shape (n_samples, n_features)

        Returns:
            np.ndarray of shape (n_samples,)
        """
        X = np.asarray(X, dtype=float)

        y_pred = np.full(X.shape[0], self.initial_prediction_, dtype=float)

        for tree in self.trees_:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred

    def _compute_feature_importances(self):
        """
        Compute global feature importances for the ensemble.

        The final importance is obtained by aggregating the importances
        of all trees and normalizing the result so that the values sum to 1.
        """
        importances = np.zeros(self.n_features_in_, dtype=float)

        for tree in self.trees_:
            importances += self.learning_rate * tree.feature_importances_

        total = importances.sum()
        if total > 0:
            importances /= total

        return importances