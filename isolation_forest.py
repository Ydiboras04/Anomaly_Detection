import numpy as np

def c_factor(n: int) -> float:
    '''
    Facteur de normalisation
    '''

    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0

    euler_mascheroni = 0.5772156649

    return 2.0 * (np.log(n-1) + euler_mascheroni) - (2.0 * (n - 1) / n)

def anomaly_score(avg_paths: np.ndarray, n_samples: int) -> np.ndarray:
    '''
    Calcule le score d'anomalie
    '''
    c_n = c_factor(n_samples)

    if c_n == 0:
        return np.zeros_like(avg_paths, dtype=float)

    return 2.0 ** (- (avg_paths / c_n))

class Node:
    def __init__(self, left=None, right=None, feature: int = None, split_value: float = None, size: int = 0, is_leaf: bool = False):
        self.left = left
        self.right = right
        self.feature = feature
        self.split_value = split_value
        self.size = size
        self.is_leaf = is_leaf

class IsolationTree:
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: np.ndarray, current_depth: int = 0) -> Node:
        '''
        Construit l'arbre de manière récursive
        '''
        n_samples, n_features = X.shape

        #condition d'arret
        if current_depth >= self.max_depth or n_samples <= 1:
            return Node(size=n_samples, is_leaf=True)

        #choix aléatoire d'une feature
        feature_idx = np.random.randint(0, n_features)
        feat_values = X[:, feature_idx]

        min_val = feat_values.min()
        max_val = feat_values.max()

        if min_val == max_val:
            return Node(size=n_samples, is_leaf=True)

        #choix aléatoire de la valeur split
        split_value = np.random.uniform(min_val, max_val)

        #séparation des données
        left_mask = feat_values < split_value
        right_mask = ~left_mask

        #construction sous arbres
        left_child = self.fit(X[left_mask], current_depth + 1)
        right_child = self.fit(X[right_mask], current_depth + 1)

        return Node(
            left=left_child,
            right=right_child,
            feature=feature_idx,
            split_value=split_value,
            size=n_samples,
            is_leaf=False,
        )
        
    def path_length(self, x: np.ndarray, node: Node = None, current_depth: int = 0) -> float:
        '''
        Calcule la profondeur h(X)
        '''

        if node is None:
            node = self.root

        if node.is_leaf:
            return current_depth + c_factor(node.size)

        if x[node.feature] < node.split_value:
            return self.path_length(x, node.left, current_depth + 1)
        else:
            return self.path_length(x, node.right, current_depth + 1)

class IsolationForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.1,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.trees = []
        self.threshold_ = 0.5  # Seuil calculé après entraînement

    def fit(self, X: np.ndarray):
        """Entraîne la forêt d'isolement sur le jeu de données X."""
        n_samples = X.shape[0]

        # Détermination de la taille réelle du sous-échantillon pour un arbre
        sample_size = min(n_samples, self.max_samples)

        # Profondeur maximale théorique pour chaque arbre : log2(sample_size)
        max_depth = int(np.ceil(np.log2(max(sample_size, 2))))

        self.trees = []

        # Construction de n_estimators arbres
        for _ in range(self.n_estimators):
            # Tirage aléatoire sans remise de 'sample_size' indices
            indices = np.random.choice(n_samples, size=sample_size, replace=False)
            X_sub = X[indices]

            # Instanciation et entraînement d'un arbre
            tree = IsolationTree(max_depth=max_depth)
            tree.root = tree.fit(X_sub)
            self.trees.append(tree)

        # Calcul du seuil de décision basé sur le paramètre contamination
        scores = self.decision_function(X)
        # On définit le seuil au percentile correspondant aux données les plus anormales
        self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))

        return self

    def compute_paths(self, X: np.ndarray) -> np.ndarray:
        """Calcule la profondeur moyenne E(h(x)) pour chaque point x à travers tous les
        arbres."""
        n_samples = X.shape[0]
        paths = np.zeros(n_samples)

        # Somme des longueurs de chemin dans chaque arbre
        for tree in self.trees:
            for i in range(n_samples):
                paths[i] += tree.path_length(X[i])

        # Moyenne sur l'ensemble de la forêt
        return paths / self.n_estimators

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Retourne le score d'anomalie pour chaque point de X (entre 0 et 1)."""
        avg_paths = self.compute_paths(X)
        sample_size = min(X.shape[0], self.max_samples)

        # Application de la formule finale s(x, n)
        return anomaly_score(avg_paths, sample_size)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        # -1 si le score dépasse le seuil (anomalie), 1 sinon (normal)
        return np.where(scores >= self.threshold_, -1, 1)
        
#test
if __name__ == "__main__":
    np.random.seed(42)

    # 1. Génération de 200 points "normaux" regroupés autour de (0, 0)
    X_normal = np.random.randn(200, 2) * 0.5

    # 2. Génération de 10 points "anormaux" très éloignés du groupe
    X_outliers = np.random.uniform(low=-4, high=4, size=(10, 2))

    # Fusion des données
    X_train = np.vstack([X_normal, X_outliers])

    # 3. Entraînement de notre IsolationForest from scratch
    clf = IsolationForest(n_estimators=100, max_samples=256, contamination=0.05)
    clf.fit(X_train)

    # 4. Évaluation des 10 anomalies générées
    outlier_scores = clf.decision_function(X_outliers)
    outlier_predictions = clf.predict(X_outliers)

    print("=== TEST SUR LES POINTS ANORMAUX ===")
    print(f"Seuil calculé (contamination 5%) : {clf.threshold_:.4f}\n")
    for i, (score, pred) in enumerate(
        zip(outlier_scores, outlier_predictions)
    ):
        label = "Anomalie (-1)" if pred == -1 else "Normal (1)"
        print(
            f"Point {i+1:2d} | Position: {X_outliers[i]} | Score: {score:.4f} | Détection: {label}"
        )