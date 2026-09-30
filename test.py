from isolation_forest import c_factor, anomaly_score
import numpy as np

#test de facteur de normalisation et de score d'anomalie
subsample_size = 256
c_256 = c_factor(subsample_size)
print(f"Facteur de normalisation c(256) = {c_256:.4f}")

# Simulation de 3 points avec des profondeurs moyennes E(h(x)) différentes :
# 1. Un point très isolé (profondeur faible = 3.0)
# 2. Un point "moyen" (profondeur égale à c(n) ≈ 10.24)
# 3. Un point très enfoui (profondeur élevée = 18.0)
test_paths = np.array([3.0, c_256, 18.0])
scores = anomaly_score(test_paths, subsample_size)

for path, score in zip(test_paths, scores):
    print(f"Profondeur moyenne : {path:6.2f} -> Score d'anomalie : {score:.4f}")