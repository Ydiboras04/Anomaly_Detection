**Anomaly Detection: an Isolation Forest built from scratch.** This project implements the Isolation Forest algorithm (Liu, Ting & Zhou, 2008) in pure Python and NumPy, without scikit-learn. It detects anomalies in numerical data.

**How it works.** Anomalies are rare and far from the rest of the data, so random splits isolate them quickly. The forest builds many random trees. The fewer splits it takes to isolate a point, the more likely that point is an anomaly.

**What's in [isolation_forest.py](isolation_forest.py):**
- **`c_factor(n)`**: the normalisation factor. It's the average path length of an unsuccessful search in a binary search tree of `n` points, using the Euler–Mascheroni constant.
- **`anomaly_score()`**: the standard score `s(x, n) = 2^(-E(h(x)) / c(n))`. A score close to 1 means an anomaly; a score well below 0.5 means a normal point.
- **`Node`**: one node of a tree, either an internal split or a leaf.
- **`IsolationTree`**: builds one tree recursively. At each node it picks a random feature and a random split value between that feature's min and max. `path_length()` measures how deep a point goes before reaching a leaf.
- **`IsolationForest`**: the main model, with a scikit-learn-style API:
  - `fit(X)` trains `n_estimators` trees, each on a random subsample of `max_samples` points, with depth capped at `log2(sample_size)`. It then sets the decision threshold from the `contamination` rate.
  - `decision_function(X)` returns an anomaly score for each point.
  - `predict(X)` returns `-1` for an anomaly and `1` for a normal point.
- **Built-in demo:** 200 normal points clustered around (0, 0), plus 10 outliers spread across [-4, 4]. It trains the forest with 5% contamination and prints the score and label for each outlier.

**Stack:** Python 3.14 and NumPy 2.5.3, in a local `.venv`.

**A bug to fix:** `decision_function` normalises with `min(X.shape[0], self.max_samples)`, the size of the data being *scored*, not the subsample size used in training. The demo scores only 10 outliers, so it normalises with `c(10)` instead of `c(210)`. Those scores are then on a different scale from the threshold, which was computed on all 210 training points. The fix is to save the training `sample_size` in `fit()` (for example as `self.sample_size_`) and reuse it in `decision_function`.

I can also put this description into your readme.txt, in English or in French to match the code comments.
