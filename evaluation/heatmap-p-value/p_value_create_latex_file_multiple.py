import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from itertools import combinations
import numpy as np



data_align = {
    "Approaches": [
        "GA2Vec-ANCPRBT",
"GA2Vec-ANCESM",
"GA2Vec-ANCT5",
"GA2Vec-LP",
"GA2Vec-LT",
"GA2Vec-WT",
"GA2Vec-GM",
"GA2Vec-GM&LT",
"GA2Vec-GM&WT",
"GA2Vec-GM&LP",
"GA2Vec-LP&LT",
"GA2Vec-LP&WT",
"GA2Vec-LT&WT",
"GA2Vec-GM&LT&LP",
"GA2Vec-GM&LP&WT",
"GA2Vec-LT&LP&WT",
"MONACO",
        "SMETANA",
        "multiMAGNA++"
    ],
"F1": [0.132, 0.135, 0.133, 0.134, 0.161, 0.161, 0.164, 0.161, 0.145, 0.141, 0.137,0.130,0.151,   0.162, 0.141, 0.137,0.520, 0.132, 0.034],
    "c=2": [0.888, 0.885, 0.885, 0.883, 0.899, 0.875, 0.887, 0.907, 0.890, 0.890, 0.885,0.885, 0.897   , 0.904, 0.885,0.888, 0.950, 0.869, 0.129],
"c=3": [0.845, 0.839, 0.844, 0.842, 0.830, 0.852, 0.867, 0.854, 0.853, 0.848, 0.840,0.847,0.838  , 0.848, 0.847,0.836, 0.888, 0.860, 0.066],
"c=4": [0.766, 0.769, 0.774, 0.765, 0.802, 0.773, 0.803, 0.813, 0.785, 0.775, 0.787,0.776,0.786  ,0.796, 0.789,0.781, 0.834, 0.782, 0.046],
"c=5": [0.724, 0.713, 0.702, 0.748, 0.726, 0.728, 0.757, 0.717, 0.709, 0.748, 0.745,0.736,0.748   , 0.753, 0.737,0.715, 0.769, 0.735, 0.031],
"c=6": [0.649, 0.658, 0.624, 0.647, 0.645, 0.688, 0.638, 0.673, 0.673, 0.623,0.652,0.656, 0.689    , 0.701, 0.641,0.671, 0.740, 0.643, 0.004],
"c=7": [0.547, 0.559, 0.553, 0.579, 0.6, 0.585, 0.660, 0.625, 0.649, 0.581,0.614,0.619,0.572   , 0.676, 0.586, 0.561,0.678, 0.589, 0.0049],
"c=8": [0.433, 0.440, 0.481, 0.428, 0.472, 0.509, 0.523, 0.428, 0.433, 0.436, 0.405,0.5,0.480   ,0.475, 0.494, 0.43,0.595, 0.481, 0],
"c=9": [0.402, 0.395, 0.433, 0.466, 0.375, 0.523, 0.379, 0.382, 0.491, 0.421, 0.389,0.410,0.415    , 0.378, 0.462,0.470, 0.625, 0.514, 0],
"c=10": [0.375, 0.357, 0.383, 0.428, 0.333, 0.578, 0.266, 0.470, 0.388, 0.325, 0.333,0.382,0.484   , 0.4, 0.485, 0.521, 0.532, 0.402, 0],
    "CIQ": [0, 0.062, 0.074, 0.022, 0.048, 0.032, 0.028, 0.053, 0.041, 0.040, 0.054,0.039, 0.052    , 0.110, 0.043,0.063, 0.011, 0.021, 0.00014],
    "ICQ": [0.168, 0.165, 0.165, 0.178, 0.185, 0.180, 0.186, 0.182, 0.176, 0.173, 0.174,0.170, 0.177   , 0.182, 0.175, 0.169, 0.151, 0.088, 0.00006],
    "Sensitivity": [0.448, 0.448, 0.448, 0.278, 0.244, 0.284, 0.229, 0.369, 0.352, 0.374, 0.392,0.373,0.381  , 0.440, 0.419, 0.424, 0.328, 0.341, 0.263]
}


# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

# List of comparisons provided by the user
comparisons = [
    "GA2Vec-ANCPRBT vs GA2Vec-ANCESM", "GA2Vec-ANCPRBT vs GA2Vec-ANCT5", "GA2Vec-ANCPRBT vs MONACO",
    "GA2Vec-ANCPRBT vs SMETANA", "GA2Vec-ANCPRBT vs BEAMS", "GA2Vec-ANCPRBT vs multiMAGNA++",
    "GA2Vec-ANCPRBT vs HubAlign", "GA2Vec-ANCPRBT vs IsoRankN", "GA2Vec-ANCESM vs GA2Vec-ANCT5",
    "GA2Vec-ANCESM vs MONACO", "GA2Vec-ANCESM vs SMETANA", "GA2Vec-ANCESM vs BEAMS",
    "GA2Vec-ANCESM vs multiMAGNA++", "GA2Vec-ANCESM vs HubAlign", "GA2Vec-ANCESM vs IsoRankN",
    "GA2Vec-ANCT5 vs MONACO", "GA2Vec-ANCT5 vs SMETANA", "GA2Vec-ANCT5 vs BEAMS",
    "GA2Vec-ANCT5 vs multiMAGNA++", "GA2Vec-ANCT5 vs HubAlign", "GA2Vec-ANCT5 vs IsoRankN",
    "MONACO vs SMETANA", "MONACO vs BEAMS", "MONACO vs multiMAGNA++",
    "MONACO vs HubAlign", "MONACO vs IsoRankN", "SMETANA vs BEAMS",
    "SMETANA vs multiMAGNA++", "SMETANA vs HubAlign", "SMETANA vs IsoRankN",
    "BEAMS vs multiMAGNA++", "BEAMS vs HubAlign", "BEAMS vs IsoRankN",
    "multiMAGNA++ vs HubAlign", "multiMAGNA++ vs IsoRankN", "HubAlign vs IsoRankN"
]

# Compute pairwise Wilcoxon tests for all method pairs
method_pairs = list(combinations(df.index, 2))
p_values = {}

metrics = ["c=2", "c=3", "c=4", "c=5","c=6","c=7","c=8", "c=9","c=10", "CIQ", "ICQ", "Sensitivity"]

for method1, method2 in method_pairs:
    data1 = df.loc[method1, metrics].values
    data2 = df.loc[method2, metrics].values

    # Check if data1 and data2 are identical
    if np.all(data1 == data2):
        p_value = np.nan  # Assign NaN for identical data
    else:
        try:
            stat, p_value = wilcoxon(data1, data2)
        except ValueError:
            p_value = np.nan  # Assign NaN if Wilcoxon test fails

    p_values[(method1, method2)] = p_value

# Print p-values for the provided comparisons
for comparison in comparisons:
    method1, method2 = comparison.split(' vs ')
    if (method1, method2) in p_values:
        print(f"{comparison}: {p_values[(method1, method2)]:.4f}")
    elif (method2, method1) in p_values:
        print(f"{comparison}: {p_values[(method2, method1)]:.4f}")
    else:
        print(f"{comparison}: NaN")
