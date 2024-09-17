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
"MONACO",
        "SMETANA",
        "BEAMS",
        "multiMAGNA++",
        "IsoRankN"
    ],
    "F1": [0.157, 0.154, 0.153, 0.530, 0.157, 0, 0.039, 0.251],
    "c=2": [0.881, 0.889, 0.886, 0.948, 0.889, 0, 0.128, 0.714],
"c=3": [0.823, 0.821, 0.817, 0.881, 0.854,0, 0.044, 0.677],
"c=4": [0.775, 0.765, 0.774, 0.819, 0.768, 0, 0.052, 0.634],
"c=5": [0.716, 0.726, 0.722, 0.765, 0.720, 0, 0.0028, 0.601],
"c=6": [0.585, 0.605, 0.604, 0.744, 0.628, 0, 0.002, 0.571],
"c=7": [0.519, 0.496, 0.515, 0.687, 0.502, 0, 0.005, 0.422],
"c=8": [0.473, 0.467, 0.457, 0.589, 0.532, 0, 0, 0.453],
    "CIQ": [0.071, 0.061, 0.065, 0.010, 0.018, 0, 0.0007, 0.011],
    "ICQ": [0.177, 0.177, 0.178, 0.157, 0.092, 0, 0.0001, 0.093],
    "Sensitivity": [0.448, 0.438, 0.442, 0.328, 0.337,0, 0.259, 0.197]
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

metrics = ["NCV", "$GS^3$", "$NCV-GS^3$", "GC", "c=2", "CIQ", "ICQ", "Sensitivity"]

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
