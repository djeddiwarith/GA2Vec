import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data_align = {
    "Approaches": [
        "GA2Vec-ANCPRBT",
"GA2Vec-ANCESM",
"GA2Vec-ANCT5",
"MONACO",
        "SMETANA",
        "multiMAGNA++",
        "IsoRankN"
    ],
    "F1": [0.157, 0.154, 0.153, 0.530, 0.157,  0.039, 0.251],
    "c=2": [0.881, 0.889, 0.886, 0.948, 0.889, 0.128, 0.714],
"c=3": [0.823, 0.821, 0.817, 0.881, 0.854,0.044, 0.677],
"c=4": [0.775, 0.765, 0.774, 0.819, 0.768,  0.052, 0.634],
"c=5": [0.716, 0.726, 0.722, 0.765, 0.720, 0.0028, 0.601],
"c=6": [0.585, 0.605, 0.604, 0.744, 0.628, 0.002, 0.571],
"c=7": [0.519, 0.496, 0.515, 0.687, 0.502,  0.005, 0.422],
"c=8": [0.473, 0.467, 0.457, 0.589, 0.532,  0, 0.453],
    "CIQ": [0.071, 0.061, 0.065, 0.010, 0.018,  0.0007, 0.011],
    "ICQ": [0.177, 0.177, 0.178, 0.157, 0.092,  0.0001, 0.093],
    "Sensitivity": [0.448, 0.438, 0.442, 0.328, 0.337, 0.259, 0.197]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

# Plot heatmap
plt.figure(figsize=(16, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (7227-6239-4932-9606-10090-562-4896-10116)")
plt.savefig("Figure8species.eps")
