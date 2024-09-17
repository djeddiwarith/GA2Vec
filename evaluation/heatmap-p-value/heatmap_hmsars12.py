import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data_align = {
    "Approaches": [
"GA2Vec-ANCPRBT-f2",
        "GA2Vec-ANCPRBT",
"GA2Vec-ANCESM",
"GA2Vec-ANCT5",
"MONACO",
        "SMETANA",
        "BEAMS",
        "multiMAGNA++",
        "HubAlign",
    ],
    "NCV": [0.664, 0.578, 0.590, 0.608, 1.0, 1.0, 0.477,0.693, 0.462],
    "$GS^3$": [0.05, 0.02, 0.02, 0.04, 0.24, 0.20, 0.07, 0.05, 0],
    "$NCV-GS^3$": [0.178, 0.096, 0.114, 0.164, 0.604, 0.556, 0.182, 0.190, 0],
    "GC": [0.886,  0.941, 0.920, 0.887, 0.994, 0.999, 0.963, 0.768, 0.780],
    "c=2": [0.844,  0.847, 0.838, 0.844, 0.896, 0.952, 0.796, 0.202, 0.209],
    "CIQ": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    "ICQ": [0.029,  0.033, 0.033, 0.032, 0.004, 0.003, 0.025, 0.0009, 0.001],
    "Sensitivity": [0.057, 0.054, 0.054, 0.053, 0.014, 0.004, 0.050, 0.058, 0.040]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

# Plot heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="Greys", cbar=True,  annot_kws={"size": 14, "weight": "bold"}, linewidths=.5, linecolor='white')
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (hmsars1-hmsars2)")
plt.savefig("hmsars12.eps", dpi=300, bbox_inches='tight', pad_inches=0.1)
