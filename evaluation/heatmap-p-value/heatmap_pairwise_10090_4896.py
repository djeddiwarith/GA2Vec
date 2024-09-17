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
"IsoRankN"
    ],
"F1": [0.266, 0.259, 0.280, 0.268, 0.326, 0.334, 0.909, 0.739, 0.219, 0.240],
    "NCV": [0.590,0.507, 0.515, 0.508, 0.367, 0.371, 0.980, 0.119, 0.801, 0.350],
    "$GS^3$": [0,0, 0, 0, 0.02, 0.01, 0, 0, 0, 0.02],
    "$NCV-GS^3$": [0.010,0.014, 0.013, 0.011, 0.090, 0.052, 0.103, 0, 0, 0.078],
    "GC": [0.893,0.795, 0.874, 0.815, 0.918, 0.911, 0.958, 0.307, 0.487, 0.727],
    "c=2": [0.860,0.873, 0.845, 0.850, 0.75, 0.758, 0.937, 0, 0.138, 0.755],
    "CIQ": [0,0, 0, 0, 0, 0, 0.006, 0, 0, 0.009],
    "ICQ": [0.312,0.314, 0.331, 0.317, 0.135, 0.141, 0.065, 0, 0.0001, 0.056],
    "Sensitivity": [0.046,0.035, 0.041, 0.036, 0.034, 0.035, 0.045, 0, 0.068, 0.031]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="Greys", cbar=True,  annot_kws={"size": 14, "weight": "bold"}, linewidths=.5, linecolor='white')
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (10090-4896)")
plt.savefig("Figure-10090-4896.eps", dpi=300, bbox_inches='tight', pad_inches=0.1)
