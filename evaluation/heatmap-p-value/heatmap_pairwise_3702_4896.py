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
        "BEAMS",
"multiMAGNA++",
        "HubAlign",
"IsoRankN"
    ],
    "F1": [0.053, 0.057, 0.058, 0.084, 0.082, 0.033,   0.016, 0.012, 0.046],
    "NCV": [0.551, 0.543, 0.550, 0.394, 0.385, 0.920,   0.429, 0.258, 0.231],
    "$GS^3$": [0, 0, 0, 0.01, 0, 0,    0.03, 0, 0.01],
    "$NCV-GS^3$": [0.015, 0.016, 0.016, 0.06, 0.037, 0.024,    0.108, 0, 0.050],
    "GC": [0.801, 0.837, 0.798, 0.927, 0.905, 0.940,     0.25, 0.115, 0.735],
    "c=2": [0.782, 0.772, 0.739, 0.827, 0.821, 0.846,    0.038, 0, 0.333],
    "CIQ": [0.057, 0.055, 0.064, 0.035, 0.040, 0.055,      0.009, 0, 0.005],
    "ICQ": [0.450, 0.452, 0.453, 0.485, 0.497, 0.176,     0.001, 0, 0.256],
    "Sensitivity": [0.006, 0.006, 0.006, 0.003, 0.003, 0.008,    0.0009, 0.0005, 0.004]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (3702-4896)")
plt.savefig("Figure-3702-4896.eps")
