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
"F1": [0, 0, 0, 0, 0, 0.715, 0, 0, 0],
    "NCV": [0.196, 0.192, 0.147, 0.123, 0.100, 0.645, 0.055, 0, 0.037],
    "$GS^3$": [0, 0, 0, 0, 0, 0, 0.23, 0, 0],
    "$NCV-GS^3$": [0.002, 0.002, 0, 0.020, 0.015, 0, 0.111, 0, 0],
    "GC": [0.591, 0.601, 0.676, 0.757, 0.768, 0.872, 0.257, 0, 0.896],
    "c=2": [0.875, 0.8, 0.8, 0.777, 0.833, 0.851, 0, 0, 0],
    "CIQ": [0.076, 0.076, 0, 0, 0, 0.018, 0, 0,0],
    "ICQ": [0.006, 0.005, 0.006, 0.005, 0.005, 0.037, 0, 0, 0],
    "Sensitivity": [0.005, 0.005, 0.003, 0.001, 0.0019, 0.011, 0, 0, 0]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (6239-10116)")
plt.savefig("Figure-6239-10116.eps")
