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
"F1": [0, 0, 0, 0.440, 0, 0.561, 0, 0, 0.404],
    "NCV": [0.724, 0.744, 0.732, 0.462, 0.520, 0.703, 0.735, 0.594, 0.406],
    "$GS^3$": [0, 0, 0, 0.01, 0.02, 0, 0.01, 0, 0.02],
    "$NCV-GS^3$": [0.018, 0.011, 0.019, 0.121, 0.102, 0.074, 0.063, 0, 0.137],
    "GC": [0.494, 0.459, 0.514, 0.887, 0.844, 0.907, 0.162, 0.187, 0.656],
    "c=2": [0.909, 0.904, 0.903, 0.892, 0.879, 0.885, 0.101, 0.139, 0.776],
    "CIQ": [0.043, 0.023, 0.034, 0.0025, 0.019, 0.003, 0.0012, 0, 0.004],
    "ICQ": [0.482, 0.480, 0.475, 0.496, 0.475, 0.322, 0.0002, 0.0002, 0.447],
    "Sensitivity": [0.060, 0.062, 0.060, 0.046, 0.047, 0.051, 0.062, 0.030, 0.041]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (7227-6239)")
plt.savefig("Figure-7227-6239.eps")
