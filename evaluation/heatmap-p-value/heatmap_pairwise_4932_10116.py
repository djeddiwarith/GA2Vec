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
"F1": [0.190, 0.187, 0.184, 0.222, 0.222, 0.159, 0.108, 0.083, 0.281],
    "NCV": [0.073, 0.073, 0.077, 0.070, 0.070, 0.580, 0.050, 0.033, 0.019],
    "$GS^3$": [0, 0, 0, 0.02, 0.01, 0, 0.33, 0, 0],
    "$NCV-GS^3$": [0.009, 0.009, 0.007, 0.041, 0.023, 0.018, 0.129, 0, 0],
    "GC": [0.875, 0.882, 0.918, 0.894, 0.903, 0.940, 0.436, 0.408, 0.903],
    "c=2": [0.818, 0.809, 0.818, 0.838, 0.812, 0.874, 0.141, 0.046, 0.714],
    "CIQ": [0.083, 0.162, 0.187, 0.194, 0.214, 0.051, 0.4, 0, 0.333],
    "ICQ": [0.485, 0.466, 0.479, 0.509, 0.527, 0.064, 0, 0, 0.503],
    "Sensitivity": [0.009, 0.008, 0.009, 0.009, 0.009, 0.043, 0.010, 0.003, 0.043]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (4932-10116)")
plt.savefig("Figure-4932-10116.eps")
