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
"F1": [0.201, 0.207, 0.208, 0.594, 0.197, 0.690, 0.114, 0.100, 0.564],
    "NCV": [0.644, 0.610, 0.616, 0.843, 0.441, 1.0, 0.406, 0.274, 0.648],
    "$GS^3$": [0, 0, 0, 0.01, 0.01, 0, 0.01, 0, 0.01],
    "$NCV-GS^3$": [0.015, 0.017, 0.011, 0.126, 0.075, 0.064, 0.072, 0, 0.108],
    "GC": [0.590, 0.668, 0.637, 0.939, 0.871, 0.932, 0.313, 0.358, 0.690],
    "c=2": [0.807, 0.817, 0.808, 0.908, 0.862, 0.947, 0.210, 0.253, 0.870],
    "CIQ": [0.029, 0.049, 0.039, 0.0015, 0.006, 0.0013, 0.002, 0, 0.001],
    "ICQ": [0.289, 0.298, 0.294, 0.153, 0.116, 0.146, 0.0002, 0.0001, 0.138],
    "Sensitivity": [0.093, 0.097, 0.086, 0.072, 0.082, 0.072, 0.067, 0.050, 0.060]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (7227-10090)")
plt.savefig("Figure-7227-10090.eps")
