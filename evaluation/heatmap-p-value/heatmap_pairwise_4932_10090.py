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
    "F1": [0.381, 0.371, 0.367, 0.585, 0.428, 0.799, 0.251, 0.285, 0.430],
    "NCV": [0.436, 0.433, 0.424, 0.501, 0.298, 1.0, 0.567, 0.390, 0.637],
    "$GS^3$": [0, 0, 0, 0.01, 0.02, 0, 0.01, 0, 0.01],
    "$NCV-GS^3$": [0.012, 0.031, 0.017, 0.089, 0.078, 0.100, 0.083, 0, 0.086],
    "GC": [0.865, 0.812, 0.826, 0.960, 0.938, 0.960, 0.477, 0.531, 0.772],
    "c=2": [0.862, 0.856, 0.843, 0.909, 0.909, 1.0, 0.162, 0.151, 0.740],
    "CIQ": [0.073, 0.103, 0.069, 0.005, 0.026, 0.013, 0.008, 0.032, 0.011],
    "ICQ": [0.351, 0.338, 0.346, 0.195, 0.138, 0.127, 0.000007, 0.0002, 0.123],
    "Sensitivity": [0.056, 0.058, 0.054, 0.049, 0.049, 0.072, 0.083, 0.0934, 0.043]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (4932-10090)")
plt.savefig("Figure-4932-10090.eps")
