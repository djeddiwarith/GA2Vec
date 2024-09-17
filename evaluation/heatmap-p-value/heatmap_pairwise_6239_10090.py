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
"F1": [0, 0, 0, 0.726, 0, 0.830,0, 0, 0.704],
    "NCV": [0.691, 0.688, 0.680, 0.695, 0.495, 0.637, 0.609, 0.404, 0.457],
    "$GS^3$": [0, 0, 0, 0, 0.01, 0, 0.01, 0, 0.01],
    "$NCV-GS^3$": [0.020, 0.021, 0.021, 0.011, 0.083, 0, 0.85, 0, 0.124],
    "GC": [0.688, 0.706, 0.698, 0.626, 0.845, 0.953, 0.348, 0.432, 0.789],
    "c=2": [0.849, 0.840, 0.848, 0.970, 0.790, 1.0, 0.188, 0.229, 0.896],
    "CIQ": [0.097, 0.060, 0.056, 0.010, 0.039, 0.002, 0.003, 0, 0.009],
    "ICQ": [0.327, 0.317, 0.323, 0.168, 0.120, 0.004, 0.0001, 0.0009, 0.129],
    "Sensitivity": [0.064, 0.064, 0.062, 0.049, 0.056, 0.007, 0.054, 0.039, 0.045]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (6239-10090)")
plt.savefig("Figure-6239-10090.eps")
