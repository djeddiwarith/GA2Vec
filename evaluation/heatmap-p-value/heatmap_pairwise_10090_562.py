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
"F1": [0.448, 0.442, 0.465, 0.473, 0.514, 0.910, 0.421, 0.441, 0.537],
    "NCV": [0.119, 0.116, 0.121, 0.107, 0.096, 0.961, 0.980, 0.980, 0.064],
    "$GS^3$": [0, 0, 0, 0.01, 0.01, 0, 0.01, 0.17, 0.09],
    "$NCV-GS^3$": [0.015, 0, 0.011, 0.024, 0.030, 0.051, 0.113, 0.413, 0.076],
    "GC": [0.782, 0.805, 0.763, 0.818, 0.800, 0.872, 0.417, 0.440, 0.622],
    "c=2": [0.649, 0.642, 0.666, 0.875, 0.875, 1.0, 0.163, 0.139, 0.571],
    "CIQ": [0.041, 0.046, 0.060, 0.031, 0.037, 0.012, 0, 0.041, 0.011],
    "ICQ": [0.378, 0.384, 0.379, 0.157, 0.157, 0.059, 0, 0.018, 0.051],
    "Sensitivity": [0.020, 0.018, 0.020, 0.019, 0.019, 0.068, 0.009, 0.115, 0.011]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (10090-562)")
plt.savefig("Figure-10090-562.eps")
