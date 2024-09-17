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
"F1": [0.455, 0.46, 0.471, 0.528, 0.518, 0.509, 0.352, 0.424, 0.471],
    "NCV": [0.187, 0.176, 0.190, 0.147, 0.149, 0.710, 0.253, 0.115, 0.187],
    "$GS^3$": [0, 0, 0, 0.01, 0, 0, 0.01, 0, 0],
    "$NCV-GS^3$": [0.012, 0.011, 0.013, 0.040, 0.027, 0.011, 0.060, 0, 0.019],
    "GC": [0.854, 0.802, 0.833, 0.872, 0.880, 0.950, 0.502, 0.625, 0.718],
    "c=2": [0.842, 0.856, 0.858, 0.793, 0.795, 0.834, 0.094, 0.080, 0.515],
    "CIQ": [0.013, 0.012, 0.015, 0.016, 0.011, 0.021, 0.003, 0.001, 0.024],
    "ICQ": [0.508, 0.508, 0.503, 0.528, 0.535, 0.075, 0, 0, 0.289],
    "Sensitivity": [0.081, 0.0812, 0.083, 0.075, 0.075, 0.201, 0.080, 0.037, 0.075]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (9606-562)")
plt.savefig("Figure-9606-562.eps")
