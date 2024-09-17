import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data_align = {
    "Approaches": [
"GA2Vec-ANCPRBT-f2",
"GA2Vec-ANCESM-f2",
"GA2Vec-ANCT5-f2",
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
    "NCV": [0.888, 0.888, 0.888, 0.666, 0.666, 0.666, 0.814, 0.814, 0.814, 0.888, 0.888, 0.074],
    "$GS^3$": [0.31, 0.33, 0.15, 0.25, 0.12, 0.27, 0.32, 0.32, 0.28, 0.58, 0.41, 0],
    "$NCV-GS^3$": [0.522, 0.544, 0.369, 0.408, 0.288, 0.426, 0.512, 0.512, 0.474, 0.717, 0.606, 0],
    "GC": [0.833, 0.833, 0.833, 0.75, 0.9, 0.75, 0.8, 0.8, 0.833, 0.6, 0.6, 1.0],
    "c=2": [0.875, 0.857, 0.875, 0.857, 0.857, 0.857, 0.875, 0.875, 0.777, 0.875, 0.270, 0],
    "CIQ": [0.333, 0.333, 0.333, 0.307, 0.352, 0.5, 0.333, 0.333, 0.333, 0.476, 0.315, 0],
    "ICQ": [0.764, 0.764, 0.764, 0.830, 0.750, 0.830, 0.764, 0.764, 0.800, 0.075, 0.250, 0],
    "Sensitivity": [0.003, 0.003, 0.003, 0.002, 0.003, 0.003, 0.003, 0.003, 0.003, 0.0028, 0.002, 0.0007]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="Greys", cbar=True,  annot_kws={"size": 14, "weight": "bold"}, linewidths=.5, linecolor='white')
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (SARS1-SARS2)")
plt.savefig("Figure-sars12.eps", dpi=300, bbox_inches='tight', pad_inches=0.1)
