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
    "F1": [0, 0, 0, 0, 0, 0.686, 0, 0, 0],
    "NCV": [0.174, 0.178, 0.180, 0.162, 0.150, 0.630, 0.625, 0.454, 0.322],
    "$GS^3$": [0, 0, 0, 0.01, 0, 0, 0.01, 0, 0.01],
    "$NCV-GS^3$": [0.015, 0.014, 0.018, 0.031, 0.019, 0.019, 0.094, 0, 0.052],
    "GC": [0.607, 0.576, 0.568, 0.692, 0.718, 0.824, 0.181, 0.217, 0.470],
    "c=2": [0.855, 0.857, 0.817, 0.822, 0.8, 0.824, 0.077, 0.066, 0.388],
    "CIQ": [0, 0, 0, 0, 0, 0.007, 0.001, 0.001, 0.023],
    "ICQ": [0.555, 0.552, 0.545, 0.583, 0.599, 0.059, 0, 0, 0.201],
    "Sensitivity": [0.023, 0.025, 0.027, 0.026, 0.024, 0.045, 0.043, 0.034, 0.038]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (6239-562)")
plt.savefig("Figure-6239-562.eps")
