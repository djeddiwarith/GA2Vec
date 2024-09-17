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
"F1": [0.247, 0.249, 0.248, 0.349, 0.249, 0.345, 0.056, 0.056, 0.287],
    "NCV": [0.622, 0.638, 0.620, 0.655, 0.437, 0.625, 0.783, 0.645, 0.545],
    "$GS^3$": [0, 0, 0, 0.01, 0.01, 0, 0, 0, 0],
    "$NCV-GS^3$": [0.012, 0.014, 0.014, 0.094, 0.073, 0.067, 0.056, 0, 0.028],
    "GC": [0.585, 0.630, 0.583, 0.906, 0.867, 0.917, 0.273, 0.286, 0.700],
    "c=2": [0.907, 0.903, 0.913, 0.879, 0.866, 0.886, 0.101, 0.123, 0.693],
    "CIQ": [0.057, 0.049, 0.042, 0.020, 0.011, 0.012, 0.001, 0.0002, 0.029],
    "ICQ": [0.569, 0.565, 0.563, 0.601, 0.591, 0.346, 0.0002, 0.0003, 0.464],
    "Sensitivity": [0.072, 0.071, 0.072, 0.058, 0.058, 0.078, 0.062, 0.055, 0.056]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (7227-4932)")
plt.savefig("Figure-7227-4932.eps")
