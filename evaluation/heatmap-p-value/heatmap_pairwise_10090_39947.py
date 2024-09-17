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
"F1": [0, 0, 0, 0.036, 0, 0.935, 0.033, 0.054, 0],
    "NCV": [0.167, 0.128, 0.127, 0.155, 0.167, 1.0, 0.151, 0.062, 0.133],
    "$GS^3$": [0, 0, 0, 0.02, 0.01, 0, 0.07, 0, 0.03],
    "$NCV-GS^3$": [0, 0, 0, 0.049, 0.038, 0, 0.102, 0, 0.066],
    "GC": [0.949, 0.913, 0.943, 0.936, 0.937, 0.978, 0.408, 0.407, 0.681],
    "c=2": [1.0, 1.0, 1.0, 0, 0, 1.0, 0.25, 0.266, 0.666],
    "CIQ": [0, 0, 0, 0, 0, 0.023, 0, 0, 0],
    "ICQ": [0.250, 0.306, 0.304, 0.128, 0.121, 0.062, 0, 0, 0.063],
    "Sensitivity": [0.011, 0.011, 0.010, 0.011, 0.015, 0.036, 0.006, 0.003, 0.0075]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (10090-39947)")
plt.savefig("Figure-10090-39947.eps")
