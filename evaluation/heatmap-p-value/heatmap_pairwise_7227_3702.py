import numpy as np
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
"F1": [0.067, 0.065, 0.069, 0.063, 0.072, 0.234, 0.186, 0.008, 0.035],
    "NCV": [0.635, 0.632, 0.633, 0.490, 0.476, 0.616, 0.316, 0.675, 0.476],
    "$GS^3$": [0, 0, 0, 0.01, 0.01, 0, 0, 0, 0],
    "$NCV-GS^3$": [0.010, 0.010, 0.013, 0.070, 0.070, 0.035, 0, 0, 0.026],
    "GC": [0.455, 0.480, 0.438, 0.853, 0.826, 0.876, 0.161, 0.141, 0.469],
    "c=2": [0.773, 0.775, 0.736, 0.774, 0.793, 0.833, 0, 0.055, 0.589],
    "CIQ": [0.045, 0.040, 0.042, 0.004, 0.065, 0.037, 0, 0.0001, 0.010],
    "ICQ": [0.333, 0.339, 0.334, 0.334, 0.341, 0.131, 0, 0.0003, 0.193],
    "Sensitivity": [0.032, 0.032, 0.033, 0.019, 0.017, 0.033, 0, 0.002, 0.0060]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (7227-3702)")
plt.savefig("Figure-7227-3702.eps")
