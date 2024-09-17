import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = {
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
"F1": [0.019, 0.02, 0.20, 0.016, 0.016, 0.116, 0.108, 0, 0.030],
    "NCV": [0.258, 0.258, 0.257, 0.146, 0.153, 0.949, 0.047, 0.015, 0.056],
    "$GS^3$": [0, 0, 0, 0.01, 0.02, 0, 0, 0, 0],
    "$NCV-GS^3$": [0.001, 0.002, 0.002, 0.045, 0.050, 0.012, 0, 0, 0],
    "GC": [0.990, 0.890, 0.991, 1.0, 0.862, 1.0, 0.25, 0, 0.736],
    "c=2": [0, 0, 0, 1.0, 0, 1.0, 0, 0, 1.0],
    "CIQ": [0.349, 0.345, 0.347, 0.194, 0.27, 0.061, 0, 0, 0.057],
    "ICQ": [0.366, 0.368, 0.359, 0.453, 0.446, 0.055, 0, 0, 0.413],
    "Sensitivity": [0.001, 0.0008, 0.001, 0.0004, 0.0002, 0.0006, 0, 0.0003, 0.0007]
}

# Create DataFrame
df = pd.DataFrame(data)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (3702-39947)")
plt.savefig("Figure-3702-39947.eps")
