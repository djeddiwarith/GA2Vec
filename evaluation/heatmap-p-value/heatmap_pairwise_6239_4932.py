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
"F1": [0, 0, 0, 0.355, 0, 0.611, 0, 0, 0.292],
    "NCV": [0.510, 0.532, 0.507, 0.837, 0.375, 0.675, 0.949, 0.949, 0.719],
    "$GS^3$": [0, 0, 0, 0.01, 0.02, 0.01, 0.01, 0.16, 0],
    "$NCV-GS^3$": [0.020, 0.016, 0.023, 0.099, 0.077, 0.114, 0.076, 0.389, 0.039],
    "GC": [0.699, 0.732, 0.698, 0.900, 0.852, 0.925, 0.273, 0.456, 0.710],
    "c=2": [0.839, 0.921, 0.922, 0.871, 0.8, 0.891, 0.111, 0.367, 0.664],
    "CIQ": [0.017, 0.043, 0.066, 0.009, 0, 0.015, 0.0008, 0.028, 0.017],
    "ICQ": [0.533, 0.529, 0.527, 0.589, 0.599, 0.239, 0.0003, 0.123, 0.463],
    "Sensitivity": [0.017, 0.042, 0.041, 0.035, 0.024, 0.048, 0.039, 0.043, 0.039]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (6239-4932)")
plt.savefig("Figure-6239-4932.eps")
