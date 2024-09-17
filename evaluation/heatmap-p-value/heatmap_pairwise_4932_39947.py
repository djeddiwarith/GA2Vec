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
"F1": [0.035, 0.033, 0.036, 0.027, 0.027, 0.152, 0.033, 0.072, 0.034],
    "NCV": [0.097, 0.077, 0.08, 0.070, 0.082, 0.581, 0.062, 0.018, 0.045],
    "$GS^3$": [0, 0, 0, 0.02, 0.01, 0, 0.11, 0, 0],
    "$NCV-GS^3$": [0.017, 0.007, 0.020, 0.041, 0.028, 0.010, 0.081, 0, 0],
    "GC": [0.852, 0.918, 0.937, 0.894, 0.932, 0.937, 0.257, 0.208, 0.514],
    "c=2": [1.0, 1.0, 1.0, 1.0, 1.0, 0.890, 0.125, 0.058, 0.473],
    "CIQ": [0, 0, 0, 0.102, 0.2, 0.050, 0, 0, 0],
    "ICQ": [0.486, 0.480, 0.487, 0.476, 0.479, 0.062, 0.001, 0, 0.069],
    "Sensitivity": [0.006, 0.005, 0.004, 0.005, 0.005, 0.038, 0.004, 0.001, 0.002]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (4932-39947)")
plt.savefig("Figure-4932-39947.eps")
