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
"F1": [0, 0, 0, 0.522, 0, 0.588, 0.0004, 0.0004, 0.484],
    "NCV": [0.842, 0.839, 0.839, 0.496, 0.502, 0.765, 0.483, 0.483, 1.0],
    "$GS^3$": [0, 0, 0, 0.01, 0.01, 0, 0.01, 0.13, 0],
    "$NCV-GS^3$": [0.010, 0.009, 0.009, 0.086, 0.081, 0.091, 0.052, 0.245, 0.006],
    "GC": [0.740, 0.727, 0.727, 0.939, 0.906, 0.953, 0.351, 0.389, 0.893],
    "c=2": [0.906, 0.917, 0.912, 0.875, 0.873, 0.875, 0.134, 0.155, 0.720],
    "CIQ": [0.088, 0.082, 0.084, 0.006, 0.019, 0.016, 0.0007, 0.014, 0.004],
    "ICQ": [0.450, 0.451, 0.449, 0.443, 0.442, 0.250, 0.0001, 0.0001, 0.361],
    "Sensitivity": [0.164, 0.169, 0.167, 0.133, 0.105, 0.170, 0.067, 0.058, 0.121]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (6239-9606)")
plt.savefig("Figure-6239-9606.eps")
