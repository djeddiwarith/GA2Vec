import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data_align = {
    "Approaches": [
"GA2Vec-ANCPRBT-f2",
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
"F1": [0.463,0.456, 0.457, 0.457, 0.508, 0.483, 0.478, 0.224, 0.250, 0.440],
    "NCV": [0.701,0.670, 0.680, 0.675, 0.481, 0.417, 1.0, 0.521, 0.521, 0.419],
    "$GS^3$": [0,0, 0, 0, 0.02, 0.03, 0.02, 0.01, 0.13, 0.04],
    "$NCV-GS^3$": [0.010,0.016, 0.015, 0.015, 0.095, 0.107, 0.144, 0.051, 0.255, 0.127],
    "GC": [0.855, 0.801, 0.801, 0.814, 0.949, 0.945, 0.959, 0.511, 0.572, 0.827],
    "c=2": [0.907, 0.907, 0.906, 0.909, 0.934, 0.861, 0.877, 0.120, 0.151, 0.725],
    "CIQ": [0.090,0.104, 0.116, 0.113, 0.022, 0.049, 0.030, 0.0009, 0.032, 0.032],
    "ICQ": [0.514,0.509, 0.515, 0.511, 0.716, 0.520, 0.215, 0.00006, 0.0002, 0.431],
    "Sensitivity": [0.153,0.149, 0.152, 0.147, 0.041, 0.108, 0.206, 0.107, 0.093, 0.087]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="Greys", cbar=True,  annot_kws={"size": 14, "weight": "bold"}, linewidths=.5, linecolor='white')
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (4932-9606)")
plt.savefig("Figure-4932-9606.eps", dpi=300, bbox_inches='tight', pad_inches=0.1)
