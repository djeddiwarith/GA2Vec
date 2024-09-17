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
"F1": [0.267, 0.250, 0.259, 0.266, 0.301, 0.324, 0.475, 0.164, 0.169, 0.292],
    "NCV": [0.578,0.529, 0.504, 0.535, 0.282, 0.377, 0.732, 0.183, 0.236, 0.137],
    "$GS^3$": [0, 0, 0, 0, 0.01, 0.02, 0, 0.02, 0, 0],
    "$NCV-GS^3$": [0.004, 0.003, 0.005, 0.002, 0.049, 0.085, 0.025, 0.065, 0, 0],
    "GC": [0.889, 0.908, 0.864, 0.921, 0.939, 0.953, 0.959, 0.455, 0.129, 0.844],
    "c=2": [0.887, 0.892, 0.901, 0.897, 0.868, 0.956, 0.850, 0.090, 0.054, 0.712],
    "CIQ": [0.098,0.086, 0.082, 0.118, 0.012, 0.021, 0.020, 0.004, 0.0007, 0.010],
    "ICQ": [0.419, 0.415, 0.419, 0.419, 0.472, 0.175, 0.108, 0.0001, 0.0002, 0.453],
    "Sensitivity": [0.098,0.103, 0.092, 0.104, 0.067, 0.140, 0.177, 0.042, 0.001, 0.033]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="Greys", cbar=True,  annot_kws={"size": 14, "weight": "bold"}, linewidths=.5, linecolor='white')
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (9606-4896)")
plt.savefig("Figure-9606-4896.eps", dpi=300, bbox_inches='tight', pad_inches=0.1)
