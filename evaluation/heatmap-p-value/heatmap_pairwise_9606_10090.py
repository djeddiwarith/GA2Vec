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
"F1": [0.490, 0.487, 0.494, 0.489,0.769, 0.676, 0.727, 0.439, 0.169, 0.645],
    "NCV": [0.924, 0.818, 0.859, 0.818, 1.0, 0.377, 1.0, 0.245, 0.236, 0.219],
    "$GS^3$": [0, 0, 0, 0, 0.01, 0.02, 0, 0.02, 0, 0],
    "$NCV-GS^3$": [0, 0.009, 0.008, 0.005, 0.127, 0.085, 0.082, 0.061, 0, 0],
    "GC": [0.869, 0.855, 0.871, 0.855, 0.975, 0.953, 0.976, 0.654, 0.129, 0.938],
    "c=2": [0.921,0.935, 0.928, 0.931, 0.967, 0.956, 0.973, 0.282, 0.054, 0.883],
    "CIQ": [0.168,0.083, 0.120, 0.161, 0.003, 0.021, 0.005, 0.003, 0.0007, 0.017],
    "ICQ": [0.321,0.323, 0.328, 0.328, 0.221, 0.175, 0.231, 0.0002, 0.0002, 0.125],
    "Sensitivity": [0.216,0.204, 0.203, 0.200, 0.122, 0.14, 0.181, 0.087, 0.001, 0.100]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="Greys", cbar=True,  annot_kws={"size": 14, "weight": "bold"}, linewidths=.5, linecolor='white')
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (9606-10090)")
plt.savefig("Figure-9606-10090.eps", dpi=300, bbox_inches='tight', pad_inches=0.1)
