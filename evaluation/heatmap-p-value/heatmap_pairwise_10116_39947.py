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
"F1": [0, 0, 0, 0, 0, 0.139, 0.335, 0.014, 0.1],
    "NCV": [0.126, 0.174, 0.107, 0.146, 0.146, 0.374, 0.882, 0.882, 0.146],
    "$GS^3$": [0, 0, 0, 0, 0, 0, 0.18, 0.13, 0.19],
    "$NCV-GS^3$": [0, 0, 0, 0, 0, 0, 0.397, 0.333, 0.166],
    "GC": [1.0, 1.0, 1.0, 0.9, 0.769, 0.978, 0.236, 0.307, 0.812],
    "c=2": [0, 0, 0, 0.666, 1.0, 1.0, 0.25, 0.276, 0.75],
    "CIQ": [0, 0, 0, 0, 0, 0.357, 0.090, 0, 0.4],
    "ICQ": [0.410, 0.365, 0.336, 0.561, 0.559, 0.143, 0, 0.055, 0.337],
    "Sensitivity": [0.0007, 0.0009, 0.0002, 0.0009, 0.0009, 0.005, 0.005, 0.006, 0.0006]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

# Plot heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (10116-39947)")
plt.savefig("Figure-10116-39947.eps")
