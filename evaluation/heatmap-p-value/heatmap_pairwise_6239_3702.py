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
"F1": [0, 0, 0, 0, 0, 0.592, 0, 0, 0],
    "NCV": [0.552, 0.553, 0.552, 0.463, 0.453, 0.676, 0.926, 0.926, 0.418],
    "$GS^3$": [0, 0, 0, 0.01, 0.01, 0, 0.01, 0.14, 0],
    "$NCV-GS^3$": [0.013, 0.012, 0.014, 0.081, 0.075, 0.050, 0.088, 0.362, 0.025],
    "GC": [0.641, 0.551, 0.629, 0.824, 0.755, 0.882, 0.147, 0.373, 0.507],
    "c=2": [0.826, 0.869, 0.826, 0.921, 0.8, 0.878, 0.062, 0.357, 0.729],
    "CIQ": [0.013, 0.022, 0.031, 0.028, 0, 0.048, 0.0009, 0.018, 0.013],
    "ICQ": [0.395, 0.403, 0.397, 0.394, 0.599, 0.080, 0.0002, 0.112, 0.185],
    "Sensitivity": [0.009, 0.009, 0.010, 0.005, 0.024, 0.009, 0.001, 0.001, 0.001]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (6239-3702)")
plt.savefig("Figure-6239-3702.eps")
