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
"F1": [0.209, 0.219, 0.209, 0.230, 0.339, 0.324, 0.933, 0.335, 0.247, 0.367],
    "NCV": [0.463, 0.340, 0.349, 0.365, 0.200, 0.206, 1.0, 0.121, 0.086, 0.070],
    "$GS^3$": [0, 0, 0, 0, 0.06, 0.03, 0, 0.31, 0, 0],
    "$NCV-GS^3$": [0.005, 0.013, 0.019, 0.018, 0.108, 0.073, 0.112, 0.194, 0, 0],
    "GC": [0.934, 0.870, 0.828, 0.876, 0.934, 0.928, 0.973, 0.607, 0.468, 0.943],
    "c=2": [0.666,0.7, 0.714, 0.7, 0.9, 0.909, 1.0, 0.289, 0.344, 0.846],
    "CIQ": [0, 0, 0, 0.208, 0, 0, 0.009, 0.2, 0, 0],
    "ICQ": [0.253, 0.327, 0.283, 0.295, 0.162, 0.154, 0.064, 0, 0, 0.119],
    "Sensitivity": [0.038,0.028, 0.027, 0.030, 0.020, 0.020, 0.039, 0.015, 0.007, 0.010]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="Greys", cbar=True,  annot_kws={"size": 14, "weight": "bold"}, linewidths=.5, linecolor='white')
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (10090-10116)")
plt.savefig("Figure-10090-10116.eps", dpi=300, bbox_inches='tight', pad_inches=0.1)
