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
"F1": [0.247,0.245, 0.244, 0.247, 0.273, 0.281, 0.212, 0.091, 0.115, 0.212],
    "NCV": [0.709, 0.645, 0.648, 0.659, 0.481, 0.454, 0.882, 0.977, 0.205, 0.311],
    "$GS^3$": [0, 0, 0, 0, 0.02, 0.03, 0, 0.01, 0, 0],
    "$NCV-GS^3$": [0.012,0.020, 0.021, 0.023, 0.095, 0.113, 0.062, 0.076, 0, 0.037],
    "GC": [0.885,0.835, 0.862, 0.843, 0.949, 0.942, 0.937, 0.110, 0.405, 0.807],
    "c=2": [0.927, 0.928, 0.927, 0.928, 0.934, 0.939, 0.915, 0.064, 0.108, 0.751],
    "CIQ": [0.062,0.055, 0.046, 0.057, 0.022, 0.029, 0.027, 0.002, 0, 0.022],
    "ICQ": [0.649,0.649, 0.648, 0.646, 0.716, 0.724, 0.449, 0.0003, 0.0003, 0.527],
    "Sensitivity": [0.051,0.047, 0.047, 0.048, 0.041, 0.038, 0.057, 0.002, 0.016, 0.026]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="Greys", cbar=True,  annot_kws={"size": 14, "weight": "bold"}, linewidths=.5, linecolor='white')
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (4932-4896)")
plt.savefig("Figure-4932-4896.eps", dpi=300, bbox_inches='tight', pad_inches=0.1)
