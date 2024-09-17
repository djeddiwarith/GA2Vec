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
    "F1": [0.086, 0.085, 0.084, 0.093, 0.101, 0.076, 0.019, 0.042,     0.052],
    "NCV": [0.652, 0.648, 0.661, 0.505, 0.472, 0.882, 0.977, 0.977,     0.374],
    "$GS^3$": [0, 0, 0, 0.02, 0.01, 0, 0.01, 0.02, 0.02],
    "$NCV-GS^3$": [0.023, 0.017, 0.018, 0.090, 0.073, 0.062, 0.076, 0.127,    0.092],
    "GC": [0.785, 0.760, 0.754, 0.917, 0.908, 0.937, 0.110,0.482,      0.788],
    "c=2": [0.865, 0.882, 0.877, 0.868, 0.774, 0.865, 0.064,0.5,      0.761],
    "CIQ": [0.086, 0.062, 0.079, 0.031, 0.077, 0.046, 0.002, 0.007,        0.037],
    "ICQ": [0.499, 0.497, 0.497, 0.497, 0.522, 0.241, 0.0003,0.170,         0.190],
    "Sensitivity": [0.029, 0.026, 0.030, 0.018, 0.016, 0.034, 0.002, 0.003,       0.016]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (4932-3702)")
plt.savefig("Figure-4932-3702.eps")
