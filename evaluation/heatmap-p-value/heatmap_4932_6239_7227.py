import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data_align = {
    "Approaches": [
        "GA2Vec-ANCPRBT-f2",
        "GA2Vec-ANCPRBT",
"GA2Vec-ANCESM",
"GA2Vec-ANCT5",
"GA2Vec-LP",
"GA2Vec-GM",
"GA2Vec-LT",
"GA2Vec-WT",
"GA2Vec-LP&LT",
"GA2Vec-GM&LT",
"GA2Vec-GM&LP",
"GA2Vec-GM&WT",
"GA2Vec-LP&WT",
"GA2Vec-LT&WT",
"GA2Vec-GM&LP&LT",
"GA2Vec-GM&LP&WT",
"GA2Vec-LP&WT&LT",
"MONACO",
        "SMETANA",
         "BEAMS",
        "multiMAGNA++",
        "IsoRankN"
    ],
    "F1": [0.091,0.091, 0.092, 0.091, 0.101, 0.105, 0.105, 0.106, 0.104, 0.105, 0.102, 0.095, 0.095, 0.102, 0.097, 0.091,0.102, 0.367, 0.071, 0.487, 0.006, 0.169],
    "c=2": [0.928,0.923, 0.923, 0.921, 0.936, 0.943, 0.939, 0.919, 0.928, 0.938, 0.936, 0.927, 0.925, 0.922, 0.939, 0.919, 0.922, 0.877, 0.875, 0.879, 0.082, 0.755],
"c=3": [0.795, 0.798, 0.802, 0.792, 0.848, 0.812, 0.823, 0.808, 0.806, 0.815, 0.822, 0.822, 0.808, 0.827, 0.807, 0.788, 0.827, 0.847, 0.755, 0.878, 0.077, 0.687],
    "CIQ": [0.079,0.069, 0.070, 0.060, 0.030, 0.015, 0.041, 0.018, 0.033, 0.039, 0.033, 0.031, 0.027, 0.036, 0.048, 0.040, 0.036, 0.013, 0.017, 0.012, 0.0009, 0.032],
    "ICQ": [0.148,0.148, 0.148, 0.150, 0.158, 0.166, 0.164, 0.155, 0.156, 0.164, 0.156, 0.155, 0.155, 0.152, 0.152, 0.149, 0.152, 0.159, 0.159, 0.121, 0.000009, 0.189],
    "Sensitivity": [0.113,0.106, 0.106, 0.103, 0.065, 0.062, 0.061, 0.060, 0.084, 0.088, 0.087, 0.087, 0.089, 0.092, 0.100, 0.104,0.092, 0.085, 0.0844, 0.094, 0.078, 0.050]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

# Plot heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True,cmap="Greys", cbar=True,  annot_kws={"size": 14, "weight": "bold"}, linewidths=.5, linecolor='white')
plt.title("Dataset (4932-6239-7227)")
plt.savefig("Figure-4932-6239-7227.eps", dpi=300, bbox_inches='tight', pad_inches=0.1)
