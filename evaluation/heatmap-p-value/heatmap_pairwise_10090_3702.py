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
"F1": [0.100, 0.094, 0.095, 0.110, 0.103, 0.803, 0.047, 0.057, 0.093],
    "NCV": [0.503, 0.490, 0.508, 0.402, 0.407, 1.0, 0.549, 0.549, 0.316],
    "$GS^3$": [0, 0, 0, 0.01, 0, 0, 0.02, 0.09, 0.05],
    "$NCV-GS^3$": [0.010, 0.008, 0.020, 0.062, 0.042, 0.029, 0.09, 0.226, 0.128],
    "GC": [0.836, 0.809, 0.911, 0.954, 0.922, 0.973, 0.301, 0.461, 0.873],
    "c=2": [0.631, 0.578, 0.687, 0.875, 0.5, 1.0, 0.048, 0.414, 0.642],
    "CIQ": [0.042, 0.058, 0.033, 0.015, 0.012, 0.041, 0.0004, 0.020, 0.014],
    "ICQ": [0.236, 0.239, 0.231, 0.096, 0.098, 0.053, 0.0001, 0.080, 0.061],
    "Sensitivity": [0.040, 0.030, 0.030, 0.021, 0.023, 0.035, 0.003, 0.004, 0.023]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (10090-3702)")
plt.savefig("Figure-10090-3702.eps")
