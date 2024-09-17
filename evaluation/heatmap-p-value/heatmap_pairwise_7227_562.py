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
"F1": [0.210, 0.210, 0.213, 0.215, 0.211, 0.334, 0.102, 0.112, 0.143],
    "NCV": [0.203, 0.198, 0.193, 0.170, 0.164, 0.536, 0.418, 0.280, 0.298],
    "$GS^3$": [0, 0, 0, 0.01, 0.01, 0, 0.01, 0, 0],
    "$NCV-GS^3$": [0.014, 0.014, 0.015, 0.032, 0.034, 0.010, 0.065, 0, 0.018],
    "GC": [0.564, 0.559, 0.641, 0.757, 0.767, 0.849, 0.164, 0.182, 0.431],
    "c=2": [0.787, 0.793, 0.788, 0.781, 0.779, 0.804, 0.071, 0.083, 0.465],
    "CIQ": [0.026, 0.018, 0.021, 0.005, 0.014, 0.002, 0.007, 0.0005, 0.026],
    "ICQ": [0.554, 0.555, 0.559, 0.579, 0.586, 0.100, 0, 0, 0.255],
    "Sensitivity": [0.043, 0.043, 0.0412, 0.041, 0.040, 0.071, 0.055, 0.041, 0.055]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (7227-562)")
plt.savefig("Figure-7227-562.eps")
