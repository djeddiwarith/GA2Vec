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
        # "SMETANA",
        "BEAMS",
        "multiMAGNA++",
        "HubAlign",
"IsoRankN"
    ],
"F1": [0.086, 0.073, 0.080, 0.116, 0.110, 0.084, 0.048, 0.039, 0.060],
    "NCV": [0.250, 0.241, 0.248, 0.213, 0.204, 0.847, 0.564, 0.564, 0.210],
    "$GS^3$": [0, 0, 0, 0, 0, 0, 0.02, 0.12, 0.03],
    "$NCV-GS^3$": [0.009, 0.005, 0.011, 0.021, 0.030, 0.017, 0.103, 0.262, 0.085],
    "GC": [0.722, 0.610, 0.735, 0.827, 0.811, 0.814, 0.06, 0.2, 0.553],
    "c=2": [0.642, 0.642, 0.615, 0.722, 0.722, 0.746, 0.037, 0.24, 0.228],
    "CIQ": [0.014, 0.003, 0.008, 0.007, 0.004, 0.044, 0.020, 0.066, 0.019],
    "ICQ": [0.518, 0.523, 0.524, 0.543, 0.551, 0.124, 0.0001, 0.099, 0.141],
    "Sensitivity": [0.018, 0.018, 0.017, 0.012, 0.009, 0.028, 0.001, 0.002, 0.017]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)
# Normalize MNE
#df["MNE"] = 1 - (df["MNE"] - df["MNE"].min()) / (df["MNE"].max() - df["MNE"].min())


plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True,cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (562-3702)")
plt.savefig("figure-562-3702.eps")
