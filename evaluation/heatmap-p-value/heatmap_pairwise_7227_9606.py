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
"F1": [0.270, 0.270, 0.273, 0.451, 0.242, 0.440, 0.092, 0.103, 0.381],
    "NCV": [0.995, 0.981, 0.985, 0.592, 0.649, 0.746, 0.707, 0.707, 0.553],
    "$GS^3$": [0, 0, 0, 0.01, 0.01, 0, 0, 0.12, 0.02],
    "$NCV-GS^3$": [0.011, 0.012, 0.010, 0.093, 0.078, 0.072, 0.045, 0.288, 0.097],
    "GC": [0.710, 0.628, 0.692, 0.929, 0.909, 0.942, 0.305, 0.320, 0.671],
    "c=2": [0.887, 0.887, 0.887, 0.863, 0.865, 0.871, 0.157, 0.139, 0.667],
    "CIQ": [0.080, 0.067, 0.073, 0.004, 0.008, 0.012, 0.0004, 0, 0.002],
    "ICQ": [0.410, 0.410, 0.411, 0.387, 0.404, 0.302, 0.0001, 0.0002, 0.340],
    "Sensitivity": [0.236, 0.231, 0.227, 0.169, 0.159, 0.204, 0.118, 0.030, 0.083]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (7227-9606)")
plt.savefig("Figure-7227-9606.eps")
