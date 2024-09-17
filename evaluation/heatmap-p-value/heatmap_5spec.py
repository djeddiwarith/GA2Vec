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
        "IsoRankN"
    ],
"F1": [0.173, 0.177, 0.175, 0.530, 0.176,0.169, 0.041, 0.267],
    "c=2": [0.901, 0.892, 0.892, 0.934, 0.904,0.958, 0.136, 0.75],
"c=3": [0.843, 0.825, 0.836, 0.887, 0.871, 0.851, 0.050, 0.721],
"c=4": [0.795, 0.786, 0.789, 0.833, 0.773, 0.817, 0.042, 0.692],
"c=5": [0.597, 0.638, 0.581, 0.786, 0.710, 0.728,0.038, 0.66],
    "CIQ": [0.070, 0.107, 0.104, 0.010, 0.018, 0.019, 0.0004, 0.007],
    "ICQ": [0.185, 0.186, 0.186, 0.163, 0.093, 0.090, 0.00000008, 0.101],
    "Sensitivity": [0.359, 0.384, 0.378, 0.266, 0.280,0.337, 0.204, 0.156]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

# Plot heatmap
plt.figure(figsize=(16, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (7227-6239-4932-9606-10090)")
plt.savefig("Figure-5spec.eps")
