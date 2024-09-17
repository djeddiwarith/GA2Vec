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
"F1": [0.132, 0.135, 0.117, 0.296, 0.310, 0.505, 0.294, 0.218, 0.417],
    "NCV": [0.207, 0.212, 0.214, 0.073, 0.073, 0.722, 0.017, 0.014, 0.016],
    "$GS^3$": [0, 0, 0, 0.01, 0.01, 0, 0.17, 0, 0],
    "$NCV-GS^3$": [0.001, 0.003, 0.003, 0.032, 0.024, 0.012, 0.055, 0, 0],
    "GC": [0.900, 0.904, 0.907, 0.949, 0.930, 0.962, 0.625, 0.717, 0.954],
    "c=2": [0.894, 0.944, 0.9, 0.914, 0.94, 0.837, 0.163, 0.392, 0.915],
    "CIQ": [0.086, 0.075, 0.064, 0.148, 0.120, 0.021, 0, 0, 0.166],
    "ICQ": [0.234, 0.234, 0.241, 0.400, 0.419, 0.033, 0, 0, 0.642],
    "Sensitivity": [0.048, 0.049, 0.049, 0.025, 0.028, 0.170, 0.009, 0.006, 0.011]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (9606-10116)")
plt.savefig("Figure-9606-10116.eps")
