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
    "F1": [0.172, 0.161, 0.177, 0.179, 0.181, 0.243,0.140, 0.176, 0.149],
    "NCV": [0.231, 0.217, 0.220, 0.186, 0.174, 0.546, 0.820, 0.749, 0.244],
    "$GS^3$": [0, 0, 0, 0, 0, 0, 0.02, 0, 0.02],
    "$NCV-GS^3$": [0.008, 0.021, 0, 0.027, 0.013, 0.012, 0.137, 0, 0.073],
    "GC": [0.765, 0.745, 0.775, 0.794, 0.812, 0.855, 0.282, 0.328, 0.578],
    "c=2": [0.850, 0.862, 0.843, 0.818, 0.812, 0.796, 0.059, 0.059, 0.493],
    "CIQ": [0.025, 0.020, 0.032, 0, 0, 0.006, 0.003, 0, 0.025],
    "ICQ": [0.642, 0.636, 0.636, 0.706, 0.667, 0.206, 0, 0, 0.260],
    "Sensitivity": [0.023, 0.023, 0.022, 0.022, 0.021, 0.045, 0.051, 0.047, 0.020]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (562-4896)")
plt.savefig("Figure-562-4896.eps")
