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
"F1": [0.088, 0.081, 0.083, 0.102, 0.102, 0.330, 0.069, 0.034, 0.118],
    "NCV": [0.119, 0.158, 0.132, 0.084, 0.081, 0.535, 0.032, 0.026, 0.027],
    "$GS^3$": [0, 0, 0, 0.01, 0, 0, 0.21, 0, 0],
    "$NCV-GS^3$": [0.001, 0.002, 0.003, 0.032, 0.013, 0.010, 0.081, 0, 0],
    "GC": [0.662, 0.633, 0.549, 0.770, 0.753, 0.876, 0.312, 0.325, 0.743],
    "c=2": [0.846, 0.84, 0.84, 0.897, 0.890, 0.811, 0.164, 0.142, 0.737],
    "CIQ": [0.108, 0.097, 0.090, 0.222, 0.086, 0.003, 0.2, 0, 0.166],
    "ICQ": [0.382, 0.356, 0.344, 0.486, 0.513, 0.047, 0, 0, 0.542],
    "Sensitivity": [0.016, 0.021, 0.019, 0.015, 0.015, 0.041, 0.007, 0.005, 0.009]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (7227-10116)")
plt.savefig("Figure-7227-10116.eps")
