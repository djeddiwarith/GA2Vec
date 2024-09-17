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
"F1": [0.027, 0.026, 0, 0.044,  0.022, 0.328, 0.011, 0, 0.015],
    "NCV": [0.109, 0.114, 0.108, 0.063, 0.083, 0.534, 0.040, 0.007, 0.041],
    "$GS^3$": [0, 0, 0, 0.01, 0.01, 0, 0.12, 0, 0],
    "$NCV-GS^3$": [0.003, 0.005, 0.004, 0.026, 0.022, 0, 0.068, 0, 0],
    "GC": [0.748, 0.652, 0.726, 0.841, 0.838, 0.872, 0.187, 0.25, 0.571],
    "c=2": [0.7, 0.727, 0.7, 0.764, 0.928, 0.809, 0.142, 0, 0.737],
    "CIQ": [0, 0, 0, 0, 0.048, 0.003, 0, 0, 0.166],
    "ICQ": [0.357, 0.371, 0.381, 0.431, 0.384, 0.040, 0, 0, 0.542],
    "Sensitivity": [0.009, 0.009, 0.009, 0.005, 0.008, 0.036, 0.003, 0, 0.009]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (7227-39947)")
plt.savefig("Figure-7227-39947.eps")
