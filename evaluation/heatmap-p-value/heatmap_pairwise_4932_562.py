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
"F1": [0.301, 0.296, 0.295, 0.331, 0.342, 0.226, 0.209,0.265, 0.256],
    "NCV": [0.308, 0.308, 0.297, 0.256, 0.256, 0.621,0.583, 0.340, 0.292],
    "$GS^3$": [0, 0, 0, 0.01, 0.01, 0, 0.01, 0, 0.01],
    "$NCV-GS^3$": [0.017, 0.028, 0.023, 0.058, 0.040, 0.027, 0.087, 0, 0.048],
    "GC": [0.750, 0.777, 0.753, 0.852, 0.865, 0.904, 0.287, 0.365, 0.558],
    "c=2": [0.828, 0.822, 0.828, 0.827, 0.805, 0.851, 0.083, 0.098, 0.318],
    "CIQ": [0.010, 0.018, 0.010, 0.018, 0.012, 0.29, 0.009,0.001, 0.019],
    "ICQ": [0.515, 0.503, 0.515, 0.518, 0.521, 0.196,0,0, 0.131],
    "Sensitivity": [0.056, 0.057, 0.056, 0.052, 0.049, 0.075, 0.068, 0.043, 0.043]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (4932-562)")
plt.savefig("Figure-4932-562.eps")
