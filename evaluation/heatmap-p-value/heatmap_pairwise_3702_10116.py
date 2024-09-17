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
"F1": [0, 0, 0, 0, 0, 0.016, 0.014, 0.057, 0.025],
    "NCV": [0.095, 0.123, 0.106, 0.079, 0.061, 0.935, 0.047, 0.035, 0.028],
    "$GS^3$": [0, 0, 0, 0.02, 0.01, 0, 0.20, 0, 0],
    "$NCV-GS^3$": [0.011, 0.013, 0.007, 0.035, 0.019, 0, 0.097, 0, 0],
    "GC": [0.925, 0.924, 0.926, 0.857, 0.882, 0.951, 0, 0, 1.0],
    "c=2": [0, 0, 0, 0, 1.0, 1.0, 0, 0, 1.0],
    "CIQ": [0, 0, 0, 0, 0, 0.060, 0.2, 0, 0],
    "ICQ": [0.417, 0.393, 0.403, 0.391, 0.412, 0.027, 0, 0.002, 0.416],
    "Sensitivity": [0.002, 0.002, 0.002, 0.0009, 0.001, 0.0041, 0.00003, 0.00006, 0.001]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (3702-10116)")
plt.savefig("Figure-3702-10116.eps")
