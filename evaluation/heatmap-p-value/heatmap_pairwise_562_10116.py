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
    "F1": [0.083, 0.08, 0.08, 0.076, 0.148, 0.343, 0.262, 0.338, 0.272],
    "NCV": [0.043, 0.047, 0.047, 0.041, 0.037, 0.502, 0.117, 0.095, 0.0160],
    "$GS^3$": [0,0.04, 0, 0.02, 0.09, 0, 0.24, 0, 0],
    "$NCV-GS^3$": [0, 0.041, 0, 0.031, 0.058, 0.016, 0.166, 0, 0],
    "GC": [0.611, 0.625, 0.625, 0.586, 0.547, 0.805, 0.440, 0.533, 0.722],
    "c=2": [0.545, 0.545, 0.545, 0.545, 0.647, 0.735, 0.098, 0.090, 0.384],
    "CIQ": [0, 0, 0, 0, 0.4, 0.0162, 0.266, 0, 0],
    "ICQ": [0.584, 0.576, 0.576, 0.576, 0.0611,0, 0,0, 0.416],
    "Sensitivity": [0.005, 0.006, 0.006, 0.006, 0.035,0, 0.012, 0.008, 0.002]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (562-10116)")
plt.savefig("Figure-562-10116.eps")
