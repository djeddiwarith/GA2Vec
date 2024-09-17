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
"F1": [0.136, 0.136, 0.139, 0.117, 0.120, 0.089, 0.146, 0.051, 0.127],
    "NCV": [0.110, 0.113, 0.105, 0.118, 0.110, 0.477, 0.164, 0.044, 0.058],
    "$GS^3$": [0, 0.01, 0.02, 0.02, 0.02, 0.01, 0.26, 0, 0],
    "$NCV-GS^3$": [0, 0.038, 0.041, 0.045, 0.050, 0.066, 0.206, 0, 0],
    "GC": [0.769, 0.823, 0.830, 0.909, 0.912, 0.949, 0.428, 0.352, 0.781],
    "c=2": [0.705, 0.705, 0.705, 1.0, 0.84, 0.844, 0.077, 0.076, 0.676],
    "CIQ": [0, 0, 0, 0.194, 0, 0, 0, 0, 0],
    "ICQ": [0.545, 0.566, 0.574, 0.453, 0.627, 0.135, 0, 0, 0.488],
    "Sensitivity": [0.005, 0.005, 0.005, 0.0004, 0.005, 0.014, 0.010, 0.0025, 0.004]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (4896-10116)")
plt.savefig("Figure-4896-10116.eps")
