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
"F1": [0, 0, 0, 0, 0, 0.713, 0, 0, 0],
    "NCV": [0.112, 0.111, 0.111, 0.081, 0.085, 0.645, 0.069, 0.013, 0.067],
    "$GS^3$": [0, 0, 0, 0.01, 0.01, 0, 0.09, 0, 0],
    "$NCV-GS^3$": [0, 0, 0.007, 0.029, 0.026, 0.013, 0.078, 0, 0],
    "GC": [0.779, 0.803, 0.791, 0.785, 0.765, 0.872, 0.18, 0.083, 0.6],
    "c=2": [0.888, 0.888, 0.888, 0.833, 0.909, 0.851, 0.173, 0, 0.7],
    "CIQ": [0, 0, 0.25, 0.222, 0.222, 0.018, 0, 0, 0.108],
    "ICQ": [0.354, 0.360, 0.365, 0.403, 0.383, 0.037, 0, 0, 0.210],
    "Sensitivity": [0.003, 0.003, 0.003, 0.003, 0.0031, 0.011, 0.002, 0.0004, 0.002]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (6239-39947)")
plt.savefig("Figure-6239-39947.eps")
