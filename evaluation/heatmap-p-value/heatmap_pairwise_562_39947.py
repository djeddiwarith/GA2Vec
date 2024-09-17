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
    ],
"F1": [0.086, 0.086, 0.086, 0.08, 0.08, 0.320, 0.011, 0.071],
    "NCV": [0.036, 0.034, 0.035, 0.036, 0.034, 0.507, 0.146, 0.067],
    "$GS^3$": [0, 0, 0, 0, 0, 0, 0.11, 0],
    "$NCV-GS^3$": [0, 0, 0, 0, 0, 0, 0.126, 0],
    "GC": [0.637, 0.729, 0.631, 0.642, 0.622, 0.811, 0.149, 0.25],
    "c=2": [1.0, 1.0, 1.0, 1.0, 1.0, 0.730, 0.068, 0.095],
    "CIQ": [0, 0, 0, 0, 0, 0.023, 0.062, 0],
    "ICQ": [0.650, 0.678, 0.674, 0.706, 0.728, 0.062, 0, 0],
    "Sensitivity": [0.002, 0.002, 0.002, 0.002, 0.002, 0.031, 0.005, 0.001]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (562-39947)")
plt.savefig("Figure-562-39947.eps")
