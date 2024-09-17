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
"F1": [0.050, 0.050, 0.051, 0.043, 0.065, 0.505, 0.022, 0.068, 0.062],
    "NCV": [0.206, 0.113, 0.104, 0.050, 0.025, 0.721, 0.022, 0.003, 0.022],
    "$GS^3$": [0, 0, 0, 0.01, 0, 0, 0.08, 0, 0],
    "$NCV-GS^3$": [0, 0.001, 0.001, 0.026, 0, 0.006, 0.042, 0, 0],
    "GC": [0.900, 0.937, 0.958, 0.941, 0.937, 0.962, 0.343, 0.5, 0.728],
    "c=2": [1.0, 1.0, 1.0, 1.0, 0.923, 0.838, 0.163, 0.285, 0.718],
    "CIQ": [0.036, 0.007, 0.011, 0.024, 0.073, 0.024, 0, 0, 0],
    "ICQ": [0.215, 0.222, 0.223, 0.283, 0.312, 0.027, 0, 0, 0.221],
    "Sensitivity": [0.021, 0.0212, 0.017, 0.012, 0.011, 0.167, 0.009, 0.0006, 0.004]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (9606-39947)")
plt.savefig("Figure-9606-39947.eps")
