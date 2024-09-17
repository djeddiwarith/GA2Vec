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
"F1": [0.043, 0.046, 0.044, 0.037, 0.037, 0.072, 0.011, 0.068, 0.028],
    "NCV": [0.136, 0.114, 0.126, 0.137, 0.110, 0.477, 0.203, 0.064, 0.150],
    "$GS^3$": [0, 0, 0, 0.01, 0.02, 0.01, 0.14, 0, 0],
    "$NCV-GS^3$": [0.014, 0, 0, 0.043, 0.050, 0.051, 0.168, 0, 0],
    "GC": [0.937, 0.898, 0.926, 0.920, 0.912, 0.946, 0.229, 0.129, 0.486],
    "c=2": [1.0, 1.0, 1.0, 0.909, 0.84, 0.864, 0, 0.071, 0.375],
    "CIQ": [0, 0, 0, 0.194, 0, 0.083, 0, 0, 0.131],
    "ICQ": [0.559, 0.570, 0.566, 0.453, 0.627, 0.132, 0.002, 0, 0.174],
    "Sensitivity": [0.003, 0.002, 0.003, 0.0004, 0.005, 0.010, 0.004, 0.001, 0.002]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (4896-39947)")
plt.savefig("Figure-4896-39947.eps")
