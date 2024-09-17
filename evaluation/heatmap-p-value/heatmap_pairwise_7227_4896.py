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
"F1": [0.136, 0.140, 0.136, 0.147, 0.146, 0.311, 0.044, 0.028, 0.134],
    "NCV": [0.470, 0.458, 0.470, 0.330, 0.325, 0.560, 0.310, 0.138, 0.241],
    "$GS^3$": [0, 0, 0, 0.01, 0.01, 0, 0.02, 0, 0.01],
    "$NCV-GS^3$": [0.004, 0.010, 0.006, 0.058, 0.053, 0.030, 0.087, 0, 0.038],
    "GC": [0.716, 0.632, 0.786, 0.821, 0.817, 0.884, 0.268, 0.270, 0.700],
    "c=2": [0.852, 0.868, 0.855, 0.841, 0.828, 0.831, 0.086, 0.055, 0.700],
    "CIQ": [0.007, 0.006, 0.004, 0.005, 0, 0.001, 0.013, 0.0001, 0.011],
    "ICQ": [0.490, 0.494, 0.491, 0.535, 0.537, 0.171, 0.001, 0.0003, 0.418],
    "Sensitivity": [0.033, 0.034, 0.034, 0.030, 0.029, 0.049, 0.026, 0.002, 0.024]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (7227-4896)")
plt.savefig("Figure-7227-4896.eps")
