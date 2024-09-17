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
"F1": [0.047, 0.047, 0.046, 0.046, 0.059, 0.417, 0.040, 0.039, 0.039],
    "NCV": [0.334, 0.288, 0.291, 0.173, 0.187, 0.726, 0.538, 0.236, 0.187],
    "$GS^3$": [0, 0, 0, 0.01, 0.02, 0, 0.01, 0, 0],
    "$NCV-GS^3$": [0.010, 0.010, 0.011, 0.035, 0.056, 0.009, 0.060, 0, 0.019],
    "GC": [0.889, 0.889, 0.909, 0.936, 0.930, 0.963, 0.202, 0.129, 0.718],
    "c=2": [0.861, 0.837, 0.8, 0.763, 0.78, 0.834, 0.048, 0.054, 0.515],
    "CIQ": [0.112, 0.071, 0.106, 0.020, 0.160, 0.033, 0.0004, 0.0007, 0.024],
    "ICQ": [0.271, 0.268, 0.270, 0.277, 0.278, 0.042, 0.0001, 0.0002, 0.289],
    "Sensitivity": [0.071, 0.057, 0.057, 0.030, 0.035, 0.164, 0.003, 0.001, 0.075]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (9606-3702)")
plt.savefig("Figure-9606-3702.eps")
