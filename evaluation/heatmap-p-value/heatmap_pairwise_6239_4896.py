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
"F1": [0, 0, 0, 0, 0, 0.683, 0, 0, 0],
    "NCV": [0.422, 0.444, 0.430, 0.321, 0.320, 0.539, 0.480, 0.289, 0.289],
    "$GS^3$": [0, 0, 0, 0.02, 0.01, 0, 0.03, 0, 0],
    "$NCV-GS^3$": [0.016, 0.0125, 0.013, 0.088, 0.049, 0.070, 0.121, 0, 0.36],
    "GC": [0.735, 0.671, 0.628, 0.790, 0.797, 0.883, 0.219, 0.298, 0.628],
    "c=2": [0.839, 0.852, 0.843, 0.819, 0.8, 0.854, 0.080, 0.106, 0.625],
    "CIQ": [0.017, 0.018, 0.019, 0, 0, 0.005, 0.004, 0, 0.013],
    "ICQ": [0.533, 0.533, 0.519, 0.567, 0.599, 0.087, 0.00009, 0, 0.362],
    "Sensitivity": [0.017, 0.019, 0.018, 0.016, 0.024, 0.021, 0.018, 0.014, 0.015]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True,  annot_kws={"size": 14, "weight": "bold"})
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("Dataset (6239-4896)")
plt.savefig("Figure-6239-4896.eps")
