import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data_align = {
    "Approaches": [
"MONACO",
        "SMETANA",
        "BEAMS",
        "multiMAGNA++",
        "HubAlign",
        "IsoRankN"
    ],
    "GA2Vec-ANCPRBT": [0, 1, 0, 16, 31, 11],
    "GA2Vec-ANCPESM": [1,1,0,16, 31,12],
    "GA2Vec-ANCT5": [0,1,0,14,31,13],
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True, cmap="Greys", cbar=True,  annot_kws={"size": 14, "weight": "bold"}, linewidths=.5, linecolor='white')
plt.xticks(fontsize=12, weight='bold')  # Increase size and boldness of x-axis labels
plt.yticks(fontsize=12, weight='bold')  # Increase size and boldness of y-axis labels
plt.title("")
plt.savefig("Figure-overall_p_value.eps", dpi=300, bbox_inches='tight', pad_inches=0.1)
