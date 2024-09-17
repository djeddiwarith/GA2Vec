import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data_align = {
    "Approaches": [
"GA2Vec-ANCPRBT-f2",
                       "GA2Vec-ANCPRBT",
"GA2Vec-ANCESM",
"GA2Vec-ANCT5",
"GA2Vec-GM",
"GA2Vec-LP",
"GA2Vec-WT",
"GA2Vec-LT",
"GA2Vec-GM&LP",
"GA2Vec-GM&WT",
"GA2Vec-GM&LT",
"GA2Vec-LP&LT",
"GA2Vec-LP&WT",
"GA2Vec-LT&WT",
"GA2Vec-GM&LP&WT",
"GA2Vec-GM&LP&LT",
"GA2Vec-LP&WT&LT",
"MONACO",
        "SMETANA",
        "BEAMS",
        "multiMAGNA++",
        "IsoRankN"
    ],
    "F1": [0.469, 0.470,0.470, 0.470, 0.522, 0.500, 0.511, 0.534, 0.503, 0.485, 0.526, 0.500, 0.483, 0.497,0.479, 0.496, 0.697, 0.709, 0.528,0.404, 0.196, 0.408],
    "c=2": [0.914, 0.914, 0.915, 0.918, 0.934, 0.917, 0.915, 0.936, 0.923, 0.921,0.941, 0.937, 0.913, 0.915,0.908, 0.927, 0.820, 0.966, 0.951,0.970,  0.124, 0.6],
"c=3": [0.826, 0.824, 0.817, 0.825, 0.842, 0.815, 0.825, 0.875, 0.836, 0.824, 0.853, 0.832, 0.809, 0.837,0.831, 0.840, 0.659, 0.918, 0.894, 0.879, 0.099, 0.477],
    "CIQ": [0.139,0.110, 0.143, 0.099, 0.065, 0.079, 0.067, 0.068, 0.076, 0.063, 0.107, 0.089, 0.068, 0.102, 0.102, 0.095, 0.053, 0.011, 0.028,0.028,  0.0006, 0.007],
    "ICQ": [0.209,0.211, 0.211, 0.210, 0.230, 0.218, 0.233, 0.246, 0.221, 0.223, 0.239, 0.226, 0.218, 0.225, 0.214, 0.225, 0.002, 0.198, 0.107, 0.075, 0.00006, 0.086],
    "Sensitivity": [0.298,0.278, 0.282, 0.265, 0.151, 0.166, 0.172, 0.148, 0.237, 0.234, 0.218, 0.258, 0.230, 0.228, 0.249, 0.252, 0.032, 0.184, 0.195,0.279,  0.153, 0.039]
}

# Create DataFrame
df = pd.DataFrame(data_align)
df.set_index("Approaches", inplace=True)

# Plot heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(df, annot=True,cmap="Greys", cbar=True,  annot_kws={"size": 14, "weight": "bold"}, linewidths=.5, linecolor='white')
plt.title("Dataset (4932-9606-10090)")
plt.savefig("Figure-4932-9606-10090.eps", dpi=300, bbox_inches='tight', pad_inches=0.1)
