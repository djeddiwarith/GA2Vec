import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Sample data: runtime in seconds for each approach across different pairwise alignments
data = {
    'Dataset': ['SARS2-SARS1', 'HMSARS2-HMSARS1','562-3702', '562-4896',
                         '562-10116', '562-39947', '3702-4896', '3702-10116', '3702-39947',
                         '4896-10116', '4896-39947',
                         '4932-562', '4932-3702', '4932-4896',
                         '4932-9606', '4932-10090', '4932-10116', '4932-39947', '6239-562', '6239-3702', '6239-4896', '6239-4932',
                         '6239-9606', '6239-10090', '6239-10116',
                         '6239-39947', '7227-562', '7227-3702', '7227-4896', '7227-4932', '7227-6239', '7227-9606', '7227-10090', '7227-10116',
                         '7227-39947', '9606-562', '9606-3702', '9606-4896', '9606-10090',
                         '9606-10116', '9606-39947', '10090-562', '10090-3702',
                         '10090-4896', '10090-10116', '10090-39947',
                         '10116-39947', '4932-9606-10090', '7227-6239-4932',
                         '7227-6239-4932-9606-10090', '7227-6239-4932-9606-10090-562-4896-10116', '10 species (all)'],
"multiMAGNA++": [1, 1, 2, 2, 2,2, 2, 2, 2, 1,1, 2, 3, 2, 10,2, 2, 1, 2, 3,1, 3, 8, 2, 1,1, 3, 4, 3, 5,4, 10, 3, 2, 2,7, 9, 7, 7, 7,1, 1, 1, 1, 1,1, 1,9, 6, 13, 15,16],
"HubAlign": [1, 1, 28, 8, 1, 1, 10, 3, 1, 1, 1, 28, 180, 12, 6,1, 1, 1, 21, 348,23, 450, 58, 54, 2,4, 200, 840, 120, 600,14, 120, 420, 1, 1,360, 1620, 120, 8, 14,21, 60, 90, 25, 2,1, 1, np.nan, np.nan, np.nan, np.nan,np.nan],

"IsoRankN": [4, np.nan, 300, 17, 4, np.nan, 53, 14, 18, 20, 8, 80, 180, 72, 660, 60, 3, 4, 60, 180,60, 120, 540, 60, 7,6, 120, 300, 60, 300,240, 900, 120, 8, 10,180, 540, 120, 180, 40,35, 15, 60, 11, 2,2, 1, 660, 360, 1140, 1500,np.nan],
"GA2Vec-ANCPRBT-f2": [34.23,3314.61,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,9907.251,85179.194,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,46067.309,43136.159,np.nan,np.nan,np.nan,np.nan,1660.481,690.429,np.nan,np.nan,67428.549,104471.902,0,0,0],
    "GA2Vec-ANCPRBT": [35.071,1422.575,154.303, 122.349, 105.074, 107.651, 145.467, 128.626, 124.967,
                       90.520, 93.103, 59.817, 194.053, 162.174, 486.413, 159.493,128.346,
                       127.099, 152.911, 188.380, 136.941, 198.802, 425.840, 155.024,
                       121.607, 118.311, 191.122, 244.656, 179.120, 276.908, 226.116, 723.787,
                       201.023, 161.143, 163.108,143.297, 607.342,
                       261.670, 338.598, 229.881, 236.457, 41.153,
                       151.189, 116.676, 102.823, 102.076,
                       48.418, 698.041, 555.729, 4296, 4140.716, 6402.601 ],

    "GA2Vec-ANCESM": [34.710, 1610.450,   147.636, 120.048, 103.758, 106.689, 139.140, 123.257, 125.849,
                      88.832, 87.482, 59.282, 176.279, 142.153, 618.387, 161.777, 121.630,
                      119.090, 151.908, 177.645, 139.455, 199.407, 531.305, 148.835,
                      122.039, 118.982, 183.092, 237.20, 237.020, 259.477, 222.886, 869.431,
                      203.409, 175.673, 157.292, 143.136, 544.687,
                      287.648, 401.744, 219.174, 221.300, 40.573,
                      146.676, 116.783, 102.252, 101.988,
                      47.986, 534.140, 866.064, 3255.583, 5853.526, 9565.493],

    "GA2Vec-ANCT5": [34.990, 417.880,135.066, 105.992, 100.337, 99.352, 123.495, 118.757, 118.336,
                     87.766, 86.421, 59.975, 150.557, 117.981, 262.776, 134.709, 120.778,
                     119.975, 132.008, 160.521, 124.523, 164.507, 280.153, 131.679,
                     114.081, 113.49, 162.586, 181.910, 150.966, 182.061, 173.863, 330.129,
                     165.159, 145.870, 145.717, 146.438, 250.374,
                     211.103, 246.184, 201.899, 204.417,
                     40.741, 153.798, 124.271, 120.582,
                     121.466, 69.529, 534.668, 418.093,
                     1440.563, 2617.304, 4155.005],

    "BEAMS": [1, 2, 25, 4, 5, 6,15, 13, 12, 2, 2, 7, 40, 7, 14400, 43200, 3, 3,
              82, 86, 50, 63, 23500, 23760, 12, 60, 72, 103, 40, 106, 259,
              19.620, 55800, 76, 70, 14400, 14400, 14700,156420, 10740, 13440, 48300, 47700, 48120, 45900, 47820, 3, 8160, 960, 432780, np.nan, np.nan],

}

# Create a DataFrame
df = pd.DataFrame(data)

# Set the index to the pairwise datasets for a better heatmap layout
df.set_index('Dataset', inplace=True)
# Define the default colormap
cmap = sns.color_palette("YlGnBu", as_cmap=True)

# Create a mask for all np.nan values
nan_mask = df.isnull()

# Create a specific mask for np.nan values in 'GA2Vec-ANCPRBT-f2'
nan_ancprbt_f2_mask = df['GA2Vec-ANCPRBT-f2'].isnull()

# Create the figure and axis
plt.figure(figsize=(12, 10))

# Plot the heatmap, but allow np.nan values to remain visible
heatmap = sns.heatmap(df, annot=True, fmt=".0f", linewidths=0.5,cmap=cmap, mask=nan_mask, cbar_kws={'label': 'Runtime (s)'}, linecolor='black')

# Overlay a red color specifically for np.nan in 'GA2Vec-ANCPRBT-f2'
for i in range(len(df)):
    if nan_ancprbt_f2_mask[i]:  # Check if np.nan is present in 'GA2Vec-ANCPRBT-f2'
        heatmap.add_patch(plt.Rectangle((df.columns.get_loc('GA2Vec-ANCPRBT-f2'), i), 1, 1, fill=True, color='red', lw=0))

# Add matrix pattern with black borders for np.nan cells in other approaches
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        if pd.isna(df.iloc[i, j]) and df.columns[j] != 'GA2Vec-ANCPRBT-f2':
            # Create a transparent rectangle as a base
            heatmap.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=0))

            # Add matrix pattern inside the rectangle
            matrix_size = 0.2  # Size of each matrix cell
            for x in np.arange(j, j + 1, matrix_size):
                for y in np.arange(i, i + 1, matrix_size):
                    rect = plt.Rectangle((x, y), matrix_size, matrix_size, fill=True, edgecolor='black', color='gray', lw=0.5)
                    heatmap.add_patch(rect)

# Set the title and labels
plt.title('Runtime for Each Approach Across Different Datasets')
plt.xlabel('Approach')
plt.ylabel('Datasets')

# Adjust layout for better spacing
plt.tight_layout()

# Save the figure
plt.savefig("runingtime_matrix_borders.eps", format='eps')

# Show the plot
plt.show()


