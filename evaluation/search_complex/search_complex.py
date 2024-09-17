import pandas as pd

# Load PPI network for human and yeast into dataframes
human_ppi_file = 'HomoSapiens_binary_hq.txt'
yeast_ppi_file = 'SaccharomycesCerevisiae_binary_hq.txt'
alignment_file = '../../results/GA2Vec/2spec/all_clusters_protbert/4932-9606/f2/alignresults_pairwise.txt'
output_file = 'alignment_with_genes_and_source.txt'
# Read PPI network files (adjust column names as necessary)
human_ppi = pd.read_csv(human_ppi_file, sep='\t')
yeast_ppi = pd.read_csv(yeast_ppi_file, sep='\t')

# Create dictionaries mapping UniProt IDs to gene names
human_uniprot_to_gene = pd.Series(human_ppi['Gene_A'].values, index=human_ppi['Uniprot_A']).to_dict()
human_uniprot_to_gene.update(pd.Series(human_ppi['Gene_B'].values, index=human_ppi['Uniprot_B']).to_dict())

yeast_uniprot_to_gene = pd.Series(yeast_ppi['Gene_A'].values, index=yeast_ppi['Uniprot_A']).to_dict()
yeast_uniprot_to_gene.update(pd.Series(yeast_ppi['Gene_B'].values, index=yeast_ppi['Uniprot_B']).to_dict())

# Create dictionaries for finding direct neighbors
human_neighbors = {}
yeast_neighbors = {}

# Populate neighbor dictionaries for fast lookup
for index, row in human_ppi.iterrows():
    human_neighbors.setdefault(row['Uniprot_A'], []).append(row['Uniprot_B'])
    human_neighbors.setdefault(row['Uniprot_B'], []).append(row['Uniprot_A'])

for index, row in yeast_ppi.iterrows():
    yeast_neighbors.setdefault(row['Uniprot_A'], []).append(row['Uniprot_B'])
    yeast_neighbors.setdefault(row['Uniprot_B'], []).append(row['Uniprot_A'])

# Helper function to map UniProt IDs to gene names and their species
def map_uniprot_to_gene_with_source(uniprot_id):
    if uniprot_id in human_uniprot_to_gene:
        return f"{human_uniprot_to_gene[uniprot_id]}_human", "human"
    elif uniprot_id in yeast_uniprot_to_gene:
        return f"{yeast_uniprot_to_gene[uniprot_id]}_yeast", "yeast"
    else:
        return uniprot_id, "unknown"

# Helper function to retrieve neighbors from the PPI network
def get_neighbors(protein_id, species):
    if species == "human":
        return human_neighbors.get(protein_id, [])
    elif species == "yeast":
        return yeast_neighbors.get(protein_id, [])
    return []

# Function to check if a gene contains the specified substring
def contains_orc(gene_name):
    return 'ORC' in gene_name.upper()

# Read alignment file and filter results based on a given gene name (e.g. "ORC")
with open(alignment_file, 'r') as file, open(output_file, 'w') as outfile:
    for line in file:
        proteins = line.strip().split()

        if len(proteins) < 2:
            continue  # skip lines with less than two proteins

        protein1, species1 = map_uniprot_to_gene_with_source(proteins[0])
        protein2, species2 = map_uniprot_to_gene_with_source(proteins[1])

        # Only print if one of the proteins has 'ORC' in the gene name
        if (contains_orc(protein1) or contains_orc(protein2)):
            if species1 == "human" and species2 == "yeast":
                # Print alignment triplet
                outfile.write(f"{protein1} aligned_to {protein2}\n")


                # Print interactions for human (protein1)
                neighbors1 = get_neighbors(proteins[0], species1)
                for neighbor in neighbors1:
                    neighbor_gene, _ = map_uniprot_to_gene_with_source(neighbor)
                    outfile.write(f"{protein1} interact_with {neighbor_gene}\n")

                # Print interactions for yeast (protein2)
                neighbors2 = get_neighbors(proteins[1], species2)
                for neighbor in neighbors2:
                    neighbor_gene, _ = map_uniprot_to_gene_with_source(neighbor)
                    outfile.write(f"{protein2} interact_with {neighbor_gene}\n")

            elif species1 == "yeast" and species2 == "human":
                # Print alignment triplet (reverse order for yeast and human)
                outfile.write(f"{protein2} aligned_to {protein1}\n")


                # Print interactions for yeast (protein1)
                neighbors1 = get_neighbors(proteins[1], species2)
                for neighbor in neighbors1:
                    neighbor_gene, _ = map_uniprot_to_gene_with_source(neighbor)
                    outfile.write(f"{protein2} interact_with {neighbor_gene}\n")

                # Print interactions for human (protein2)
                neighbors2 = get_neighbors(proteins[0], species1)
                for neighbor in neighbors2:
                    neighbor_gene, _ = map_uniprot_to_gene_with_source(neighbor)
                    outfile.write(f"{protein1} interact_with {neighbor_gene}\n")

print(f"Filtered triplet results containing 'ORC' and their interactions written to {output_file}")
