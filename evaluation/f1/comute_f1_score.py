import csv


# Function to parse PPI interaction files and create a mapping of Uniprot IDs to gene names
def parse_ppi_file(file_path):
    protein_to_gene = {}
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            protein_to_gene[row['Uniprot_A']] = row['Gene_A']
            protein_to_gene[row['Uniprot_B']] = row['Gene_B']
    return protein_to_gene


# Function to parse the DEG file and create a set of essential genes
def parse_deg_file(file_path, essential_genes):

    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=';', quotechar='"')
        next(reader)  # Skip header
        for row in reader:
            if (str(row[2])!=""):
                essential_genes.add(row[2])  # The gene name is in the third column
    return essential_genes


# Function to parse the alignment results
def parse_alignment_results(file_path):
    alignments = []
    with open(file_path, 'r') as file:
        for line in file:
            alignments.append(line.strip().split())
    return alignments


# Function to map proteins to genes and identify their species
def map_proteins_to_genes(alignments, protein_to_gene_maps):
    mapped_alignments = []
    for alignment in alignments:
        mapped_alignment = []
        for protein in alignment:
            found = False
            for species, protein_map in protein_to_gene_maps.items():
                if protein in protein_map:
                    mapped_alignment.append((protein_map[protein], species))
                    found = True
                    break
            if not found:
                mapped_alignment.append((protein, 'unknown'))
        mapped_alignments.append(mapped_alignment)
    return mapped_alignments


# Function to evaluate the F1 score based on essential genes
def evaluate_f1_score(mapped_alignments, essential_genes_map):
    tp = 0
    fp = 0
    fn = 0
    for alignment in mapped_alignments:
        species_genes = {species: set() for species in essential_genes_map.keys()}
        for gene, species in alignment:
            if species in species_genes:
                species_genes[species].add(gene)
        # for species in species_genes:
        #     print(species_genes[species].issubset(essential_genes_map[species]))

        if all(species_genes[species].issubset(essential_genes_map[species]) for species in species_genes):
            tp += 1
        else:
            if any(species_genes[species].issubset(essential_genes_map[species]) for species in species_genes):
                fp += 1
            else:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


# Paths to the input files
# Paths to the input files

alignment_results_path = '../lastresults/5spec/all_clusters_Protbert/f2/alignresults.txt'
#alignment_results_path = '../lastresults/smetana/algn_3spec.txt'
ppi_files = {
    #'562': '562_binary_hq.txt',
    #'3702': '3702_binary_hq.txt',
    #'4896': '4896_binary_hq.txt',
    '4932': '4932_binary_hq.txt',
    '6239': '6239_binary_hq.txt',
    '7227': '7227_binary_hq.txt',
    '9606': '9606_binary_hq.txt',
    '10090': '10090_binary_hq.txt',
    #'10116': '10116_binary_hq.txt',
    #'39947': '39947_binary_hq.txt',
}

deg_file_path1 = 'deg/deg_annotation_a/deg_annotation_a.csv'
deg_file_path2 = 'deg/deg_annotation_e/deg_annotation_e.csv'
deg_file_path3 = 'deg/deg_annotation_n/deg_annotation_n.csv'
deg_file_path4 = 'deg/deg_annotation_p/deg_annotation_p.csv'
# Parse the PPI files
protein_to_gene_maps = {species: parse_ppi_file(file_path) for species, file_path in ppi_files.items()}

# Parse the alignment results
alignment_results = parse_alignment_results(alignment_results_path)
essential_genes = set()
# Parse the DEG file
essential_genes = parse_deg_file(deg_file_path1, essential_genes)
essential_genes = parse_deg_file(deg_file_path2, essential_genes)
essential_genes = parse_deg_file(deg_file_path3, essential_genes)
essential_genes = parse_deg_file(deg_file_path4, essential_genes)
# Separate essential genes for each species
essential_genes_map = {species: {gene for gene in essential_genes if gene in protein_map.values()} for
                       species, protein_map in protein_to_gene_maps.items()}

# Map proteins to genes and identify their species
mapped_alignments = map_proteins_to_genes(alignment_results, protein_to_gene_maps)

# Evaluate the F1 score
f1_score = evaluate_f1_score(mapped_alignments, essential_genes_map)
print(f"F1 Score based on essential proteins: {f1_score}")
