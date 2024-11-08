import warnings
import time
import random
import igraph as ig
import community as community_louvain
from networkx.algorithms.community import girvan_newman
import math
from collections import defaultdict
import networkx as nx
import numpy as np
from ftplib import FTP
from io import StringIO
from multiprocessing import Pool, cpu_count, Manager
from networkx.algorithms import community

# Download the GO association file for a given species from UniProt-GOA via FTP
def download_annotations(species):
    # Connect to the FTP server and navigate to the appropriate directory
    ftp = FTP('ftp.ebi.ac.uk')
    ftp.login()
    ftp.cwd(f'/pub/databases/GO/goa/{species.upper()}/')

    # Find the name of the latest annotation file
    latest_file = None
    latest_date = None
    for filename in ftp.nlst():
        if filename.endswith('.gz'):
            file_date = ftp.sendcmd(f'MDTM {filename}').split()[1]
            if latest_date is None or file_date > latest_date:
                latest_file = filename
                latest_date = file_date

    # Download the latest annotation file
    file_data = StringIO()
    ftp.retrbinary(f'RETR {latest_file}', file_data.write)

    # Parse the annotation data into a dictionary of gene-GO term associations
    annotations = {}
    file_data.seek(0)
    for line in file_data:
        if line.startswith('!'):
            continue
        fields = line.strip().split('\t')
        gene = fields[1]
        go_term = fields[4]
        if gene not in annotations:
            annotations[gene] = set()
        annotations[gene].add(go_term)

    return annotations


def compute_similarity_for_node(node_data):
    n1, go_terms1, network1_ids, ListProtbert1, network1_embd, annotations2, network2_ids, ListProtbert2, network2_embd, alpha, go_embeddings, protein_go_embeddings = node_data
    threshold1 = 0.6
    go_terms1 = set(go_terms1)
    similarities = []
    similarity_score_emb = 0

    for n2 in network2_ids:
        try:
            succes_emb_sim = False

            embed1 = network1_embd[ListProtbert1[network1_ids.index(n1)]]
            embed2 = network2_embd[ListProtbert2[network2_ids.index(n2)]]

            similarity_score_emb = np.dot(embed1, embed2) / (
                    np.linalg.norm(embed1) * np.linalg.norm(embed2)
            )

            succes_emb_sim = True
        except:
            pass

        go_terms2 = set(annotations2.get(n2, []))
        intersect = len(go_terms1.intersection(go_terms2))

        embeddings1 = [go_embeddings[go_term] for go_term in go_terms1 if go_term in go_embeddings]
        if embeddings1:
            aggregated_embedding = np.mean(embeddings1, axis=0)
        else:
            # Handle cases where no valid embeddings are found
            aggregated_embedding = np.zeros(200)  # Use appropriate embedding dimension
        protein_go_embeddings_n1 = aggregated_embedding

        embeddings2 = [go_embeddings[go_term] for go_term in go_terms2 if go_term in go_embeddings]
        if embeddings2:
            aggregated_embedding = np.mean(embeddings2, axis=0)
        else:
            # Handle cases where no valid embeddings are found
            aggregated_embedding = np.zeros(200)  # Use appropriate embedding dimension
        protein_go_embeddings_n2 = aggregated_embedding
        similarity_score_emb_got = np.dot(protein_go_embeddings_n1, protein_go_embeddings_n2) / (
                np.linalg.norm(protein_go_embeddings_n1) * np.linalg.norm(protein_go_embeddings_n2)
        )

        if intersect > 0:
            numerator = intersect
            denominator = math.sqrt(len(go_terms1) * len(go_terms2))
            simgo = numerator / denominator

            if succes_emb_sim:
                weighted_similarity = alpha * similarity_score_emb_got + (1 - alpha) * similarity_score_emb
                # weighted_similarity = alpha * simgo + (1 - alpha) * similarity_score_emb
                # weighted_similarity = alpha * simgo + (1 - alpha) * similarity_score_emb_got
                if (float(weighted_similarity) > threshold1):
                    similarities.append((n1, n2, weighted_similarity))

            elif float(simgo) > threshold1:
                similarities.append((n1, n2, simgo))

    return similarities


def compute_similarities(network1, annotations1, network2, annotations2, similarities, network1_ids, network2_ids,
                         network1_embd, network2_embd, alpha=1.0):
    # Load pre-trained embeddings and IDs

    network1_ids = [entry.split('|')[1] for entry in network1_ids]
    network2_ids = [entry.split('|')[1] for entry in network2_ids]

    B = {key: i for i, key in enumerate(network1_ids)}
    ListProtbert1 = [B[key] for key in network1_ids]

    C = {key: i for i, key in enumerate(network2_ids)}
    ListProtbert2 = [C[key] for key in network2_ids]

    network1_embd = network1_embd[ListProtbert1, :]
    network2_embd = network2_embd[ListProtbert2, :]

    # Convert the annotations to sets
    for go in annotations1:
        annotations1[go] = set(annotations1[go])
    for go in annotations2:
        annotations2[go] = set(annotations2[go])
    loaded_embedding = np.load('data/input/datavp/GOTembed/embeddings.npy', allow_pickle=True).item()

    go_embeddings = {}

    # Reconstruct the dictionary using the loaded embeddings list
    for i, t in enumerate(loaded_embedding):
        go_embeddings[t] = loaded_embedding[t]

    # Prepare input data for multiprocessing
    node_data_list = []
    protein_go_embeddings = {}
    for n1 in network1.nodes():
        go_terms1 = annotations1.get(n1, [])
        node_data_list.append((n1, go_terms1, network1_ids, ListProtbert1, network1_embd, annotations2, network2_ids,
                               ListProtbert2, network2_embd, alpha, go_embeddings, protein_go_embeddings))

    # Multiprocessing using all available CPU cores
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(compute_similarity_for_node, node_data_list)

    # Consolidate the results from all processes

    for similarity_list in results:
        for n1, n2, sim in similarity_list:
            similarities[(n1, n2)] = sim

    return similarities


def read_network(network_file):
    edges = []
    with open(network_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # Skip the first line
            fields = line.strip().split('\t')
            if len(fields) == 2:
                edges.append((fields[0], fields[1], {'weight': 1}))
            elif len(fields) == 3:
                edges.append((fields[0], fields[1], {'weight': float(fields[2])}))
            else:
                continue
    return edges


def fitness_function_node_similarity(solution, similarities):
    aligned_nodes = set()

    # Calculate the sum of similarities for aligned nodes
    for cluster in solution:
        for node_pair in combinations(cluster, 2):
            aligned_nodes.add(node_pair)

    total_similarity = sum(similarities[(node_pair[0], node_pair[1])] for node_pair in aligned_nodes)

    return total_similarity


def parse_annotations_intern(filename, network):
    annotations = defaultdict(list)
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('!'):
                continue
            fields = line.strip().split('\t')
            if fields[1] in network.nodes() and fields[6] != "IEA":
                for go_term in fields[4].split(';'):
                    annotations[fields[1]].append(go_term)
    return annotations


def normalize_bit_score(bit_score):
    similarity = 1 / (1 + math.exp(-bit_score))
    return similarity


def compute_similarities_between_two_species_coronavirus_with_human(network1_file, network2_file, output_file,
                                                                    similarities, annotation1file,
                                                                    annotation2file, annotation3file, network1_ids,
                                                                    network2_ids, network1_embd, network2_embd, alpha):
    # Read the network files as graphs

    network1 = nx.Graph(read_network(network1_file))

    network2 = nx.Graph(read_network(network2_file))

    # Parse the GO annotations for species 1 and species 2
    annotations1 = parse_annotations_first_coronavirus_human(annotation1file, annotation3file, network1)
    annotations2 = parse_annotations_first_coronavirus_human(annotation2file, annotation3file, network2)

    # Compute the pairwise similarities between nodes in the two networks
    similarities = compute_similarities(network1, annotations1, network2, annotations2, similarities, network1_ids,
                                        network2_ids, network1_embd, network2_embd, alpha)

    # Output the similarities to a file
    with open(output_file, 'w') as f:
        for nodes, similarity in similarities.items():
            f.write(f"{nodes[0]}\t{nodes[1]}\t{similarity}\n")
    return similarities


def compute_similarities_between_two_species_coronavirus(network1_file, network2_file, output_file, similarities,
                                                         annotation1file,
                                                         annotation2file, network1_ids, network2_ids, network1_embd,
                                                         network2_embd, alpha):
    # Read the network files as graphs

    network1 = nx.Graph(read_network(network1_file))

    network2 = nx.Graph(read_network(network2_file))

    # Parse the GO annotations for species 1 and species 2
    annotations1 = parse_annotations_first_coronavirus(annotation1file, network1)
    annotations2 = parse_annotations_first_coronavirus(annotation2file, network2)

    # Compute the pairwise similarities between nodes in the two networks
    similarities = compute_similarities(network1, annotations1, network2, annotations2, similarities, network1_ids,
                                        network2_ids, network1_embd, network2_embd, alpha)

    # Output the similarities to a file
    with open(output_file, 'w') as f:
        for nodes, similarity in similarities.items():
            f.write(f"{nodes[0]}\t{nodes[1]}\t{similarity}\n")
    return similarities


def compute_similarities_between_two_species(network1_file, network2_file, output_file, similarities, annotation1file,
                                             annotation2file, network1_ids, network2_ids, network1_embd, network2_embd,
                                             alpha):
    # Read the network files as graphs

    network1 = nx.Graph(read_network(network1_file))

    network2 = nx.Graph(read_network(network2_file))

    # Parse the GO annotations for species 1 and species 2
    annotations1 = parse_annotations_intern(annotation1file, network1)
    annotations2 = parse_annotations_intern(annotation2file, network2)

    # Compute the pairwise similarities between nodes in the two networks
    similarities = compute_similarities(network1, annotations1, network2, annotations2, similarities, network1_ids,
                                        network2_ids, network1_embd, network2_embd, alpha=alpha)

    # Output the similarities to a file
    with open(output_file, 'w') as f:
        for nodes, similarity in similarities.items():
            f.write(f"{nodes[0]}\t{nodes[1]}\t{similarity}\n")
    return similarities


class C:
    k = 0


def girvan_newman_clustering(graph):
    # Girvan-Newman algorithm
    comp = girvan_newman(graph)
    clusters = [tuple(c) for c in next(comp)]

    return clusters


def create_clusters_girvan_newman():
    # Call Girvan-Newman clustering instead of max clique
    return girvan_newman_clustering(sim_ppi)


def louvain_clustering(graph):
    partition = community_louvain.best_partition(graph)
    clusters = {}
    for node, cluster_id in partition.items():
        clusters.setdefault(cluster_id, []).append(node)
    return clusters


def create_clusters():
    # Call Louvain clustering instead of max clique
    return louvain_clustering(sim_ppi)


def save_alignment_to_file2(alignment, filename):
    try:
        with open(filename, 'w') as file:
            for cluster in alignment:
                # Filter out single protein clusters
                if len(cluster) > 1:
                    proteins_to_write = []
                    for v in cluster:
                        if isinstance(v, list):
                            proteins_to_write.extend(v)
                        else:
                            proteins_to_write.append(v)
                    # Only write if more than one protein is present
                    if len(proteins_to_write) > 1:
                        file.write(' '.join(proteins_to_write) + '\n')
        print(f"File saved successfully to {filename}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


def compute_sim_between_networks(datafile, similarities, output_file, alpha):
    f = open(datafile, 'r')
    lines = f.readlines()
    print("datafile opened.")
    # global k
    C.k = int(lines[0])  # number of networks
    # print(k)
    print("reading networks file......")
    model_name = "Protbert"  # or "ESM", "T5", "Protbert" .
    for i in range(1, C.k + 1):
        netfile = lines[i].strip()
        for j in range(i + 1, C.k + 1):
            netfile2 = lines[j].strip()
            print(netfile, netfile2)
            # new
            if (netfile == "data/input/datavp2/datavp/10116.tab") and (
                    netfile2 == "data/input/datavp2/datavp/39947.tab"):
                print("10116-39947")
                annotation1file = "data/input/datavp2/datavp/gaf/10116.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/39947.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/rattus/rattus_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/oryza/oryza_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/rattus/rattus_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/oryza/oryza_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)

            # new
            if (netfile == "data/input/datavp2/datavp/4896.tab") and (
                    netfile2 == "data/input/datavp2/datavp/39947.tab"):
                print("4896-39947")
                annotation1file = "data/input/datavp2/datavp/gaf/4896.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/39947.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/pombe/pombe_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/oryza/oryza_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/pombe/pombe_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/oryza/oryza_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/4896.tab") and (
                    netfile2 == "data/input/datavp2/datavp/10116.tab"):
                print("4896-10116")
                annotation1file = "data/input/datavp2/datavp/gaf/4896.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/10116.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/pombe/pombe_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/rattus/rattus_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/pombe/pombe_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/rattus/rattus_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)

            # new
            if (netfile == "data/input/datavp2/datavp/3702.tab") and (
                    netfile2 == "data/input/datavp2/datavp/39947.tab"):
                print("3702-39947")
                annotation1file = "data/input/datavp2/datavp/gaf/3702.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/39947.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/aradopsis/aradopsis_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/oryza/oryza_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/aradopsis/aradopsis_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/oryza/oryza_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/3702.tab") and (
                    netfile2 == "data/input/datavp2/datavp/10116.tab"):
                print("3702-10116")
                annotation1file = "data/input/datavp2/datavp/gaf/3702.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/10116.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/aradopsis/aradopsis_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/rattus/rattus_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/aradopsis/aradopsis_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/rattus/rattus_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/3702.tab") and (
                    netfile2 == "data/input/datavp2/datavp/4896.tab"):
                print("3702-4896")
                annotation1file = "data/input/datavp2/datavp/gaf/3702.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/4896.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/aradopsis/aradopsis_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/pombe/pombe_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/aradopsis/aradopsis_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/pombe/pombe_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)

            # new
            if (netfile == "data/input/datavp2/datavp/562.tab") and (
                    netfile2 == "data/input/datavp2/datavp/39947.tab"):
                print("562-39947")
                annotation1file = "data/input/datavp2/datavp/gaf/562.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/39947.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/ecoli/ecoli_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/oryza/oryza_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/ecoli/ecoli_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/oryza/oryza_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/562.tab") and (
                    netfile2 == "data/input/datavp2/datavp/10116.tab"):
                print("562-10116")
                annotation1file = "data/input/datavp2/datavp/gaf/562.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/10116.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/ecoli/ecoli_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/rattus/rattus_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/ecoli/ecoli_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/rattus/rattus_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/562.tab") and (
                    netfile2 == "data/input/datavp2/datavp/4896.tab"):
                print("562-4896")
                annotation1file = "data/input/datavp2/datavp/gaf/562.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/4896.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/ecoli/ecoli_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/pombe/pombe_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/ecoli/ecoli_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/pombe/pombe_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/562.tab") and (
                    netfile2 == "data/input/datavp2/datavp/3702.tab"):
                print("562-3702")
                annotation1file = "data/input/datavp2/datavp/gaf/562.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/3702.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/ecoli/ecoli_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/aradopsis/aradopsis_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/ecoli/ecoli_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/aradopsis/aradopsis_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)

            # new
            if (netfile == "data/input/datavp2/datavp/10090.tab") and (
                    netfile2 == "data/input/datavp2/datavp/39947.tab"):
                print("10090-39947")
                annotation1file = "data/input/datavp2/datavp/gaf/10090.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/39947.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/mouse/mouse_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/oryza/oryza_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/mouse/mouse_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/oryza/oryza_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/10090.tab") and (
                    netfile2 == "data/input/datavp2/datavp/10116.tab"):
                print("10090-10116")
                annotation1file = "data/input/datavp2/datavp/gaf/10090.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/10116.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/mouse/mouse_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/rattus/rattus_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/mouse/mouse_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/rattus/rattus_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/10090.tab") and (
                    netfile2 == "data/input/datavp2/datavp/4896.tab"):
                print("10090-4896")
                annotation1file = "data/input/datavp2/datavp/gaf/10090.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/4896.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/mouse/mouse_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/pombe/pombe_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/mouse/mouse_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/pombe/pombe_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/10090.tab") and (
                    netfile2 == "data/input/datavp2/datavp/3702.tab"):
                print("10090-3702")
                annotation1file = "data/input/datavp2/datavp/gaf/10090.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/3702.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/mouse/mouse_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/aradopsis/aradopsis_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/mouse/mouse_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/aradopsis/aradopsis_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)

            # new
            if (netfile == "data/input/datavp2/datavp/9606.tab") and (
                    netfile2 == "data/input/datavp2/datavp/39947.tab"):
                print("9606-39947")
                annotation1file = "data/input/datavp2/datavp/gaf/9606.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/39947.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/human/human_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/oryza/oryza_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/human/human_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/oryza/oryza_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/9606.tab") and (
                    netfile2 == "data/input/datavp2/datavp/10116.tab"):
                print("9606-10116")
                annotation1file = "data/input/datavp2/datavp/gaf/9606.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/10116.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/human/human_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/rattus/rattus_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/human/human_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/rattus/rattus_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/9606.tab") and (
                    netfile2 == "data/input/datavp2/datavp/4896.tab"):
                print("9606-4896")
                annotation1file = "data/input/datavp2/datavp/gaf/9606.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/4896.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/human/human_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/pombe/pombe_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/human/human_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/pombe/pombe_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/9606.tab") and (
                    netfile2 == "data/input/datavp2/datavp/3702.tab"):
                print("9606-3702")
                annotation1file = "data/input/datavp2/datavp/gaf/9606.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/3702.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/human/human_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/aradopsis/aradopsis_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/human/human_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/aradopsis/aradopsis_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)

            # new
            if (netfile == "data/input/datavp2/datavp/4932.tab") and (
                    netfile2 == "data/input/datavp2/datavp/39947.tab"):
                print("4932-39947")
                annotation1file = "data/input/datavp2/datavp/gaf/4932.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/39947.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/yeast/Yeast_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/oryza/oryza_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/yeast/Yeast_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/oryza/oryza_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/4932.tab") and (
                    netfile2 == "data/input/datavp2/datavp/10116.tab"):
                print("4932-10116")
                annotation1file = "data/input/datavp2/datavp/gaf/4932.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/10116.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/yeast/Yeast_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/rattus/rattus_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/yeast/Yeast_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/rattus/rattus_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/4932.tab") and (
                    netfile2 == "data/input/datavp2/datavp/4896.tab"):
                print("4932-4896")
                annotation1file = "data/input/datavp2/datavp/gaf/4932.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/4896.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/yeast/Yeast_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/pombe/pombe_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/yeast/Yeast_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/pombe/pombe_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/4932.tab") and (
                    netfile2 == "data/input/datavp2/datavp/3702.tab"):
                print("4932-3702")
                annotation1file = "data/input/datavp2/datavp/gaf/4932.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/3702.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/yeast/Yeast_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/aradopsis/aradopsis_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/yeast/Yeast_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/aradopsis/aradopsis_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)

            if (netfile == "data/input/datavp2/datavp/6239.tab") and (
                    netfile2 == "data/input/datavp2/datavp/562.tab"):
                print("6239-562")
                annotation1file = "data/input/datavp2/datavp/gaf/6239.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/562.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/celeg/celeg_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/ecoli/ecoli_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/celeg/celeg_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/ecoli/ecoli_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/6239.tab") and (
                    netfile2 == "data/input/datavp2/datavp/39947.tab"):
                print("6239-39947")
                annotation1file = "data/input/datavp2/datavp/gaf/6239.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/39947.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/celeg/celeg_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/oryza/oryza_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/celeg/celeg_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/oryza/oryza_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/6239.tab") and (
                    netfile2 == "data/input/datavp2/datavp/10116.tab"):
                print("6239-10116")
                annotation1file = "data/input/datavp2/datavp/gaf/6239.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/10116.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/celeg/celeg_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/rattus/rattus_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/celeg/celeg_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/rattus/rattus_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/6239.tab") and (
                    netfile2 == "data/input/datavp2/datavp/4896.tab"):
                print("6239-4896")
                annotation1file = "data/input/datavp2/datavp/gaf/6239.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/4896.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/celeg/celeg_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/pombe/pombe_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/celeg/celeg_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/pombe/pombe_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/6239.tab") and (
                    netfile2 == "data/input/datavp2/datavp/3702.tab"):
                print("6239-3702")
                annotation1file = "data/input/datavp2/datavp/gaf/6239.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/3702.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/celeg/celeg_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/aradopsis/aradopsis_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/celeg/celeg_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/aradopsis/aradopsis_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)

            # new
            if (netfile == "data/input/datavp2/datavp/7227.tab") and (
                    netfile2 == "data/input/datavp2/datavp/39947.tab"):
                print("7227-39947")
                annotation1file = "data/input/datavp2/datavp/gaf/7227.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/39947.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/droso/droso_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/oryza/oryza_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/droso/droso_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/oryza/oryza_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/7227.tab") and (
                    netfile2 == "data/input/datavp2/datavp/10116.tab"):
                print("7227-10116")
                annotation1file = "data/input/datavp2/datavp/gaf/7227.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/10116.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/droso/droso_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/rattus/rattus_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/droso/droso_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/rattus/rattus_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/7227.tab") and (
                    netfile2 == "data/input/datavp2/datavp/4896.tab"):
                print("7227-4896")
                annotation1file = "data/input/datavp2/datavp/gaf/7227.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/4896.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/droso/droso_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/pombe/pombe_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/droso/droso_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/pombe/pombe_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            # new
            if (netfile == "data/input/datavp2/datavp/7227.tab") and (
                    netfile2 == "data/input/datavp2/datavp/3702.tab"):
                print("7227-3702")
                annotation1file = "data/input/datavp2/datavp/gaf/7227.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/3702.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/droso/droso_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/aradopsis/aradopsis_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/droso/droso_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/aradopsis/aradopsis_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file,
                                                                        similarities,
                                                                        annotation1file, annotation2file,
                                                                        network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            if (netfile == "data/input/coronavirus_blast_sim2/human_sars2.tab") and (
                    netfile2 == "data/input/coronavirus_blast_sim2/human_sars1.tab"):
                print("sars-human2-sars-human1")
                annotation1file = "data/input/sars-goa-uniprotkb/uniprotkb_severe_acute_respiratory_2024_01_01.tsv"
                annotation2file = "data/input/sars-goa-uniprotkb/uniprotkb_severe_acute_respiratory_2024_01_01.tsv"
                annotation3file = "data/input/datavp2/datavp/gaf/9606.gaf"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/sars_human/sars_and_human_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/sars_human/sars_and_human_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/sars_human/sars_and_human_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/sars_human/sars_and_human_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species_coronavirus_with_human(netfile, netfile2,
                                                                                               output_file,
                                                                                               similarities,
                                                                                               annotation1file,
                                                                                               annotation2file,
                                                                                               annotation3file,
                                                                                               network1_ids,
                                                                                               network2_ids,
                                                                                               network1_embd,
                                                                                               network2_embd, alpha)
            if (netfile == "data/input/coronavirus_blast_sim2/sars2.tab") and (
                    netfile2 == "data/input/coronavirus_blast_sim2/sars1.tab"):
                print("sars2-sars1")
                annotation1file = "data/input/sars-goa-uniprotkb/uniprotkb_severe_acute_respiratory_2024_01_01.tsv"
                annotation2file = "data/input/sars-goa-uniprotkb/uniprotkb_severe_acute_respiratory_2024_01_01.tsv"
                network1_ids = np.load(
                    f'data/input/embedding-species/{model_name}/sars12/sars_ids.npy')
                network2_ids = np.load(
                    f'data/input/embedding-species/{model_name}/sars12/sars_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/sars12/sars_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/sars12/sars_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species_coronavirus(netfile, netfile2, output_file,
                                                                                    similarities,
                                                                                    annotation1file, annotation2file,
                                                                                    network1_ids,
                                                                                    network2_ids, network1_embd,
                                                                                    network2_embd, alpha)

            # new

            if (netfile == "data/input/datavp2/datavp/7227.tab") and (
                    netfile2 == "data/input/datavp2/datavp/562.tab"):
                print("7227-562")
                annotation1file = "data/input/datavp2/datavp/gaf/7227.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/562.gaf"
                network1_ids = np.load(f'data/input/embedding-species/{model_name}/droso/droso_ids.npy')
                network2_ids = np.load(f'data/input/embedding-species/{model_name}/ecoli/ecoli_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/droso/droso_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/ecoli/ecoli_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file, similarities,
                                                                        annotation1file, annotation2file, network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)

            if (netfile == "data/input/datavp2/datavp/7227.tab") and (
                    netfile2 == "data/input/datavp2/datavp/6239.tab"):
                print("7227-6239")
                annotation1file = "data/input/datavp2/datavp/gaf/7227.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/6239.gaf"
                network1_ids = np.load(f'data/input/embedding-species/{model_name}/droso/droso_ids.npy')
                network2_ids = np.load(f'data/input/embedding-species/{model_name}/celeg/celeg_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/droso/droso_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/celeg/celeg_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file, similarities,
                                                                        annotation1file, annotation2file, network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            if (netfile == "data/input/datavp2/datavp/7227.tab") and (
                    netfile2 == "data/input/datavp2/datavp/4932.tab"):
                print("7227-4932")
                annotation1file = "data/input/datavp2/datavp/gaf/7227.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/4932.gaf"
                network1_ids = np.load(f'data/input/embedding-species/{model_name}/droso/droso_ids.npy')
                network2_ids = np.load(f'data/input/embedding-species/{model_name}/yeast/Yeast_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/droso/droso_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/yeast/Yeast_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file, similarities,
                                                                        annotation1file, annotation2file, network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            if (netfile == "data/input/datavp2/datavp/7227.tab") and (
                    netfile2 == "data/input/datavp2/datavp/9606.tab"):
                print("7227-9606")
                annotation1file = "data/input/datavp2/datavp/gaf/7227.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/9606.gaf"
                network1_ids = np.load(f'data/input/embedding-species/{model_name}/droso/droso_ids.npy')
                network2_ids = np.load(f'data/input/embedding-species/{model_name}/human/human_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/droso/droso_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/human/human_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file, similarities,
                                                                        annotation1file, annotation2file, network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            if (netfile == "data/input/datavp2/datavp/7227.tab") and (
                    netfile2 == "data/input/datavp2/datavp/10090.tab"):
                print("7227-10090")
                annotation1file = "data/input/datavp2/datavp/gaf/7227.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/10090.gaf"
                network1_ids = np.load(f'data/input/embedding-species/{model_name}/droso/droso_ids.npy')
                network2_ids = np.load(f'data/input/embedding-species/{model_name}/mouse/mouse_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/droso/droso_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/mouse/mouse_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file, similarities,
                                                                        annotation1file, annotation2file, network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            if (netfile == "data/input/datavp2/datavp/6239.tab") and (
                    netfile2 == "data/input/datavp2/datavp/4932.tab"):
                print("6239-4932")
                annotation1file = "data/input/datavp2/datavp/gaf/6239.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/4932.gaf"
                network1_ids = np.load(f'data/input/embedding-species/{model_name}/celeg/celeg_ids.npy')
                network2_ids = np.load(f'data/input/embedding-species/{model_name}/yeast/Yeast_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/celeg/celeg_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/yeast/Yeast_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file, similarities,
                                                                        annotation1file, annotation2file, network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            if (netfile == "data/input/datavp2/datavp/6239.tab") and (
                    netfile2 == "data/input/datavp2/datavp/9606.tab"):
                print("6239-9606 ")
                annotation1file = "data/input/datavp2/datavp/gaf/6239.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/9606.gaf"
                network1_ids = np.load(f'data/input/embedding-species/{model_name}/celeg/celeg_ids.npy')
                network2_ids = np.load(f'data/input/embedding-species/{model_name}/human/human_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/celeg/celeg_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/human/human_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file, similarities,
                                                                        annotation1file, annotation2file, network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            if (netfile == "data/input/datavp2/datavp/6239.tab") and (
                    netfile2 == "data/input/datavp2/datavp/10090.tab"):
                print("6239-10090")
                annotation1file = "data/input/datavp2/datavp/gaf/6239.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/10090.gaf"
                network1_ids = np.load(f'data/input/embedding-species/{model_name}/celeg/celeg_ids.npy')
                network2_ids = np.load(f'data/input/embedding-species/{model_name}/mouse/mouse_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/celeg/celeg_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/mouse/mouse_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file, similarities,
                                                                        annotation1file, annotation2file, network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            if (netfile == "data/input/datavp2/datavp/4932.tab") and (
                    netfile2 == "data/input/datavp2/datavp/9606.tab"):
                print("4932-9606")
                annotation1file = "data/input/datavp2/datavp/gaf/4932.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/9606.gaf"
                network1_ids = np.load(f'data/input/embedding-species/{model_name}/yeast/Yeast_ids.npy')
                network2_ids = np.load(f'data/input/embedding-species/{model_name}/human/human_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/yeast/Yeast_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/human/human_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file, similarities,
                                                                        annotation1file, annotation2file, network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            if (netfile == "data/input/datavp2/datavp/4932.tab") and (
                    netfile2 == "data/input/datavp2/datavp/10090.tab"):
                print("4932-10090")
                annotation1file = "data/input/datavp2/datavp/gaf/4932.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/10090.gaf"
                network1_ids = np.load(f'data/input/embedding-species/{model_name}/yeast/Yeast_ids.npy')
                network2_ids = np.load(f'data/input/embedding-species/{model_name}/mouse/mouse_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/yeast/Yeast_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/mouse/mouse_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file, similarities,
                                                                        annotation1file, annotation2file, network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
            if (netfile == "data/input/datavp2/datavp/9606.tab") and (
                    netfile2 == "data/input/datavp2/datavp/10090.tab"):
                print("9606-10090")
                annotation1file = "data/input/datavp2/datavp/gaf/9606.gaf"
                annotation2file = "data/input/datavp2/datavp/gaf/10090.gaf"
                network1_ids = np.load(f'data/input/embedding-species/{model_name}/human/human_ids.npy')
                network2_ids = np.load(f'data/input/embedding-species/{model_name}/mouse/mouse_ids.npy')
                network1_embd = np.load(
                    f'data/input/embedding-species/{model_name}/human/human_embeddings.npy').astype(
                    np.float32)
                network2_embd = np.load(
                    f'data/input/embedding-species/{model_name}/mouse/mouse_embeddings.npy').astype(
                    np.float32)
                similarities = compute_similarities_between_two_species(netfile, netfile2, output_file, similarities,
                                                                        annotation1file, annotation2file, network1_ids,
                                                                        network2_ids, network1_embd, network2_embd,
                                                                        alpha)
    return similarities


def readnetwork(datafile, similarities, beta=0.5):
    f = open(datafile, 'r')
    lines = f.readlines()
    print("datafile opened.")
    # global k
    C.k = int(lines[0])  # number of networks
    # print(k)
    print("reading PPI networks file......")
    for i in range(1, C.k + 1):
        netfile = lines[i].strip()
        print("opening  {netfile}  ...".format(netfile=netfile))
        try:
            with open(netfile, 'r') as f1:
                nn = 0
                ne = 0
                for j in f1:
                    lineArr = j.strip().split()
                    try:
                        node1 = lineArr[0]
                        node2 = lineArr[1]
                    except:
                        continue
                    if ppi.has_node(node1):
                        node1 = lineArr[0]
                    else:
                        ppi.add_node(node1)
                        sim_ppi.add_node(node1)
                        nn = nn + 1
                    if ppi.has_node(node2):
                        node2 = lineArr[1]
                    else:
                        ppi.add_node(node2)
                        sim_ppi.add_node(node2)
                        nn = nn + 1
                    if not ppi.has_edge(node1, node2):
                        ppi.add_edge(node1, node2)
                        ne = ne + 1
                print("There are {nn} nodes and {ne} edge.".format(nn=nn, ne=ne))
        except FileNotFoundError:
            print("can not find {netfile} ppi network file!\n".format(netfile=netfile))

    print("reading BLAST similiarity file......")
    for i1 in lines[C.k + 1:]:
        netfile = i1.strip()
        print("opening {netfile}. ".format(netfile=netfile))
        try:
            with open(netfile, 'r') as f2:
                for j1 in f2:
                    lineArr = j1.strip().split()
                    sim_node1 = lineArr[0]
                    sim_node2 = lineArr[1]
                    w = float(lineArr[2])
                    test1 = (sim_node1, sim_node2)
                    test2 = (sim_node2, sim_node1)
                    if ppi.has_node(sim_node1):
                        if ppi.has_node(sim_node2):
                            if sim_node1 != sim_node2:
                                if (test1 in similarities):
                                    simtotal = beta * normalize_bit_score(w) + (1 - beta) * similarities[test1]
                                    sim_ppi.add_edge(sim_node1, sim_node2, weight=simtotal)
                                if (test2 in similarities):
                                    simtotal = beta * normalize_bit_score(w) + (1 - beta) * similarities[test2]
                                    sim_ppi.add_edge(sim_node1, sim_node2, weight=simtotal)
                                else:
                                    sim_ppi.add_edge(sim_node1, sim_node2, weight=normalize_bit_score(w))
                f2.close()
        except FileNotFoundError:
            print("can not find {netfile} BLAST similarity file!\n".format(netfile=netfile))
    f.close()
    print("Finished reading BLAST similarities.")
    print("In total, There are {nn} nodes and {ne} edges.".format(nn=sim_ppi.number_of_nodes(),
                                                                  ne=sim_ppi.number_of_edges()))




def CW(clique):
    result = 0.0
    for u in clique:
        for v in clique:
            if sim_ppi.has_edge(u, v):
                result = result + sim_ppi.get_edge_data(u, v)['weight']
    return result / 2




def parse_annotations_first_coronavirus_human(filename, filename2, network):
    annotations = defaultdict(list)
    with open(filename, 'r') as f:
        for line in f:
            try:
                fields = line.strip().split('\t')
                # print(fields)
                if fields[0] in network.nodes():
                    for go_term in fields[5].split(';'):
                        annotations[fields[0]].append(go_term)
            except:
                continue
    with open(filename2, 'r') as f:
        for line in f:
            if line.startswith('!'):
                continue
            fields = line.strip().split('\t')
            if fields[1] in network.nodes() and fields[6] != "IEA":
                for go_term in fields[4].split(';'):
                    annotations[fields[1]].append(go_term)
    # Write annotations to the output file
    with open("data/working/annotation_coronavirus.txt", 'w') as output_f:
        for protein, annotation_list in annotations.items():
            for annt in annotation_list:
                output_f.write(f"{protein}\t{annt}\n")

    return annotations


def parse_annotations_first_coronavirus(filename, network):
    annotations = defaultdict(list)
    with open(filename, 'r') as f:
        for line in f:
            try:
                fields = line.strip().split('\t')
                # print(fields)
                if fields[0] in network.nodes():
                    for go_term in fields[5].split(';'):
                        annotations[fields[0]].append(go_term)
            except:
                continue
    # Write annotations to the output file
    with open("data/working/annotation_coronavirus.txt", 'w') as output_f:
        for protein, annotation_list in annotations.items():
            for annt in annotation_list:
                output_f.write(f"{protein}\t{annt}\n")

    return annotations


def save_annotations_file(filename, annotations):
    with open(filename, 'w') as output_f:
        for protein, annotation_list in annotations.items():
            for annt in annotation_list:
                output_f.write(f"{protein}\t{annt}\n")

    return annotations


def parse_annotations_first(filename, network):
    annotations = defaultdict(list)
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('!'):
                continue
            fields = line.strip().split('\t')
            # print(fields)
            if fields[1] in network.nodes() and fields[6] != "IEA":
                for go_term in fields[4].split(';'):
                    annotations[fields[1]].append(go_term)
    if (filename == "data/input/datavp2/datavp/gaf/7227.gaf"):
        with open("data/working/annotation_fly.txt", 'w') as output_f:
            for protein, annotation_list in annotations.items():
                for annt in annotation_list:
                    output_f.write(f"{protein}\t{annt}\n")

    return annotations


def parse_annotations(filename, network, annotations):
    local_annotations = defaultdict(list)
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('!'):
                continue
            fields = line.strip().split('\t')
            if fields[1] in network.nodes() and fields[6] != "IEA":
                for go_term in fields[4].split(';'):
                    annotations[fields[1]].append(go_term)
                    local_annotations[fields[1]].append(go_term)
    if (filename == "data/input/datavp2/datavp/gaf/9606.gaf"):
        with open("data/working/annotation_human.txt", 'w') as output_f:
            for protein, annotation_list in local_annotations.items():
                for annt in annotation_list:
                    output_f.write(f"{protein}\t{annt}\n")
    if (filename == "data/input/datavp2/datavp/gaf/10090.gaf"):
        with open("data/working/annotation_mouse.txt", 'w') as output_f:
            for protein, annotation_list in local_annotations.items():
                for annt in annotation_list:
                    output_f.write(f"{protein}\t{annt}\n")
    if (filename == "data/input/datavp2/datavp/gaf/6239.gaf"):
        with open("data/working/annotation_worm.txt", 'w') as output_f:
            for protein, annotation_list in local_annotations.items():
                for annt in annotation_list:
                    output_f.write(f"{protein}\t{annt}\n")
    if (filename == "data/input/datavp2/datavp/gaf/4932.gaf"):
        with open("data/working/annotation_yeast.txt", 'w') as output_f:
            for protein, annotation_list in local_annotations.items():
                for annt in annotation_list:
                    output_f.write(f"{protein}\t{annt}\n")

    return annotations


def read_network(network_file):
    edges = []
    with open(network_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # Skip the first line
            fields = line.strip().split('\t')
            if len(fields) == 2:
                edges.append((fields[0], fields[1], {'weight': 1}))
            elif len(fields) == 3:
                edges.append((fields[0], fields[1], {'weight': float(fields[2])}))
            else:
                continue
    return edges


def calculate_node_similarity(node1, node2):
    # Replace this with your actual similarity calculation logic
    tuple1 = (node1, node2)
    tuple2 = (node2, node1)
    if (tuple1 in similarities):
        return similarities[tuple1]
    if (tuple2 in similarities):
        return similarities[tuple2]
    else:
        return 0.0


def save_alignment_to_file(alignment, filename):
    with open(filename, 'w') as file:
        for cluster in alignment:
            # Only write clusters with more than one element
            if len(cluster) > 1:
                for v in cluster:
                    file.write(v + ' ')
                file.write('\n')


def delete_similar_clusters(list_of_lists):
    unique_clusters = set()
    result = []

    for cluster in list_of_lists:
        sorted_cluster = [tuple(sorted(node)) for node in cluster]
        cluster_tuple = tuple(sorted(sorted_cluster))

        if cluster_tuple not in unique_clusters:
            unique_clusters.add(cluster_tuple)
            result.append(cluster)

    return result


def ciq_between_two(c1, c2):
    mutual_k = 0
    n = 0
    kset_t = []
    for nn in c1:
        kname = nn[0:2]
        if not kset_t.__contains__(kname):
            kset_t.append(kname)
    kset2 = []
    for nn in c2:
        if not kset2.__contains__(nn[0:2]) and kset_t.__contains__(nn[0:2]):
            kset2.append(kname)
    mutual_k = len(kset2)
    kset_t.clear()
    # calculate edge weights
    for nn1 in c1:
        for nn2 in c2:
            if ppi.has_edge(nn1, nn2) and nn1[0:2] == nn2[0:2]:
                if not kset_t.__contains__(nn1[0:2]):
                    kset_t.append(nn1[0:2])
                n = n + 1
    e = len(kset_t)
    weight = 0.0
    if e < 2:
        weight = 0.0
    elif mutual_k != 0:
        weight = (e * 1.0) / (mutual_k * 1.0)
    else:
        weight = 0.0
    result = (n * weight, n)
    return result


starttime = time.time()

# datafile = "data/input/policy_coronavirus/new_data_2spec_sars1and2.txt"
datafile = "data/input/data_policy/new_data_5spec.txt"
global beta
global alpha
global ppi
global sim_ppi
global annotations
global clusters
global similarities
output_file = 'data/working/similarities_proteins.txt'
ppi = nx.Graph()
sim_ppi = nx.Graph()
similarities = {}
clusters = {}

resultfile = "data/working/result_all_gen100_mut0.3_pop100_alpha0.0_beta0.5_fitness.txt"
# Suppress specific warning
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

similarities = compute_sim_between_networks(datafile, similarities, output_file, alpha=0.5)
print("End computing similarities")
print(len(similarities))

readnetwork(datafile, similarities)
annotation1file = "data/input/datavp2/datavp/gaf/7227.gaf"
annotation2file = "data/input/datavp2/datavp/gaf/9606.gaf"
annotation3file = "data/input/datavp2/datavp/gaf/10090.gaf"
annotation4file = "data/input/datavp2/datavp/gaf/6239.gaf"
annotation5file = "data/input/datavp2/datavp/gaf/4932.gaf"
annotation6file = "data/input/datavp2/datavp/gaf/562.gaf"
annotation7file = "data/input/datavp2/datavp/gaf/3702.gaf"
annotation8file = "data/input/datavp2/datavp/gaf/4896.gaf"
annotation9file = "data/input/datavp2/datavp/gaf/10116.gaf"
annotation10file = "data/input/datavp2/datavp/gaf/39947.gaf"
annotation_coronavirus = "data/input/sars-goa-uniprotkb/uniprotkb_severe_acute_respiratory_2024_01_01.tsv"
# Read the file content
with open(datafile, 'r') as file:
    lines = file.readlines()

# Filter lines that end with ".tab"
tab_lines = [line.strip() for line in lines if line.strip().endswith(".tab")]

# Extract NCBI identifiers
ncbi_identifiers = [line.split("/")[-1].replace(".tab", "") for line in tab_lines]

# Print the identifiers
print(ncbi_identifiers)
# Parse the GO annotations for species 1 and species 2
speciesCoronavirus = False
if (not speciesCoronavirus):
    annotfilefirst = f"data/input/datavp2/datavp/gaf/{ncbi_identifiers[0]}.gaf"
    print(annotfilefirst)
    annotations = parse_annotations_first(annotfilefirst, sim_ppi)
    print("len_annotation ", ncbi_identifiers[0])
    print(len(annotations))
    for ncbiid in ncbi_identifiers[1:]:
        annotfile = f"data/input/datavp2/datavp/gaf/{ncbiid}.gaf"
        annotations = parse_annotations(annotfile, sim_ppi, annotations)
        print("len_annotation ", ncbiid)
        print(len(annotations))
    save_annotations_file("data/working/annotations_all.txt", annotations)
else:
    print("Start reading Coronavirus annotation")
    annotations = parse_annotations_first_coronavirus(annotation_coronavirus, sim_ppi)
    print("Finish reading Coronavirus annotation", len(annotations))
    annotations = parse_annotations(annotation2file, sim_ppi, annotations)
    print(len(annotations))

#################label_propagation_communities###################
clusters2 = community.label_propagation_communities(sim_ppi)
clusters_list = list(clusters2)
# Convert values to a list of lists
cluster_from_label_propagation = [list(value) for value in clusters_list]

##############louvain trap########
clst = nx.community.louvain_communities(sim_ppi, weight='weight', seed=123)
# print(clst)
cluster_louvain_trap = [list(s) for s in clst]
print("clustering done!")

####################################

g = ig.Graph.TupleList(sim_ppi.edges(), directed=False)
wtrap = g.community_walktrap(steps=4)
clust = wtrap.as_clustering()
# Convert the generator to a list of communities
list_cluster_wtrap = []
for community_idx, cluster in enumerate(clust):
    # Get the list of vertices in the current cluster
    vertices_list = g.vs[cluster]['name']
    list_cluster_wtrap.append(vertices_list)

#################### Modularity maximization################
frozensetmodmax = community.greedy_modularity_communities(sim_ppi, weight='weight')
# Transforming frozensets to lists
lists_list_modmax = [list(frozenset) for frozenset in frozensetmodmax]

###########Best#####Fast Greedy Modularity Optimization###################
import networkx as nx

print("start clustering...")

# Apply Fast Greedy Modularity Optimization on the weighted graph
partition = nx.community.greedy_modularity_communities(sim_ppi, resolution=2, weight='weight')
# Convert frozensets to lists
cluster_greedy_modularity_opt = [list(frozenset_item) for frozenset_item in partition]
print("clustering done!")

def fitness_function_org(solution):
    # This is a generic fitness function example
    aligned_nodes = set()

    for cluster in solution:
        # Add the nodes in each cluster to the set of aligned nodes
        aligned_nodes.update(cluster)

    # Return the negative of the size of the aligned nodes set
    # since genetic algorithms typically aim to maximize a fitness function
    return -len(aligned_nodes)


from itertools import combinations, product


def fitness_function_opt(solution):
    aligned_nodes = set(node for cluster in solution for node in cluster)

    total_similarity = sum(
        calculate_node_similarity(node1, node2) for node1, node2 in product(aligned_nodes, repeat=2) if node1 < node2)

    return -total_similarity  # Negative because genetic algorithms maximize fitness


def parallel_fitness_evaluation(args):
    solution, similarities, result_queue = args
    fitness = fitness_function_org(solution)
    result_queue.put(fitness)


def crossover(parent1, parent2):
    # Single-point crossover
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child


def mutate(solution, mutation_rate, candidate_clusters):
    mutated_solution = []

    for cluster in solution:
        if random.random() > mutation_rate and candidate_clusters:
            mutated_solution.append(random.choice(candidate_clusters))
        else:
            mutated_solution.append(cluster)

    return mutated_solution


def genetic_algorithm(candidate_clusters, generations=100, population_size=100, mutation_rate=0.3, similarities=None):
    # Create the pool outside the loop
    with Pool(processes=cpu_count()) as pool:
        population = [[random.choice(candidate_clusters) for _ in range(len(candidate_clusters))] for _ in
                      range(population_size)]

        for generation in range(generations):
            fitness_scores = []
            with Manager() as manager:
                result_queue = manager.Queue()

                # Evaluate fitness in parallel
                pool.map(parallel_fitness_evaluation,
                         [(solution, similarities, result_queue) for solution in population])

                # Retrieve fitness scores from the result queue
                while not result_queue.empty():
                    fitness_scores.append(result_queue.get())

            parents = [population[i] for i in
                       sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True)[
                       :population_size // 2]]
            offspring = []

            for _ in range(population_size - len(parents)):
                parent1, parent2 = random.sample(parents, 2)
                child = crossover(parent1, parent2)
                child = mutate(child, mutation_rate, candidate_clusters)
                offspring.append(child)

            population = parents + offspring

        best_solution = max(population, key=lambda x: fitness_function_org(x))
        #best_solution = max(population, key=lambda x: fitness_function_opt(x))

    return best_solution

concatenated_list = cluster_greedy_modularity_opt + cluster_from_label_propagation + list_cluster_wtrap + cluster_louvain_trap
print("concatenated_list!", len(concatenated_list))

print("begin genetic algorithm")
best_solution = genetic_algorithm(concatenated_list)

best_solution = delete_similar_clusters(best_solution)
print("end genetic algorithm")

save_alignment_to_file2(best_solution, resultfile)
endtime = time.time()
dtime = endtime - starttime
print(dtime)
