# GA2Vec
Global Network Alignment Algorithm

This repository contains Python code for a global network alignment algorithm.
Network alignment is a fundamental task in network analysis that aims 
to find a mapping between the nodes of different networks, maximizing the similarity between aligned nodes while preserving the networks' topological structure.

Written By: Warith Eddine DJEDDI (waritheddine.jeddi@isikef.u-jendouba.tn),
            Sadok BEN YAHIA (sadok.ben@taltech.ee; say@mmmi.sdu.dk) and
            Gayo DIALLO (gayo.diallo@u-bordeaux.fr)

This README describes the usage of the command line interface of GA2Vec. 
The executable GA2Vec is compiled for Linux x86_64 and Windows 64bit platform.


The GA2Vec program identifies a global alignment of multiple input protein-protein interaction networks. 

# Features

    - Computes similarities between the nodes of networks based on network topology and node attributes.
    - Implements community detection algorithms, such as Label Propagation, Louvain, and Modularity Maximization, to identify clusters in networks.
    - Applies genetic algorithms to optimize the alignment by maximizing the similarity between aligned nodes.
    - Supports parallel fitness evaluations for efficient computation.
    
# Requirements

    Python 3.x
    NetworkX
    igraph

# Input
1) Data files containing the paths for the PPI interaction networks and their BLAST bit score files are located in the folder 'data/input/data_policy'
2) The Anc2vec embedding file is in the folder 'data/input/datavp/GOTembed/embeddings.npy'.
    #### Coronavirus and Virus-Host species input data
3) The protein-protein interaction networks of coronavirus and virus-host species, along with their BLAST bit score files, are located under the directory: 'data/input/coronavirus_blast_sim2'
4) The Gene Ontology annotation of coronavirus and Virus-Host species is found in the file: "data/input/sars-goa-uniprotkb". Please initialize "speciesCoronavirus = True" (see Line 1764 in the "GA2Vec.py") in order to parse the GO annotations for coronavirus and Virus-Host species.
5) The pre-trained embedding files for each embedding model (i.e., ProtBERT, T5, and ESM-2) for the coronavirus and Virus-Host species are located in the folder: 'data/input/embedding-species'
    #### Eukaryotic and Prokaryotic species input data
6) The GO association files for the eukaryotic and prokaryotic species from UniProt-GOA are located in the folder 'data/input/datavp2/datavp/gaf'
7) The protein-protein interaction networks of eukaryotic and prokaryotic species, along with their BLAST bit score files, are located in the directory: 'data/input/datavp2/datavp'
8) The pretrained embedding files for each model (i.e., ProtBERT, T5, and ESM-2) for eukaryotic and prokaryotic species are located in the folder: "data/input/embedding-species/"

# Usage
    1. Download the input data from this repository: "https://doi.org/10.5281/zenodo.14054769", and place the folder 'data,' which contains all the necessary files (PPI network, BLAST files, embedding files for each species, gene ontology annotations, etc.), inside the 'GA2Vec' folder.
    1. Set the input data file path (datafile) and output file path (output_file) in the Python script.
    2. Run the Python script: python GA2Vec.py
    3. The aligned nodes will be saved in the specified output file.

# Configuration

    - Adjust parameters such as:
         * alpha (see Line 1588), 
         * beta (see Line 1280), 
         * mutation_rate, population and generations (see Line 1727), 
         * embedding model (see Line 344),
         * Adjust the value of threshold1 (see Line 55)
    - Customize clustering algorithms and similarity measures based on specific requirements.
    - Adjust the strategies (i.e., strategy 1, strategy 2, or strategy 3) to be used in order to reweight the PPI networks (see lines 103, 104, and 105).
    - Adjust the fitness function of the genetic algorithm, try invoking the function 'fitness_function_opt()' (see Line 1760 and Line 1704), which computes the total similarity of aligned nodes. Alternatively, to use the simpler fitness function, call 'fitness_function_org()' (see Line 1759 and Line 1704), which is based on a generic fitness function. This function returns the negative size of the aligned node set, as genetic algorithms typically aim to maximize the fitness value.
# Evaluation
We provide the Python code for generating heatmaps for each compared species, searching for conserved protein complexes, and we also provide the evaluation metrics, including the p-value and F1 score:
1) The file "evaluation/heatmap-p-value/p_value_overall_all_metrics.py" is used to compute the one-sided Wilcoxon p-values and allows visualization of the table values. Each cell in this table indicate the number of pairwise alignments where the performance of a GA2Vec variant significantly exceeds that of a baseline method. 
2) The file "evaluation/heatmap-p-value/p_value_overall_f1_score.py" is used to compute the one-sided Wilcoxon p-values and enables visualization of the p-values between the F1 scores achieved by GA2Vec variants and those obtained by each baseline method.
3) Inside the folder "evaluation/heatmap-p-value," we provide all the Python code used to generate the heatmap for each alignment dataset. 
4) The file "evaluation/search_complex/search_complex.py" is used to generate the triplets (i.e., source, relation, target) for each pair of proteins in order to be visualized using the Cytoscape tool. The source and target are the proteins belonging to the compared species, and the relation refers to the type of interaction (e.g., "aligned_to," "interacts_with"). 
5) Under the "results" folder, we provide all the alignment files for GA2Vec and its competitors.
# License

This project is licensed under the MIT License - see the LICENSE file for details.

# Important Note
The alignment process of GA2Vec leverages GO annotations and pre-trained protein sequence embeddings, including \textit{ProtBERT} [1], \textit{ProtT5-XL-UniRef50} [1], \textit{ESM-2} [2], and the ontological embeddings generated by Anc2Vec [3]. For more details, please refer to our published paper in the IEEE/ACM Transactions on Computational Biology and Bioinformatics (IEEE-TCBB) journal [4].

    References:  
     [1] Elnaggar, A., Heinzinger, M., Dallago, C., Rehawi, G., Wang, Y., Jones, L., ... & Rost, B. (2021). Prottrans: Toward understanding the language of life through self-supervised learning. IEEE transactions on pattern analysis and machine intelligence, 44(10), 7112-7127.
     [2] Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., ... & Rives, A. (2022). Language models of protein sequences at the scale of evolution enable accurate structure prediction. BioRxiv, 2022, 500902.
      [3] Edera, A. A., Milone, D. H., & Stegmayer, G. (2022). Anc2vec: embedding gene ontology terms by preserving ancestors relationships. Briefings in Bioinformatics, 23(2), bbac003.
      [4] Djeddi, W. E., Yahia, S. B., & Nguifo, E. M. (2025). A degree centrality-enhanced computational approach for local network alignment leveraging knowledge graph embeddings. Expert Systems with Applications, 275, 126755.
   





