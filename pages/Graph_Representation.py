import streamlit as st

st.markdown("<h1 style='text-align: center;'>Protein-Ligand Graph Representation Documentation</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 24px'>This section contains information on how we create "
            "graphs of the protein-ligand complex before binding affinity prediction with PLAIG. There are three main "
            "steps when generating protein-ligand graphs that we follow: Protein-Pocket Extraction, Feature Extraction, "
            "and Graph Creation. Please read below for an in-depth explanation of the listed steps.</p>", unsafe_allow_html=True)
st.image("pages/Substructure Example.png")
st.markdown("<p style='text-align: center; font-size: 20px'>Example of a Protein Pocket Extraction</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: left; font-size: 20px'><b>Protein-Pocket Extraction:</b> Before creating graphs, "
            "we extract and store the protein binding pocket from the original protein .pdb file. This is to streamline "
            "the binding affinity prediction process, as residues not involved with binding interactions "
            "would unnecessarily increase the computational time of the GNN model. When the ligand and protein .pdb "
            "files are read in, we use RDKit to obtain a list of the specific xyz-coordinates for each atom. Then, we "
            "loop through all the atoms and compute the euclidean distance of every protein atom to every ligand atom. "
            "For every protein atom that is within 10 Å of any ligand atom, we record the corresponding atom's "
            "protein chain ID. The recorded protein chains are the ones used to create the protein binding pocket. "
            "We use the Python package Biopython to parse through the original protein .pdb file write the atoms from "
            "the saved chains to a custom .pdb file. This .pdb file is the protein binding pocket that we extract "
            "features from in the subsequent steps.</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: left; font-size: 20px'><b>Feature Extraction:</b> The first step of feature "
            "extraction loops through all the protein-ligand atom distances and saves the distances that are within "
            "3 Å, the specified cutoff distance. The protein atoms involved in these distances are the ones that we use for computing the interaction "
            "features with BINANA. We also normalize the distances using min-max normalization, as they are stored as "
            "edge features when creating graphs. Next, we compute the protein-ligand interaction features with BINANA. "
            "The specific features that we use are listed and explained in the BINANA section of this webpage. These "
            "features are passed with the list of protein-ligand atom pairs that meet the specified cutoff distance "
            "into our edge feature function. The edge feature function generates a dictionary where each key represents "
            "a BINANA chemical interaction and each value is a list of tuples. Each tuple contains the 3D coordinates "
            "of the protein and ligand atoms involved in that specific interaction. For interactions categorized under "
            "'Electrostatic Energies,' the tuples also include the corresponding electrostatic energy value as a "
            "floating-point number. After the interaction features are stored, we move onto calculating a variety of "
            "topological structural features for the ligand and entire protein pocket. A list of all these features can "
            "be found in the Supporting Information section through the citation on the Home page. We use RDKit to "
            "compute these features and store them into two lists, one for the ligand and one for the protein. When "
            "all features are computed and stored in their respective data structures, we start the graph creation "
            "process.</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: left; font-size: 20px'><b>Graph Creation:</b> We use the Networkx Python library to "
            "create graph structures of each protein-ligand complex. There are three categories of features that we "
            "store inside every graph, atomic-level features, edge-level features, and graph-level features. Starting "
            "with the atomic-level features, we loop through every atom in the ligand and store its atomic number, "
            "residue (one-hot encoded for each amino acid and ligand code), hybridization number, degree, aromaticity, number of hydrogens, number of "
            "radical electrons, mass, formal charge, Gasteiger charge, and xyz-coordinate. Each atom is added as a "
            "node to the graph along with its atomic-level features as a torch tensor. The protein atoms and associated "
            "atomic-level features are stored as nodes in the graph using the same method as the ligand atoms. Next, "
            "the edges and edge features are added to the graph. Our graphs contain edges between ligand atoms and "
            "edges between ligand atoms and protein atoms. However, edges are not stored between protein atoms. Edge "
            "features are stored as torch tensors every time an edge is created between two atoms. There are 11 "
            "components to each edge feature tensor in this order: electrostatic energy, halogen bond, hydrogen bond, "
            "hydrophobic contact, metal contact, π-π stacking, t-stacking, salt bridge, cation-π, "
            "bond type (single, double, triple), and edge distance. For ligand-ligand edges, the first 9 indexes of "
            "the tensor are left as zeros, since interaction features are not computed between ligand atoms. For "
            "protein-ligand edges, the BINANA interaction features are stored in the tensor as either 0 or 1 (1 if "
            "the interaction is present and 0 if the interaction is not present). The only exception is electrostatic "
            "energy, which is stored as the calculated float-type number. Finally, the whole protein and ligand "
            "topological structural features are stored as graph-level features for the overall graph using the "
            "Networkx graph.graph dictionary. After the graph creation process is completed, all features are "
            "normalized using the z-score normalization technique. The finalized graph is returned to the main "
            "algorithm and converted into a Pytorch Data object for use in PLAIG's hybridized GNN model.</p>", unsafe_allow_html=True)
st.image("pages/Graph Representation.png")
st.markdown("<p style='text-align: center; font-size: 20px'>Illustration of Graphs Created for Binding Affinity Prediction</p>", unsafe_allow_html=True)

