import time
import pickle
import os
import warnings
import streamlit as st
import PLAIG_Run
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from io import StringIO
from openbabel import openbabel


current_directory = os.getcwd()
st.markdown("<h1 style='text-align: center;'>PLAIG Testing</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 24px'>In this section, you will be able to test PLAIG's binding "
            "affinity predictions on pre-saved data, as well as upload your own files for prediction. There are "
            "three different options to choose from:</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: left; font-size: 20px'>1. General and Refined Set Demo: This demo allows you to "
            "submit files from the PDBbindv.2020 general and refined sets. These protein-ligand complexes are ones "
            "that were used when developing PLAIG, so the model is relatively familiar with these files. Since the "
            "general and refined sets are too large to store on this "
            "webpage, you <b>must click the link in the dropdown</b> to download files and read the directions for the demo. The output from "
            "this dropdown will give the predicted binding affinity and the experimentally-determined binding affinity "
            "for comparison.</p>", unsafe_allow_html=True)
with st.expander("General and Refined Set Demo"):
    general_index_file = "pages/pdb_key_general"
    with open(general_index_file, 'r') as general_file:
        general_text = general_file.read()
        general_text = general_text[general_text.find("3zzf"):]
        general_index_df = pd.read_csv(StringIO(general_text), sep='\s+', header=None, names=["PDB Code", "resolution", "release year", "-logKd/Ki", "Kd/Ki", "slash", "reference", "ligand name"], index_col=False)
        general_index_df = general_index_df[general_index_df["Kd/Ki"].str.contains("Kd|Ki")].reset_index(drop=True)
        general_index_df["-logKd/Ki"] = pd.to_numeric(general_index_df["-logKd/Ki"], errors='coerce')
        print(general_index_df)

    refined_index_file = "pages/pdb_key_refined"
    with open(refined_index_file, 'r') as refined_file:
        refined_text = refined_file.read()
        refined_text = refined_text[refined_text.find("2r58"):]
        refined_index_df = pd.read_csv(StringIO(refined_text), sep='\s+', header=None, names=["PDB Code", "resolution", "release year", "-logKd/Ki", "Kd/Ki", "slash", "reference", "ligand name"], index_col=False)
        refined_index_df["-logKd/Ki"] = pd.to_numeric(refined_index_df["-logKd/Ki"], errors='coerce')
        print(refined_index_df)

    st.markdown(f"<p style='text-align: center; font-size: 16px'>Would you like to demo protein-ligand complexes "
                f"from the PDBbindv.2020 general or refined set?</p>", unsafe_allow_html=True)
    dataset = st.radio("", ["General Set", "Refined Set"])
    st.markdown(f"<p style='text-align: center; font-size: 16px'>You selected the <b>{dataset}</b>.</p>", unsafe_allow_html=True)

    st.markdown('<div style="text-align: center; font-size: 16px"><a href="https://github.com/sivaGU/PLAIG/tree/main/Refined_General_Files" target="_blank">Click here '
                'to download files from the general or refined set for demo testing.</a>', unsafe_allow_html=True)

    st.markdown(f"<p style='text-align: center; font-size: 16px'>Submit general or refined set files in this "
                f"order:<br>1. xxxx_hydrogenated_pocket.pdb<br>2. xxxx_pocket.pdbqt<br>3. "
                f"xxxx_hydrogenated_ligand.pdb<br> 4. xxxx_ligand.pdbqt</p>", unsafe_allow_html=True)

    form1 = st.form(key="Options1")
    complex_files = form1.file_uploader("Choose your general or refined set files", accept_multiple_files=True)
    submitted1 = form1.form_submit_button("Submit Files")
    complex_files_paths = []
    count = 1
    if complex_files:
        if len(complex_files) > 4:
            st.warning("You can only upload up to 4 files.")
        elif all(file.name[:4] != complex_files[0].name[:4] for file in complex_files):
            st.warning("The files must come from the same complex! The files you submitted have different PDB codes.")
        else:
            for file in complex_files:
                form1.write(f"File name: {count}. {file.name}")
                new_file_path = os.path.join(current_directory, file.name)
                with open(new_file_path, 'wb') as f:
                    f.write(file.getbuffer())
                complex_files_paths.append(new_file_path)
                count += 1
            if submitted1:
                pdb_code = complex_files[0].name[:4]
                if dataset == "General Set":
                    try:
                        experimental_log_ba = general_index_df.loc[general_index_df['PDB Code'] == pdb_code, '-logKd/Ki'].iloc[0]
                        experimental_ba = 10 ** (-1 * experimental_log_ba) * (10 ** 6)
                        complex_files_paths = [tuple(complex_files_paths[i:i + 4]) for i in
                                               range(0, len(complex_files_paths), 4)]
                        prediction, graph, color_cutoff = PLAIG_Run.run_model(complex_files_paths)
                        node_colors = ["lightblue" if node < color_cutoff else "pink" for node in graph.nodes()]
                        plt.figure(figsize=(8, 6), dpi=600)
                        nx.draw(graph, with_labels=True, node_color=node_colors, edge_color="black", font_weight="bold",
                                node_size=500, width=3)
                        caption = "Visualization of protein-ligand graph with ligand atoms in blue and protein atoms in pink"
                        plt.figtext(0.5, -0.10, caption, wrap=True, horizontalalignment='center', fontsize=16,
                                    family='sans-serif')
                        st.pyplot(plt)
                        plt.clf()
                        st.markdown(
                            f"<p style='text-align: center; color: red; font-size: 24px'>PDB Code: {pdb_code}<br>{prediction[0]}<br>Experimental Binding Affinity (μM): {round(experimental_ba, 3)}</p>",
                            unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"This PDB code is not available in the {dataset}, please choose the other set.")
                else:
                    try:
                        experimental_log_ba = refined_index_df.loc[refined_index_df['PDB Code'] == pdb_code, '-logKd/Ki'].iloc[0]
                        experimental_ba = 10 ** (-1 * experimental_log_ba) * (10 ** 6)
                        complex_files_paths = [tuple(complex_files_paths[i:i + 4]) for i in
                                               range(0, len(complex_files_paths), 4)]
                        prediction, graph, color_cutoff = PLAIG_Run.run_model(complex_files_paths)
                        node_colors = ["lightblue" if node < color_cutoff else "pink" for node in graph.nodes()]
                        plt.figure(figsize=(8, 6), dpi=600)
                        nx.draw(graph, with_labels=True, node_color=node_colors, edge_color="black", font_weight="bold",
                                node_size=500, width=3)
                        caption = "Visualization of protein-ligand graph with ligand atoms in blue and protein atoms in pink"
                        plt.figtext(0.5, -0.10, caption, wrap=True, horizontalalignment='center', fontsize=16,
                                    family='sans-serif')
                        st.pyplot(plt)
                        plt.clf()
                        st.markdown(
                            f"<p style='text-align: center; color: red; font-size: 24px'>PDB Code: {pdb_code}<br>{prediction[0]}<br>Experimental Binding Affinity (μM): {round(experimental_ba, 3)}</p>",
                            unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"This PDB code is not available in the {dataset}, please choose the other set.")

st.markdown("<p style='text-align: left; font-size: 20px'>2. Pre-Docked Files Demo: Although PLAIG cannot yet predict "
            "the affinity of undocked complexes, its binding affinity prediction accuracy is on par with or better "
            "than existing docking algorithms. This demo showcases PLAIG's ability to predict the binding affinity of "
            "virtually-docked protein-ligand complexes. Inside this dropdown, you can predict the affinity from a list "
            "of receptor-ligand pairs that have been docked with AutoDock Vina. The final predictions will display the "
            "predicted affinity from PLAIG, the predicted affinity from AutoDock Vina "
            "(converted from free energy), and the experimental value.</p>", unsafe_allow_html=True)
with st.expander("Pre-Docked Files Demo"):
    receptor = st.radio("Which receptor would you like to test?", ["Androgen Receptor (AR)", "Chimeric Antigen Receptor (CAR)"])
    affinity_key = {"DHT": [8.65, 8.58], "Flutamide": [7.10, 4.11], "MethylTestosterone": [7.80, 7.77],
                    "R1881": [8.51, 8.51], "Spironolactone": [6.16, 6.16], "Testosterone": [7.80, 7.77],
                    "TolfenamicAcid": [4.33, 4.18], "CINPA1": [7.15, 5.87], "CITCO": [7.31, 6.45],
                    "Clotrimazole": [6.15, 5.57], "PK11195": [6.10, 7.55], "TO901317": [5.66, 7.41]}
    if receptor == "Androgen Receptor (AR)":
        ligand = st.radio(f"Which ligand would you like to test in complex with the {receptor}?", ["DHT", "Flutamide", "Methyl Testosterone", "R1881", "Spironolactone", "Testosterone", "Tolfenamic Acid"])
    else:
        ligand = st.radio(f"Which ligand would you like to test in complex with the {receptor}?", ["CINPA1", "CITCO", "Clotrimazole", "PK11195", "TO901317"])
    submitted2 = st.button("Submit")
    if submitted2:
        receptor_filename = receptor.split(" ")[-1].strip("()")
        ligand_filename = ligand.replace(" ", "")
        receptor_pdb_file = f"pages/example_docked_files/{receptor_filename}/pdb_hydrogenated/{receptor_filename}.pdb"
        receptor_pdbqt_file = f"pages/example_docked_files/{receptor_filename}/pdbqt_hydrogenated/{receptor_filename}.pdbqt"
        ligand_pdb_file = f"pages/example_docked_files/{receptor_filename}/pdb_hydrogenated/{ligand_filename}.pdb"
        ligand_pdbqt_file = f"pages/example_docked_files/{receptor_filename}/pdbqt_hydrogenated/{ligand_filename}.pdbqt"
        vina_affinity = 10 ** (-1 * affinity_key[ligand_filename][1]) * (10 ** 6)
        experimental_affinity = 10 ** (-1 * affinity_key[ligand_filename][0]) * (10 ** 6)
        complex_files_paths = [receptor_pdb_file, receptor_pdbqt_file, ligand_pdb_file, ligand_pdbqt_file]
        complex_files_paths = [tuple(complex_files_paths[i:i + 4]) for i in
                               range(0, len(complex_files_paths), 4)]
        prediction, graph, color_cutoff = PLAIG_Run.run_model(complex_files_paths)
        node_colors = ["lightblue" if node < color_cutoff else "pink" for node in graph.nodes()]
        plt.figure(figsize=(8, 6), dpi=600)
        nx.draw(graph, with_labels=True, node_color=node_colors, edge_color="black", font_weight="bold",
                node_size=500, width=3)
        caption = "Visualization of protein-ligand graph with ligand atoms in blue and protein atoms in pink"
        plt.figtext(0.5, -0.10, caption, wrap=True, horizontalalignment='center', fontsize=16,
                    family='sans-serif')
        st.pyplot(plt)
        plt.clf()
        st.markdown(
            f"<p style='text-align: center; color: red; font-size: 24px'>{ligand} bound to {receptor}<br>{prediction[0]}<br>Binding Affinity from AutoDock Vina (μM): {round(vina_affinity, 3)}<br>Experimental Binding Affinity (μM): {round(experimental_affinity, 3)}</p>",
            unsafe_allow_html=True)

st.markdown("<p style='text-align: left; font-size: 20px'>3. User Testing: Under this dropdown, you will be able to "
            "submit your own <b>docked</b> protein-ligand complex files for binding affinity prediction. In order to "
            "use this tool, you must have at least 2 files, one for the protein (.pdb or .pdbqt) and one for the ligand "
            "(.pdb or .pdbqt). Please follow the steps in the drop down below to clean up the files and submit them "
            "for testing. This tool is useful for determining an "
            "accurate binding affinity prediction "
            "between the protein and ligand if docking has occurred correctly.</p>", unsafe_allow_html=True)
with st.expander("User Testing"):
    st.markdown(f"<p style='text-align: center; font-size: 16px'><b>To clean up files, please follow this process "
                f"before submission:</b></p>", unsafe_allow_html=True)
    st.markdown('<div style="text-align: left; font-size: 16px"><a href="https://ccsb.scripps.edu/mgltools/downloads/" '
                'target="_blank">1. Click here to download the MGLTools package version 1.5.7 <b>Windows</b> setup .exe</a>', unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: left; font-size: 16px'>2. Follow the process of the setup installer to download "
                f"MGLTools. From there, you will need to run the following commands through command prompt to hydrate "
                f"either the .pdb or .pdbqt files that you have.</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 16px'>For the ligand: prepare_ligand - l 'ligand_name'.pdb "
                f"(or .pdbqt) -v -o 'ligand_file'.pdbqt -A hydrogens -U lps -v</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 16px'>For the protein: prepare_receptor -r "
                f"'receptor_name'.pdb (or .pdbqt) -v -o 'receptor_name'.pdbqt -A hydrogens -U lps_waters -v</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: left; font-size: 16px'>3. Once you have your .pdbqt files that have been "
                f"hydrogenated and cleaned up, submit these files down below.</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 16px'>Submit your docked protein and ligand files in this "
                f"order:<br>1. protein_name.pdbqt<br>2. ligand_name.pdbqt</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 16px'><b>Reminder: The protein and ligand files must be "
                f"docked in their proper binding conformation before using this tool.</b></p>", unsafe_allow_html=True)
    form3 = st.form(key="Options3")
    complex_files = form3.file_uploader("Choose your docked protein and ligand files", accept_multiple_files=True)
    submitted3 = form3.form_submit_button("Submit Files")
    complex_files_paths = []
    count = 1
    if complex_files:
        if len(complex_files) > 2:
            st.warning("You can only upload up to 2 files.")
        else:
            if submitted3:
                for file in complex_files:
                    form3.write(f"File name: {count}. {file.name}")
                    new_file_path = os.path.join(current_directory, file.name)
                    with open(new_file_path, 'wb') as f:
                        f.write(file.getbuffer())

                    name = os.path.splitext(os.path.basename(new_file_path))[0]
                    ob_conversion = openbabel.OBConversion()
                    ob_conversion.SetInAndOutFormats("pdbqt", "pdb")
                    mol = openbabel.OBMol()
                    ob_conversion.ReadFile(mol, new_file_path)
                    new_pdb_path = os.path.join(current_directory, f"{name}.pdb")
                    ob_conversion.WriteFile(mol, new_pdb_path)

                    complex_files_paths.append(new_pdb_path)
                    complex_files_paths.append(new_file_path)
                    print(new_pdb_path)
                    print(new_file_path)
                    count += 1

                complex_files_paths = [tuple(complex_files_paths[i:i + 4]) for i in range(0, len(complex_files_paths), 4)]
                prediction, graph, color_cutoff = PLAIG_Run.run_model(complex_files_paths)
                node_colors = ["lightblue" if node < color_cutoff else "pink" for node in graph.nodes()]
                plt.figure(figsize=(8, 6), dpi=600)
                nx.draw(graph, with_labels=True, node_color=node_colors, edge_color="black", font_weight="bold", node_size=500, width=3)
                caption = "Visualization of protein-ligand graph with ligand atoms in blue and protein atoms in pink"
                plt.figtext(0.5, -0.10, caption, wrap=True, horizontalalignment='center', fontsize=16, family='sans-serif')
                st.pyplot(plt)
                plt.clf()
                st.markdown(
                    f"<p style='text-align: center; color: red; font-size: 24px'>{prediction[0]}</p>",
                    unsafe_allow_html=True)




