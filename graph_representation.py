import os
import re
import pandas as pd
import subprocess
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors, rdFreeSASA, Lipinski, Crippen, QED, EState, MolSurf, rdmolfiles, rdchem
import networkx as nx
import torch
import torch_geometric
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from Bio import PDB
from biopandas.pdb import PandasPdb
import numpy as np
import binana
import time
import warnings


def get_ligand_df(ligand_file):
    try:
        atom_df = PandasPdb().read_pdb(ligand_file)
        atom_df = atom_df.get_model(1)
        atom_data = atom_df.df["ATOM"]
        return atom_data.filter(items=["atom_name", "x_coord", "y_coord", "z_coord"])
    except FileNotFoundError:
        print("File does not exist")
        return "DNE"


def get_pocket_df(pocket_file):
    try:
        atom_df = PandasPdb().read_pdb(pocket_file)
        atom_df = atom_df.get_model(1)
        atom_data = atom_df.df["ATOM"]
        hetatm_data = atom_df.df["HETATM"]
        hetatm_data_w_chain_id = hetatm_data.replace("", "X")
        combined_atom_df = pd.concat([atom_data, hetatm_data_w_chain_id], ignore_index=True)
        return combined_atom_df.filter(items=["atom_number", "atom_name", "x_coord", "y_coord", "z_coord"])
    except FileNotFoundError:
        print("File does not exist")
        return "DNE"


def get_electrostatic_energies_edges(electrostatic_energies, ligand_df, pocket_df):
    electrostatic_energy_edges = []

    energies = [electrostatic_energy[2]["energy"] for electrostatic_energy in electrostatic_energies["labels"]]
    mean_energy = np.mean(energies)
    sd_energy = np.std(energies)

    for electrostatic_energy in electrostatic_energies["labels"]:
        ligand_atom = electrostatic_energy[0]
        pocket_atom = electrostatic_energy[1]
        energy = electrostatic_energy[2]["energy"]
        normalized_energy = (energy - mean_energy) / sd_energy

        ligand_atom_string_list = ligand_atom.split(":")
        ligand_atom_name = ligand_atom_string_list[2][:ligand_atom_string_list[2].index('(')]
        ligand_coordinates = (float((ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "x_coord"]).iloc[0]),
                              float((ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "y_coord"]).iloc[0]),
                              float((ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "z_coord"]).iloc[0]))

        pocket_atom_string_list = pocket_atom.split(":")
        pocket_atom_number = pocket_atom_string_list[2][
                             pocket_atom_string_list[2].index('(') + 1: pocket_atom_string_list[2].index(')')]
        pocket_atom_name = pocket_atom_string_list[2][:pocket_atom_string_list[2].index('(')]
        pocket_x_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                    pocket_df["atom_number"] == int(pocket_atom_number)), "x_coord"])
        pocket_y_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                    pocket_df["atom_number"] == int(pocket_atom_number)), "y_coord"])
        pocket_z_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                    pocket_df["atom_number"] == int(pocket_atom_number)), "z_coord"])
        pocket_coordinates = (float(pocket_x_coordinate.iloc[0]), float(pocket_y_coordinate.iloc[0]), float(pocket_z_coordinate.iloc[0]))

        electrostatic_energy_edges.append((ligand_coordinates, pocket_coordinates, normalized_energy))

    return electrostatic_energy_edges


def get_hydrogen_bond_df(hydrogen_bonds, ligand_df, pocket_df):
    hydrogen_bond_df = pd.DataFrame()
    hydrogen_bond_edges = []
    acceptor_list = []
    hydrogen_list = []
    donor_list = []
    for hydrogen_bond in hydrogen_bonds["labels"]:
        donor_location = hydrogen_bond[3]
        if donor_location == "RECEPTOR":
            acceptor = hydrogen_bond[0]
            hydrogen = hydrogen_bond[1]
            donor = hydrogen_bond[2]

            acceptor_string_list = acceptor.split(":")
            acceptor_atom_name = acceptor_string_list[2][:acceptor_string_list[2].index('(')]
            acceptor_list.append(acceptor_atom_name)
            acceptor_coordinates = (float((ligand_df.loc[ligand_df["atom_name"] == acceptor_atom_name, "x_coord"]).iloc[0]), float((ligand_df.loc[(ligand_df["atom_name"] == acceptor_atom_name), "y_coord"]).iloc[0]), float((ligand_df.loc[(ligand_df["atom_name"] == acceptor_atom_name), "z_coord"]).iloc[0]))

            hydrogen_string_list = hydrogen.split(":")
            hydrogen_atom_number = hydrogen_string_list[2][
                                   hydrogen_string_list[2].index('(') + 1: hydrogen_string_list[2].index(')')]
            hydrogen_atom_name = hydrogen_string_list[2][:hydrogen_string_list[2].index('(')]
            hydrogen_list.append((hydrogen_atom_name, hydrogen_atom_number))
            hydrogen_x_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == hydrogen_atom_name) & (
                        pocket_df["atom_number"] == int(hydrogen_atom_number)), "x_coord"])
            hydrogen_y_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == hydrogen_atom_name) & (
                        pocket_df["atom_number"] == int(hydrogen_atom_number)), "y_coord"])
            hydrogen_z_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == hydrogen_atom_name) & (
                        pocket_df["atom_number"] == int(hydrogen_atom_number)), "z_coord"])
            hydrogen_coordinates = (float(hydrogen_x_coordinate.iloc[0]), float(hydrogen_y_coordinate.iloc[0]),
                                    float(hydrogen_z_coordinate.iloc[0]))

            donor_string_list = donor.split(":")
            donor_atom_number = donor_string_list[2][
                                donor_string_list[2].index('(') + 1: donor_string_list[2].index(')')]
            donor_atom_name = donor_string_list[2][:donor_string_list[2].index('(')]
            donor_list.append((donor_atom_name, donor_atom_number))
            donor_x_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == donor_atom_name) & (
                        pocket_df["atom_number"] == int(donor_atom_number)), "x_coord"])
            donor_y_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == donor_atom_name) & (
                        pocket_df["atom_number"] == int(donor_atom_number)), "y_coord"])
            donor_z_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == donor_atom_name) & (
                        pocket_df["atom_number"] == int(donor_atom_number)), "z_coord"])
            donor_coordinates = (
            float(donor_x_coordinate.iloc[0]), float(donor_y_coordinate.iloc[0]), float(donor_z_coordinate.iloc[0]))

            hydrogen_bond_edges.append((acceptor_coordinates, hydrogen_coordinates, donor_coordinates))

        else:
            acceptor = hydrogen_bond[2]
            hydrogen = hydrogen_bond[1]
            donor = hydrogen_bond[0]

            acceptor_string_list = acceptor.split(":")
            acceptor_atom_number = acceptor_string_list[2][
                                   acceptor_string_list[2].index('(') + 1: acceptor_string_list[2].index(')')]
            acceptor_atom_name = acceptor_string_list[2][:acceptor_string_list[2].index('(')]
            acceptor_list.append((acceptor_atom_name, acceptor_atom_number))
            acceptor_x_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == acceptor_atom_name) & (
                        pocket_df["atom_number"] == int(acceptor_atom_number)), "x_coord"])
            acceptor_y_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == acceptor_atom_name) & (
                        pocket_df["atom_number"] == int(acceptor_atom_number)), "y_coord"])
            acceptor_z_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == acceptor_atom_name) & (
                        pocket_df["atom_number"] == int(acceptor_atom_number)), "z_coord"])
            acceptor_coordinates = (float(acceptor_x_coordinate.iloc[0]), float(acceptor_y_coordinate.iloc[0]),
                                    float(acceptor_z_coordinate.iloc[0]))

            hydrogen_string_list = hydrogen.split(":")
            hydrogen_atom_name = hydrogen_string_list[2][:hydrogen_string_list[2].index('(')]
            hydrogen_list.append(hydrogen_atom_name)
            hydrogen_coordinates = (
            float((ligand_df.loc[(ligand_df["atom_name"] == hydrogen_atom_name), "x_coord"]).iloc[0]),
            float((ligand_df.loc[(ligand_df["atom_name"] == hydrogen_atom_name), "y_coord"]).iloc[0]),
            float((ligand_df.loc[(ligand_df["atom_name"] == hydrogen_atom_name), "z_coord"]).iloc[0]))

            donor_string_list = donor.split(":")
            donor_atom_name = donor_string_list[2][:donor_string_list[2].index('(')]
            donor_list.append(donor_atom_name)
            donor_coordinates = (float((ligand_df.loc[(ligand_df["atom_name"] == donor_atom_name), "x_coord"]).iloc[0]),
                                 float((ligand_df.loc[(ligand_df["atom_name"] == donor_atom_name), "y_coord"]).iloc[0]),
                                 float((ligand_df.loc[(ligand_df["atom_name"] == donor_atom_name), "z_coord"]).iloc[0]))

            hydrogen_bond_edges.append((acceptor_coordinates, hydrogen_coordinates, donor_coordinates))

    hydrogen_bond_df["Acceptor"] = acceptor_list
    hydrogen_bond_df["Hydrogen"] = hydrogen_list
    hydrogen_bond_df["Donor"] = donor_list

    return hydrogen_bond_df, hydrogen_bond_edges


def get_halogen_bond_df(halogen_bonds, ligand_df, pocket_df):
    halogen_bond_df = pd.DataFrame()
    halogen_bond_edges = []
    acceptor_list = []
    hydrogen_list = []
    donor_list = []

    for halogen_bond in halogen_bonds["labels"]:
        donor_location = halogen_bond[3]
        if donor_location == "RECEPTOR":
            acceptor = halogen_bond[0]
            hydrogen = halogen_bond[1]
            donor = halogen_bond[2]

            acceptor_string_list = acceptor.split(":")
            acceptor_atom_name = acceptor_string_list[2][:acceptor_string_list[2].index('(')]
            acceptor_list.append(acceptor_atom_name)
            acceptor_coordinates = (
            float((ligand_df.loc[ligand_df["atom_name"] == acceptor_atom_name, "x_coord"]).iloc[0]),
            float((ligand_df.loc[(ligand_df["atom_name"] == acceptor_atom_name), "y_coord"]).iloc[0]),
            float((ligand_df.loc[(ligand_df["atom_name"] == acceptor_atom_name), "z_coord"]).iloc[0]))

            hydrogen_string_list = hydrogen.split(":")
            hydrogen_atom_number = hydrogen_string_list[2][
                                   hydrogen_string_list[2].index('(') + 1: hydrogen_string_list[2].index(')')]
            hydrogen_atom_name = hydrogen_string_list[2][:hydrogen_string_list[2].index('(')]
            hydrogen_list.append((hydrogen_atom_name, hydrogen_atom_number))
            hydrogen_x_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == hydrogen_atom_name) & (
                        pocket_df["atom_number"] == int(hydrogen_atom_number)), "x_coord"])
            hydrogen_y_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == hydrogen_atom_name) & (
                        pocket_df["atom_number"] == int(hydrogen_atom_number)), "y_coord"])
            hydrogen_z_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == hydrogen_atom_name) & (
                        pocket_df["atom_number"] == int(hydrogen_atom_number)), "z_coord"])
            hydrogen_coordinates = (float(hydrogen_x_coordinate.iloc[0]), float(hydrogen_y_coordinate.iloc[0]),
                                    float(hydrogen_z_coordinate.iloc[0]))

            donor_string_list = donor.split(":")
            donor_atom_number = donor_string_list[2][
                                donor_string_list[2].index('(') + 1: donor_string_list[2].index(')')]
            donor_atom_name = donor_string_list[2][:donor_string_list[2].index('(')]
            donor_list.append((donor_atom_name, donor_atom_number))
            donor_x_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == donor_atom_name) & (
                        pocket_df["atom_number"] == int(donor_atom_number)), "x_coord"])
            donor_y_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == donor_atom_name) & (
                        pocket_df["atom_number"] == int(donor_atom_number)), "y_coord"])
            donor_z_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == donor_atom_name) & (
                        pocket_df["atom_number"] == int(donor_atom_number)), "z_coord"])
            donor_coordinates = (
            float(donor_x_coordinate.iloc[0]), float(donor_y_coordinate.iloc[0]), float(donor_z_coordinate.iloc[0]))

            halogen_bond_edges.append((acceptor_coordinates, hydrogen_coordinates, donor_coordinates))

        else:
            acceptor = halogen_bond[2]
            hydrogen = halogen_bond[1]
            donor = halogen_bond[0]

            acceptor_string_list = acceptor.split(":")
            acceptor_atom_number = acceptor_string_list[2][
                                   acceptor_string_list[2].index('(') + 1: acceptor_string_list[2].index(')')]
            acceptor_atom_name = acceptor_string_list[2][:acceptor_string_list[2].index('(')]
            acceptor_list.append((acceptor_atom_name, acceptor_atom_number))
            acceptor_x_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == acceptor_atom_name) & (
                        pocket_df["atom_number"] == int(acceptor_atom_number)), "x_coord"])
            acceptor_y_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == acceptor_atom_name) & (
                        pocket_df["atom_number"] == int(acceptor_atom_number)), "y_coord"])
            acceptor_z_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == acceptor_atom_name) & (
                        pocket_df["atom_number"] == int(acceptor_atom_number)), "z_coord"])
            acceptor_coordinates = (float(acceptor_x_coordinate.iloc[0]), float(acceptor_y_coordinate.iloc[0]),
                                    float(acceptor_z_coordinate.iloc[0]))

            hydrogen_string_list = hydrogen.split(":")
            hydrogen_atom_name = hydrogen_string_list[2][:hydrogen_string_list[2].index('(')]
            hydrogen_list.append(hydrogen_atom_name)
            hydrogen_coordinates = (
            float((ligand_df.loc[(ligand_df["atom_name"] == hydrogen_atom_name), "x_coord"]).iloc[0]),
            float((ligand_df.loc[(ligand_df["atom_name"] == hydrogen_atom_name), "y_coord"]).iloc[0]),
            float((ligand_df.loc[(ligand_df["atom_name"] == hydrogen_atom_name), "z_coord"]).iloc[0]))

            donor_string_list = donor.split(":")
            donor_atom_name = donor_string_list[2][:donor_string_list[2].index('(')]
            donor_list.append(donor_atom_name)
            donor_coordinates = (float((ligand_df.loc[(ligand_df["atom_name"] == donor_atom_name), "x_coord"]).iloc[0]),
                                 float((ligand_df.loc[(ligand_df["atom_name"] == donor_atom_name), "y_coord"]).iloc[0]),
                                 float((ligand_df.loc[(ligand_df["atom_name"] == donor_atom_name), "z_coord"]).iloc[0]))

            halogen_bond_edges.append((acceptor_coordinates, hydrogen_coordinates, donor_coordinates))

    halogen_bond_df["Acceptor"] = acceptor_list
    halogen_bond_df["Hydrogen"] = hydrogen_list
    halogen_bond_df["Donor"] = donor_list

    return halogen_bond_df, halogen_bond_edges


def get_hydrophobics_df(hydrophobic_contacts, ligand_df, pocket_df):
    hydrophobics_df = pd.DataFrame()
    hydrophobics_edges = []
    ligand_atom_list = []
    pocket_atom_list = []

    for hydrophobic_contact in hydrophobic_contacts["labels"]:
        ligand_atom = hydrophobic_contact[0]
        pocket_atom = hydrophobic_contact[1]

        ligand_atom_string_list = ligand_atom.split(":")
        ligand_atom_name = ligand_atom_string_list[2][:ligand_atom_string_list[2].index('(')]
        ligand_atom_list.append(ligand_atom_name)
        ligand_coordinates = (float((ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "x_coord"]).iloc[0]),
                              float((ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "y_coord"]).iloc[0]),
                              float((ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "z_coord"]).iloc[0]))

        pocket_atom_string_list = pocket_atom.split(":")
        pocket_atom_number = pocket_atom_string_list[2][
                             pocket_atom_string_list[2].index('(') + 1: pocket_atom_string_list[2].index(')')]
        pocket_atom_name = pocket_atom_string_list[2][:pocket_atom_string_list[2].index('(')]
        pocket_atom_list.append((pocket_atom_name, pocket_atom_number))
        pocket_x_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                    pocket_df["atom_number"] == int(pocket_atom_number)), "x_coord"])
        pocket_y_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                    pocket_df["atom_number"] == int(pocket_atom_number)), "y_coord"])
        pocket_z_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                    pocket_df["atom_number"] == int(pocket_atom_number)), "z_coord"])
        pocket_coordinates = (float(pocket_x_coordinate.iloc[0]), float(pocket_y_coordinate.iloc[0]), float(pocket_z_coordinate.iloc[0]))

        hydrophobics_edges.append((ligand_coordinates, pocket_coordinates))

    hydrophobics_df["Ligand Atom"] = ligand_atom_list
    hydrophobics_df["Pocket Atom"] = pocket_atom_list
    return hydrophobics_df, hydrophobics_edges


def get_metal_contacts_df(metal_contacts, ligand_df, pocket_df):
    metal_contacts_df = pd.DataFrame()
    metal_contacts_edges = []
    ligand_atom_list = []
    pocket_atom_list = []

    for metal_contact in metal_contacts["labels"]:
        ligand_atom = metal_contact[0]
        pocket_atom = metal_contact[1]

        ligand_atom_string_list = ligand_atom.split(":")
        ligand_atom_name = ligand_atom_string_list[2][:ligand_atom_string_list[2].index('(')]
        ligand_atom_list.append(ligand_atom_name)
        ligand_coordinates = (float((ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "x_coord"]).iloc[0]),
                              float((ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "y_coord"]).iloc[0]),
                              float((ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "z_coord"]).iloc[0]))

        pocket_atom_string_list = pocket_atom.split(":")
        pocket_atom_number = pocket_atom_string_list[2][
                             pocket_atom_string_list[2].index('(') + 1: pocket_atom_string_list[2].index(')')]
        pocket_atom_name = pocket_atom_string_list[2][:pocket_atom_string_list[2].index('(')]
        pocket_atom_list.append((pocket_atom_name, pocket_atom_number))
        pocket_x_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                    pocket_df["atom_number"] == int(pocket_atom_number)), "x_coord"])
        pocket_y_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                    pocket_df["atom_number"] == int(pocket_atom_number)), "y_coord"])
        pocket_z_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                    pocket_df["atom_number"] == int(pocket_atom_number)), "z_coord"])
        pocket_coordinates = (float(pocket_x_coordinate.iloc[0]), float(pocket_y_coordinate.iloc[0]), float(pocket_z_coordinate.iloc[0]))

        metal_contacts_edges.append((ligand_coordinates, pocket_coordinates))

    metal_contacts_df["Ligand Atom"] = ligand_atom_list
    metal_contacts_df["Pocket Atom"] = pocket_atom_list
    return metal_contacts_df, metal_contacts_edges


def get_salt_bridge_df(salt_bridges, ligand_df, pocket_df):
    salt_bridge_df = pd.DataFrame()
    salt_bridge_edges = []
    ligand_atoms_list = []
    pocket_atoms_list = []

    for salt_bridge in salt_bridges["labels"]:
        ligand_atoms = salt_bridge[0].strip("[]").split("/")
        ligand_atom_grouper = []
        ligand_coordinates = []
        for ligand_atom in ligand_atoms:
            ligand_atom = ligand_atom.strip()
            ligand_atom_string_list = ligand_atom.split(":")
            ligand_atom_name = ligand_atom_string_list[2][:ligand_atom_string_list[2].index('(')]
            ligand_coordinates.append((float(
                (ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "x_coord"]).iloc[0]), float(
                (ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "y_coord"]).iloc[0]), float(
                (ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "z_coord"]).iloc[0])))
            ligand_atom_grouper.append(ligand_atom_name)
        ligand_atoms_list.append(tuple(ligand_atom_grouper))

        pocket_atoms = salt_bridge[1].strip("[]").split("/")
        pocket_atom_grouper = []
        pocket_coordinates = []
        for pocket_atom in pocket_atoms:
            pocket_atom = pocket_atom.strip()
            pocket_atom_string_list = pocket_atom.split(":")
            pocket_atom_number = pocket_atom_string_list[2][
                                 pocket_atom_string_list[2].index('(') + 1: pocket_atom_string_list[2].index(')')]
            pocket_atom_name = pocket_atom_string_list[2][:pocket_atom_string_list[2].index('(')]
            pocket_x_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                        pocket_df["atom_number"] == int(pocket_atom_number)), "x_coord"])
            pocket_y_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                        pocket_df["atom_number"] == int(pocket_atom_number)), "y_coord"])
            pocket_z_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                        pocket_df["atom_number"] == int(pocket_atom_number)), "z_coord"])
            pocket_coordinates.append((float(pocket_x_coordinate.iloc[0]), float(pocket_y_coordinate.iloc[0]),
                                       float(pocket_z_coordinate.iloc[0])))
            pocket_atom_grouper.append((pocket_atom_name, pocket_atom_number))
        pocket_atoms_list.append(tuple(pocket_atom_grouper))
        salt_bridge_edges.append((ligand_coordinates, pocket_coordinates))

    salt_bridge_df["Ligand Atoms"] = ligand_atoms_list
    salt_bridge_df["Pocket Atoms"] = pocket_atoms_list

    return salt_bridge_df, salt_bridge_edges


def get_pi_pi_t_stacking_df(pi_pi_stackings, ligand_df, pocket_df):
    pi_pi_stacking_df = pd.DataFrame()
    ligand_atoms_list = []
    pocket_atoms_list = []
    pi_pi_edges = []

    for pi_pi_stacking in pi_pi_stackings["labels"]["pi_stacking"]:
        ligand_atoms = pi_pi_stacking[0].strip("[]").split("/")
        ligand_atom_grouper = []
        ligand_coordinates = []
        for ligand_atom in ligand_atoms:
            ligand_atom = ligand_atom.strip()
            ligand_atom_string_list = ligand_atom.split(":")
            ligand_atom_name = ligand_atom_string_list[2][:ligand_atom_string_list[2].index('(')]
            ligand_coordinates.append((float(
                (ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "x_coord"]).iloc[0]), float(
                (ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "y_coord"]).iloc[0]), float(
                (ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "z_coord"]).iloc[0])))
            ligand_atom_grouper.append(ligand_atom_name)
        ligand_atoms_list.append(tuple(ligand_atom_grouper))

        pocket_atoms = pi_pi_stacking[1].strip("[]").split("/")
        pocket_atom_grouper = []
        pocket_coordinates = []
        for pocket_atom in pocket_atoms:
            pocket_atom = pocket_atom.strip()
            pocket_atom_string_list = pocket_atom.split(":")
            pocket_atom_number = pocket_atom_string_list[2][
                                 pocket_atom_string_list[2].index('(') + 1: pocket_atom_string_list[2].index(')')]
            pocket_atom_name = pocket_atom_string_list[2][:pocket_atom_string_list[2].index('(')]
            pocket_x_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                        pocket_df["atom_number"] == int(pocket_atom_number)), "x_coord"])
            pocket_y_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                        pocket_df["atom_number"] == int(pocket_atom_number)), "y_coord"])
            pocket_z_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                        pocket_df["atom_number"] == int(pocket_atom_number)), "z_coord"])
            pocket_coordinates.append((float(pocket_x_coordinate.iloc[0]), float(pocket_y_coordinate.iloc[0]),
                                       float(pocket_z_coordinate.iloc[0])))
            pocket_atom_grouper.append((pocket_atom_name, pocket_atom_number))
        pocket_atoms_list.append(tuple(pocket_atom_grouper))
        pi_pi_edges.append((ligand_coordinates, pocket_coordinates))

    pi_pi_stacking_df["Ligand Atoms"] = ligand_atoms_list
    pi_pi_stacking_df["Pocket Atoms"] = pocket_atoms_list

    t_stacking_df = pd.DataFrame()
    ligand_atoms_list = []
    pocket_atoms_list = []
    t_edges = []

    for t_stacking in pi_pi_stackings["labels"]["T_stacking"]:
        ligand_atoms = t_stacking[0].strip("[]").split("/")
        ligand_atom_grouper = []
        ligand_coordinates = []
        for ligand_atom in ligand_atoms:
            ligand_atom = ligand_atom.strip()
            ligand_atom_string_list = ligand_atom.split(":")
            ligand_atom_name = ligand_atom_string_list[2][:ligand_atom_string_list[2].index('(')]
            ligand_coordinates.append((float(
                (ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "x_coord"]).iloc[0]), float(
                (ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "y_coord"]).iloc[0]), float(
                (ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "z_coord"]).iloc[0])))
            ligand_atom_grouper.append(ligand_atom_name)
        ligand_atoms_list.append(tuple(ligand_atom_grouper))

        pocket_atoms = t_stacking[1].strip("[]").split("/")
        pocket_atom_grouper = []
        pocket_coordinates = []
        for pocket_atom in pocket_atoms:
            pocket_atom = pocket_atom.strip()
            pocket_atom_string_list = pocket_atom.split(":")
            pocket_atom_number = pocket_atom_string_list[2][
                                 pocket_atom_string_list[2].index('(') + 1: pocket_atom_string_list[2].index(')')]
            pocket_atom_name = pocket_atom_string_list[2][:pocket_atom_string_list[2].index('(')]
            pocket_x_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                        pocket_df["atom_number"] == int(pocket_atom_number)), "x_coord"])
            pocket_y_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                        pocket_df["atom_number"] == int(pocket_atom_number)), "y_coord"])
            pocket_z_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                        pocket_df["atom_number"] == int(pocket_atom_number)), "z_coord"])
            pocket_coordinates.append((float(pocket_x_coordinate.iloc[0]), float(pocket_y_coordinate.iloc[0]),
                                       float(pocket_z_coordinate.iloc[0])))
            pocket_atom_grouper.append((pocket_atom_name, pocket_atom_number))
        pocket_atoms_list.append(tuple(pocket_atom_grouper))
        t_edges.append((ligand_coordinates, pocket_coordinates))

    t_stacking_df["Ligand Atoms"] = ligand_atoms_list
    t_stacking_df["Pocket Atoms"] = pocket_atoms_list

    return pi_pi_stacking_df, t_stacking_df, pi_pi_edges, t_edges


def get_cation_pi_df(cation_pis, ligand_df, pocket_df):
    cation_pi_df = pd.DataFrame()
    ligand_atoms_list = []
    pocket_atoms_list = []
    cation_pi_edges = []

    for cation_pi in cation_pis["labels"]:
        ligand_atoms = cation_pi[0].strip("[]").split("/")
        ligand_atom_grouper = []
        ligand_coordinates = []
        for ligand_atom in ligand_atoms:
            ligand_atom = ligand_atom.strip()
            ligand_atom_string_list = ligand_atom.split(":")
            ligand_atom_name = ligand_atom_string_list[2][:ligand_atom_string_list[2].index('(')]
            ligand_coordinates.append((float(
                (ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "x_coord"]).iloc[0]), float(
                (ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "y_coord"]).iloc[0]), float(
                (ligand_df.loc[(ligand_df["atom_name"] == ligand_atom_name), "z_coord"]).iloc[0])))
            ligand_atom_grouper.append(ligand_atom_name)
        ligand_atoms_list.append(tuple(ligand_atom_grouper))

        pocket_atoms = cation_pi[1].strip("[]").split("/")
        pocket_atom_grouper = []
        pocket_coordinates = []
        for pocket_atom in pocket_atoms:
            pocket_atom = pocket_atom.strip()
            pocket_atom_string_list = pocket_atom.split(":")
            pocket_atom_number = pocket_atom_string_list[2][
                                 pocket_atom_string_list[2].index('(') + 1: pocket_atom_string_list[2].index(')')]
            pocket_atom_name = pocket_atom_string_list[2][:pocket_atom_string_list[2].index('(')]
            pocket_x_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                        pocket_df["atom_number"] == int(pocket_atom_number)), "x_coord"])
            pocket_y_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                        pocket_df["atom_number"] == int(pocket_atom_number)), "y_coord"])
            pocket_z_coordinate = (pocket_df.loc[(pocket_df["atom_name"] == pocket_atom_name) & (
                        pocket_df["atom_number"] == int(pocket_atom_number)), "z_coord"])
            pocket_coordinates.append((float(pocket_x_coordinate.iloc[0]), float(pocket_y_coordinate.iloc[0]),
                                       float(pocket_z_coordinate.iloc[0])))
            pocket_atom_grouper.append((pocket_atom_name, pocket_atom_number))
        pocket_atoms_list.append(tuple(pocket_atom_grouper))
        cation_pi_edges.append((ligand_coordinates, pocket_coordinates))

    cation_pi_df["Ligand Atoms"] = ligand_atoms_list
    cation_pi_df["Pocket Atoms"] = pocket_atoms_list
    return cation_pi_df, cation_pi_edges


def get_binana_features(p_path, l_path, ligand_df, pocket_df, cutoff):
    ligand, pocket = binana.load_ligand_receptor.from_files(l_path, p_path)
    electrostatic_energies = binana.interactions.get_electrostatic_energies(ligand, pocket, cutoff=cutoff)
    electrostatic_energy_edges = get_electrostatic_energies_edges(electrostatic_energies, ligand_df, pocket_df)
    halogen_bonds = binana.interactions.get_halogen_bonds(ligand, pocket, dist_cutoff=cutoff)
    halogen_bond_dataframe, halogen_bond_edges = get_halogen_bond_df(halogen_bonds, ligand_df, pocket_df)
    hydrogen_bonds = binana.interactions.get_hydrogen_bonds(ligand, pocket, dist_cutoff=cutoff)
    hydrogen_bond_dataframe, hydrogen_bond_edges = get_hydrogen_bond_df(hydrogen_bonds, ligand_df, pocket_df)
    hydrophobic_contacts = binana.interactions.get_hydrophobics(ligand, pocket, cutoff=cutoff)
    hydrophobic_contacts_dataframe, hydrophobic_contacts_edges = get_hydrophobics_df(hydrophobic_contacts, ligand_df,
                                                                                     pocket_df)
    metal_contacts = binana.interactions.get_metal_coordinations(ligand, pocket, cutoff=cutoff)
    metal_contacts_dataframe, metal_contacts_edges = get_metal_contacts_df(metal_contacts, ligand_df, pocket_df)
    pi_pi_t_stackings = binana.interactions.get_all_interactions(ligand, pocket, pi_pi_general_dist_cutoff=cutoff,
                                                                 t_stacking_closest_dist_cutoff=cutoff)["pi_pi"]
    pi_pi_dataframe, t_dataframe, pi_pi_edges, t_edges = get_pi_pi_t_stacking_df(pi_pi_t_stackings, ligand_df,
                                                                                 pocket_df)
    salt_bridges = binana.interactions.get_salt_bridges(ligand, pocket, cutoff=cutoff)
    salt_bridge_dataframe, salt_bridge_edges = get_salt_bridge_df(salt_bridges, ligand_df, pocket_df)
    cation_pis = binana.interactions.get_cation_pi(ligand, pocket, cutoff=cutoff)
    cation_pi_dataframe, cation_pi_edges = get_cation_pi_df(cation_pis, ligand_df, pocket_df)

    return halogen_bond_dataframe, hydrogen_bond_dataframe, hydrophobic_contacts_dataframe, metal_contacts_dataframe, pi_pi_dataframe, t_dataframe, salt_bridge_dataframe, cation_pi_dataframe, electrostatic_energy_edges, halogen_bond_edges, hydrogen_bond_edges, hydrophobic_contacts_edges, metal_contacts_edges, pi_pi_edges, t_edges, salt_bridge_edges, cation_pi_edges


def vectorize_interaction_node_features(binana_features, pocket_df, ligand_df):
    hab_df = binana_features[0]
    hyb_df = binana_features[1]
    hypc_df = binana_features[2]
    mc_df = binana_features[3]
    pp_df = binana_features[4]
    t_df = binana_features[5]
    sb_df = binana_features[6]
    cp_df = binana_features[7]

    ligand_atom_features = {}
    pocket_atom_features = {}

    for l_row in ligand_df.itertuples(index=True):
        l_atom = l_row.atom_name
        l_coordinates = (float(l_row.x_coord), float(l_row.y_coord), float(l_row.z_coord))
        l_features = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for item in hab_df["Acceptor"].values:
            if isinstance(item, str):
                if l_atom == item:
                    l_features[1] = 1
        for item in hab_df["Hydrogen"].values:
            if isinstance(item, str):
                if l_atom == item:
                    l_features[2] = 1
        for item in hab_df["Donor"].values:
            if isinstance(item, str):
                if l_atom == item:
                    l_features[3] = 1
        for item in hyb_df["Acceptor"].values:
            if isinstance(item, str):
                if l_atom == item:
                    l_features[4] = 1
        for item in hyb_df["Hydrogen"].values:
            if isinstance(item, str):
                if l_atom == item:
                    l_features[5] = 1
        for item in hyb_df["Donor"].values:
            if isinstance(item, str):
                if l_atom == item:
                    l_features[6] = 1
        if l_atom in hypc_df["Ligand Atom"].values:
            l_features[7] = 1
        if l_atom in mc_df["Ligand Atom"].values:
            l_features[8] = 1
        for item in pp_df["Ligand Atoms"]:
            if l_atom in item:
                l_features[9] = 1
                break
        for item in t_df["Ligand Atoms"]:
            if l_atom in item:
                l_features[10] = 1
                break
        for item in sb_df["Ligand Atoms"]:
            if l_atom in item:
                l_features[11] = 1
                break
        for item in cp_df["Ligand Atoms"]:
            if l_atom in item:
                l_features[12] = 1
                break
        ligand_atom_features[l_coordinates] = l_features

    for p_row in pocket_df.itertuples(index=True):
        p_atom_name = p_row.atom_name
        p_atom_number = p_row.atom_number
        p_atom_tuple = (p_atom_name, p_atom_number)
        p_coordinates = (float(p_row.x_coord), float(p_row.y_coord), float(p_row.z_coord))
        p_features = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for item in hab_df["Acceptor"].values:
            if isinstance(item, str):
                if p_atom_tuple == item:
                    p_features[1] = 1
        for item in hab_df["Hydrogen"].values:
            if not isinstance(item, str):
                if p_atom_tuple == item:
                    p_features[2] = 1
        for item in hab_df["Donor"].values:
            if not isinstance(item, str):
                if p_atom_tuple == item:
                    p_features[3] = 1
        for item in hyb_df["Acceptor"].values:
            if not isinstance(item, str):
                if p_atom_tuple == item:
                    p_features[4] = 1
        for item in hyb_df["Hydrogen"].values:
            if not isinstance(item, str):
                if p_atom_tuple == item:
                    p_features[5] = 1
        for item in hyb_df["Donor"].values:
            if not isinstance(item, str):
                if p_atom_tuple == item:
                    p_features[6] = 1
        if p_atom_tuple in hypc_df["Pocket Atom"].values:
            p_features[7] = 1
        if p_atom_tuple in mc_df["Pocket Atom"].values:
            p_features[8] = 1
        for item in pp_df["Pocket Atoms"]:
            if p_atom_tuple in item:
                p_features[9] = 1
                break
        for item in t_df["Pocket Atoms"]:
            if p_atom_tuple in item:
                p_features[10] = 1
                break
        for item in sb_df["Pocket Atoms"]:
            if p_atom_tuple in item:
                p_features[11] = 1
                break
        for item in cp_df["Pocket Atoms"]:
            if p_atom_tuple in item:
                p_features[12] = 1
                break
        pocket_atom_features[p_coordinates] = p_features

    return ligand_atom_features, pocket_atom_features


def get_edge_features(binana_edge_features, close_pairs):
    ee_edges = binana_edge_features[0]
    hab_edges = binana_edge_features[1]
    hyb_edges = binana_edge_features[2]
    hypc_edges = binana_edge_features[3]
    mc_edges = binana_edge_features[4]
    pp_edges = binana_edge_features[5]
    t_edges = binana_edge_features[6]
    sb_edges = binana_edge_features[7]
    cp_edges = binana_edge_features[8]

    feature_edge_dict = {"Electrostatic Energies": [], "Halogen Bonds": [], "Hydrogen Bonds": [], "Hydrophobic Contacts": [], "Metal Contacts": [],
                         "Pi-Pi Stacking": [], "T-stacking": [], "Salt Bridges": [], "Cation-Pi Interactions": []}

    for item in close_pairs:
        ligand_atom_coordinates = tuple(item[2])
        # print(ligand_atom_coordinates)
        pocket_atom_coordinates = tuple(item[3])
        # print(pocket_atom_coordinates)

        for ligand_coordinates, pocket_coordinates, energy in ee_edges:
            if ligand_coordinates == ligand_atom_coordinates and pocket_coordinates == pocket_atom_coordinates:
                feature_edge_dict["Electrostatic Energies"].append((ligand_atom_coordinates, pocket_atom_coordinates, energy))
                break

        for acceptor_coordinates, hydrogen_coordinates, donor_coordinates in hab_edges:
            if ligand_atom_coordinates == acceptor_coordinates:
                if pocket_atom_coordinates == donor_coordinates:
                    feature_edge_dict["Halogen Bonds"].append((ligand_atom_coordinates, pocket_atom_coordinates))
                    break
            if pocket_atom_coordinates == acceptor_coordinates:
                if ligand_atom_coordinates == donor_coordinates:
                    feature_edge_dict["Halogen Bonds"].append((ligand_atom_coordinates, pocket_atom_coordinates))
                    break

        for acceptor_coordinates, hydrogen_coordinates, donor_coordinates in hyb_edges:
            # print(acceptor_coordinates)
            # print(donor_coordinates)
            if ligand_atom_coordinates == acceptor_coordinates:
                if pocket_atom_coordinates == donor_coordinates:
                    feature_edge_dict["Hydrogen Bonds"].append((ligand_atom_coordinates, pocket_atom_coordinates))
                    break
            if pocket_atom_coordinates == acceptor_coordinates:
                if ligand_atom_coordinates == donor_coordinates:
                    feature_edge_dict["Hydrogen Bonds"].append((ligand_atom_coordinates, pocket_atom_coordinates))
                    break

        for ligand_coordinates, pocket_coordinates in hypc_edges:
            if ligand_coordinates == ligand_atom_coordinates and pocket_coordinates == pocket_atom_coordinates:
                feature_edge_dict["Hydrophobic Contacts"].append((ligand_atom_coordinates, pocket_atom_coordinates))
                break

        for ligand_coordinates, pocket_coordinates in mc_edges:
            if ligand_coordinates == ligand_atom_coordinates and pocket_coordinates == pocket_atom_coordinates:
                feature_edge_dict["Metal Contacts"].append((ligand_atom_coordinates, pocket_atom_coordinates))
                break

        ligand_boolean = False
        pocket_boolean = False
        for ligand_coordinate_list, pocket_coordinate_list in pp_edges:
            for ligand_coordinates in ligand_coordinate_list:
                if ligand_coordinates == ligand_atom_coordinates:
                    ligand_boolean = True
                    break
            for pocket_coordinates in pocket_coordinate_list:
                if pocket_coordinates == pocket_atom_coordinates:
                    pocket_boolean = True
                    break
        if ligand_boolean and pocket_boolean:
            feature_edge_dict["Pi-Pi Stacking"].append((ligand_atom_coordinates, pocket_atom_coordinates))

        ligand_boolean = False
        pocket_boolean = False
        for ligand_coordinate_list, pocket_coordinate_list in t_edges:
            for ligand_coordinates in ligand_coordinate_list:
                if ligand_coordinates == ligand_atom_coordinates:
                    ligand_boolean = True
                    break
            for pocket_coordinates in pocket_coordinate_list:
                if pocket_coordinates == pocket_atom_coordinates:
                    pocket_boolean = True
                    break
        if ligand_boolean and pocket_boolean:
            feature_edge_dict["T-stacking"].append((ligand_atom_coordinates, pocket_atom_coordinates))

        ligand_boolean = False
        pocket_boolean = False
        for ligand_coordinate_list, pocket_coordinate_list in sb_edges:
            for ligand_coordinates in ligand_coordinate_list:
                if ligand_coordinates == ligand_atom_coordinates:
                    ligand_boolean = True
                    break
            for pocket_coordinates in pocket_coordinate_list:
                if pocket_coordinates == pocket_atom_coordinates:
                    pocket_boolean = True
                    break
        if ligand_boolean and pocket_boolean:
            feature_edge_dict["Salt Bridges"].append((ligand_atom_coordinates, pocket_atom_coordinates))

        ligand_boolean = False
        pocket_boolean = False
        for ligand_coordinate_list, pocket_coordinate_list in cp_edges:
            for ligand_coordinates in ligand_coordinate_list:
                if ligand_coordinates == ligand_atom_coordinates:
                    ligand_boolean = True
                    break
            for pocket_coordinates in pocket_coordinate_list:
                if pocket_coordinates == pocket_atom_coordinates:
                    pocket_boolean = True
                    break
        if ligand_boolean and pocket_boolean:
            feature_edge_dict["Cation-Pi Interactions"].append((ligand_atom_coordinates, pocket_atom_coordinates))

    return feature_edge_dict


def get_residue_number(atom):
    name = atom.GetPDBResidueInfo().GetResidueName()
    if name == "GLY":
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if name == "ALA":
        return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if name == "VAL":
        return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if name == "LEU":
        return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if name == "ILE":
        return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if name == "THR":
        return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if name == "SER":
        return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if name == "MET":
        return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if name == "CYS":
        return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if name == "PRO":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if name == "PHE":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if name == "TYR":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if name == "TRP":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    if name == "HIS":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    if name == "LYS":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    if name == "ARG":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    if name == "ASP":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    if name == "GLU":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    if name == "ASN":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    if name == "GLN":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


def get_hybridization_number(atom):
    hybridization = atom.GetHybridization()
    if hybridization == Chem.rdchem.HybridizationType.S:
        return [1, 0, 0, 0, 0, 0, 0, 0]
    if hybridization == Chem.rdchem.HybridizationType.SP:
        return [0, 1, 0, 0, 0, 0, 0, 0]
    if hybridization == Chem.rdchem.HybridizationType.SP2:
        return [0, 0, 1, 0, 0, 0, 0, 0]
    if hybridization == Chem.rdchem.HybridizationType.SP2D:
        return [0, 0, 0, 1, 0, 0, 0, 0]
    if hybridization == Chem.rdchem.HybridizationType.SP3:
        return [0, 0, 0, 0, 1, 0, 0, 0]
    if hybridization == Chem.rdchem.HybridizationType.SP3D:
        return [0, 0, 0, 0, 0, 1, 0, 0]
    if hybridization == Chem.rdchem.HybridizationType.SP3D2:
        return [0, 0, 0, 0, 0, 0, 1, 0]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 1]


def get_gasteiger_charge(atom):
    charge = atom.GetDoubleProp('_GasteigerCharge')
    # print(charge)
    # print(type(charge))
    if str(charge) == "nan" or str(charge) == "inf":
        # print("hello")
        return 0
    return charge


def create_submol(pocket_file, chains_residues_to_include, output_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pocket_file)

    io = PDB.PDBIO()
    io.set_structure(structure)

    class CustomSelect(PDB.Select):
        def __init__(self, chains_residues):
            self.chains_residues = chains_residues

        def accept_chain(self, chain):
            # Check if chain is in the chains_residues dictionary
            return chain.get_id() in self.chains_residues

        def accept_residue(self, residue):
            chain_id = residue.get_parent().get_id()  # Get chain ID from residue parent (chain)
            residue_number = residue.get_id()[1]  # Get residue number

            # Check if chain_id is in the chains_residues dictionary and residue_number is included for that chain
            if chain_id in self.chains_residues:
                return residue_number in self.chains_residues[chain_id]
            return False

        def accept_atom(self, atom):
            return True

    select = CustomSelect(chains_residues_to_include)

    io.save(output_file, select)


def pl_complex_to_graph(pocket_file, ligand_file, pocket_pdbqt_file, ligand_pdbqt_file, threshold):
    # Creating a submol for the protein pocket
    ligand = Chem.MolFromPDBFile(ligand_file)
    ligand_conf = ligand.GetConformer()
    ligand_positions = ligand_conf.GetPositions()
    AllChem.ComputeGasteigerCharges(ligand)
    num_ligand_atoms = ligand.GetNumAtoms()

    pocket = Chem.MolFromPDBFile(pocket_file)
    pocket_conf = pocket.GetConformer()
    pocket_positions = pocket_conf.GetPositions()
    pocket_atom_indices = []
    for ligand_index, latom_position in enumerate(ligand_positions):
        for pocket_index, patom_position in enumerate(pocket_positions):
            distance = np.linalg.norm(latom_position - patom_position)
            if distance <= 10:
                pocket_atom_indices.append(pocket_index)
    unique_pocket_atom_indices = list(set(pocket_atom_indices))
    chain_residues = {}
    for index in unique_pocket_atom_indices:
        atom = pocket.GetAtomWithIdx(index)
        chain_id = atom.GetPDBResidueInfo().GetChainId()
        residue_number = atom.GetPDBResidueInfo().GetResidueNumber()
        if chain_id not in chain_residues.keys():
            chain_residues[chain_id] = [residue_number]
        else:
            if residue_number in chain_residues[chain_id]:
                continue
            else:
                chain_residues[chain_id].append(residue_number)
    protein_name = os.path.splitext(os.path.basename(pocket_file))[0]
    print(protein_name)
    print(chain_residues)
    submol_directory = os.path.join(os.getcwd(), "Protein_Pockets")
    os.makedirs(submol_directory, exist_ok=True)
    submol_file = os.path.join(submol_directory, f"{protein_name}_submol.pdb")
    create_submol(pocket_file, chain_residues, submol_file)
    print("Done creating submol")

    pocket = Chem.MolFromPDBFile(submol_file)
    pocket_conf = pocket.GetConformer()
    pocket_positions = pocket_conf.GetPositions()

    AllChem.ComputeGasteigerCharges(pocket)

    all_positions = np.concatenate((ligand_positions, pocket_positions), axis=0)
    mean_coords = np.mean(all_positions, axis=0)
    std_coords = np.std(all_positions, axis=0)

    # Finding normalized bond distances
    distances = []
    for bond in ligand.GetBonds():
        start_atom = bond.GetBeginAtomIdx()
        start_atom_coordinates = ligand_positions[start_atom]
        end_atom = bond.GetEndAtomIdx()
        end_atom_coordinates = ligand_positions[end_atom]
        distance = np.linalg.norm(end_atom_coordinates - start_atom_coordinates)
        distances.append(distance)

    l_count = 0
    for latom_position in ligand_positions:
        p_count = 0
        for patom_position in pocket_positions:
            difference = latom_position - patom_position
            distance = np.linalg.norm(difference)
            if distance <= threshold:
                distances.append(distance)
            p_count += 1
        l_count += 1

    min_distance = min(distances)
    max_distance = max(distances)

    # Finding close protein-ligand atom pairs within the specific distance threshold
    close_pairs = []
    l_count = 0
    for latom_position in ligand_positions:
        p_count = 0
        for patom_position in pocket_positions:
            difference = latom_position - patom_position
            distance = np.linalg.norm(difference)
            if distance <= threshold:
                normalized_distance = (distance - min_distance) / (max_distance - min_distance)
                close_pairs.append((l_count, p_count, latom_position, patom_position, normalized_distance))
            p_count += 1
        l_count += 1
    print(close_pairs)

    #Obtaining gasteiger charges
    gasteiger_charges = []
    masses = []
    for atom in ligand.GetAtoms():
        gasteiger_charges.append(get_gasteiger_charge(atom))
        masses.append(atom.GetMass())
    for atom in pocket.GetAtoms():
        gasteiger_charges.append(get_gasteiger_charge(atom))
        masses.append(atom.GetMass())
    gasteiger_mean = np.mean(gasteiger_charges)
    gasteiger_sd = np.std(gasteiger_charges)
    masses_mean = np.mean(masses)
    masses_sd = np.std(masses)

    # Getting dataframe of ligand atom coordinates
    ligand_df = get_ligand_df(ligand_file)
    # Getting dataframe of pocket atom coordinates
    pocket_df = get_pocket_df(pocket_file)

    binana_features = get_binana_features(pocket_pdbqt_file, ligand_pdbqt_file, ligand_df, pocket_df, threshold)

    edge_features_dict = get_edge_features(binana_features[8:], close_pairs)

    complex_graph = nx.Graph()
    # Calculating ligand topological and structural features
    ligand_logp = Descriptors.MolLogP(ligand)
    ligand_tpsa = rdMolDescriptors.CalcTPSA(ligand)
    ligand_asphericity = rdMolDescriptors.CalcAsphericity(ligand)
    ligand_chi0n = rdMolDescriptors.CalcChi0n(ligand)
    ligand_chi0v = rdMolDescriptors.CalcChi0v(ligand)
    ligand_chi1n = rdMolDescriptors.CalcChi1n(ligand)
    ligand_chi1v = rdMolDescriptors.CalcChi1v(ligand)
    ligand_chi2n = rdMolDescriptors.CalcChi2n(ligand)
    ligand_chi2v = rdMolDescriptors.CalcChi2v(ligand)
    ligand_chi3n = rdMolDescriptors.CalcChi3n(ligand)
    ligand_chi3v = rdMolDescriptors.CalcChi3v(ligand)
    ligand_chi4n = rdMolDescriptors.CalcChi4n(ligand)
    ligand_chi4v = rdMolDescriptors.CalcChi4v(ligand)
    ligand_eccentricity = rdMolDescriptors.CalcEccentricity(ligand)
    ligand_hall_kier_alpha = rdMolDescriptors.CalcHallKierAlpha(ligand)
    ligand_inertial_shape_factor = rdMolDescriptors.CalcInertialShapeFactor(ligand)
    ligand_kappa1 = rdMolDescriptors.CalcKappa1(ligand)
    ligand_kappa2 = rdMolDescriptors.CalcKappa2(ligand)
    ligand_kappa3 = rdMolDescriptors.CalcKappa3(ligand)
    ligand_labute_asa = rdMolDescriptors.CalcLabuteASA(ligand)
    ligand_npr1 = rdMolDescriptors.CalcNPR1(ligand)
    ligand_npr2 = rdMolDescriptors.CalcNPR2(ligand)
    ligand_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(ligand)
    ligand_pbf = rdMolDescriptors.CalcPBF(ligand)
    ligand_pmi1 = rdMolDescriptors.CalcPMI1(ligand)
    ligand_pmi2 = rdMolDescriptors.CalcPMI2(ligand)
    ligand_pmi3 = rdMolDescriptors.CalcPMI3(ligand)
    ligand_phi = rdMolDescriptors.CalcPhi(ligand)
    ligand_radius_gyration = rdMolDescriptors.CalcRadiusOfGyration(ligand)
    ligand_aliphatic_carbocycles = Lipinski.NumAliphaticCarbocycles(ligand)
    ligand_aliphatic_heterocycles = Lipinski.NumAliphaticHeterocycles(ligand)
    ligand_aliphatic_rings = Lipinski.NumAromaticRings(ligand)
    ligand_aromatic_carbocycles = Lipinski.NumAromaticCarbocycles(ligand)
    ligand_aromatic_heterocycles = Lipinski.NumAromaticHeterocycles(ligand)
    ligand_aromatic_rings = Lipinski.NumAromaticRings(ligand)
    ligand_NHOH = Lipinski.NHOHCount(ligand)
    ligand_NO = Lipinski.NOCount(ligand)
    ligand_CSP3 = rdMolDescriptors.CalcFractionCSP3(ligand)
    ligand_saturated_carbocycles = Lipinski.NumSaturatedCarbocycles(ligand)
    ligand_saturated_heterocycles = Lipinski.NumSaturatedHeterocycles(ligand)
    ligand_saturated_rings = Lipinski.NumSaturatedRings(ligand)
    ligand_rings = Lipinski.RingCount(ligand)
    ligand_MolMR = Crippen.MolMR(ligand)
    ligand_qed = QED.qed(ligand)
    ligand_PEOE_VSA1 = MolSurf.PEOE_VSA1(ligand)
    ligand_PEOE_VSA2 = MolSurf.PEOE_VSA2(ligand)
    ligand_PEOE_VSA3 = MolSurf.PEOE_VSA3(ligand)
    ligand_PEOE_VSA4 = MolSurf.PEOE_VSA4(ligand)
    ligand_PEOE_VSA5 = MolSurf.PEOE_VSA5(ligand)
    ligand_PEOE_VSA6 = MolSurf.PEOE_VSA6(ligand)
    ligand_PEOE_VSA7 = MolSurf.PEOE_VSA7(ligand)
    ligand_PEOE_VSA8 = MolSurf.PEOE_VSA8(ligand)
    ligand_PEOE_VSA9 = MolSurf.PEOE_VSA9(ligand)
    ligand_PEOE_VSA10 = MolSurf.PEOE_VSA10(ligand)
    ligand_PEOE_VSA11 = MolSurf.PEOE_VSA11(ligand)
    ligand_PEOE_VSA12 = MolSurf.PEOE_VSA12(ligand)
    ligand_PEOE_VSA13 = MolSurf.PEOE_VSA13(ligand)
    ligand_PEOE_VSA14 = MolSurf.PEOE_VSA14(ligand)
    ligand_SMR_VSA1 = MolSurf.SMR_VSA1(ligand)
    ligand_SMR_VSA2 = MolSurf.SMR_VSA2(ligand)
    ligand_SMR_VSA3 = MolSurf.SMR_VSA3(ligand)
    ligand_SMR_VSA4 = MolSurf.SMR_VSA4(ligand)
    ligand_SMR_VSA5 = MolSurf.SMR_VSA5(ligand)
    ligand_SMR_VSA6 = MolSurf.SMR_VSA6(ligand)
    ligand_SMR_VSA7 = MolSurf.SMR_VSA7(ligand)
    ligand_SMR_VSA8 = MolSurf.SMR_VSA8(ligand)
    ligand_SlogP_VSA1 = MolSurf.SlogP_VSA1(ligand)
    ligand_SlogP_VSA2 = MolSurf.SlogP_VSA2(ligand)
    ligand_SlogP_VSA3 = MolSurf.SlogP_VSA3(ligand)
    ligand_SlogP_VSA4 = MolSurf.SlogP_VSA4(ligand)
    ligand_SlogP_VSA5 = MolSurf.SlogP_VSA5(ligand)
    ligand_SlogP_VSA6 = MolSurf.SlogP_VSA6(ligand)
    ligand_SlogP_VSA7 = MolSurf.SlogP_VSA7(ligand)
    ligand_SlogP_VSA8 = MolSurf.SlogP_VSA8(ligand)
    ligand_SlogP_VSA9 = MolSurf.SlogP_VSA9(ligand)
    ligand_SlogP_VSA10 = MolSurf.SlogP_VSA10(ligand)
    ligand_SlogP_VSA11 = MolSurf.SlogP_VSA11(ligand)
    ligand_SlogP_VSA12 = MolSurf.SlogP_VSA12(ligand)
    ligand_VSA_EState1 = EState.EState_VSA.VSA_EState1(ligand)
    ligand_VSA_EState2 = EState.EState_VSA.VSA_EState2(ligand)
    ligand_VSA_EState3 = EState.EState_VSA.VSA_EState3(ligand)
    ligand_VSA_EState4 = EState.EState_VSA.VSA_EState4(ligand)
    ligand_VSA_EState5 = EState.EState_VSA.VSA_EState5(ligand)
    ligand_VSA_EState6 = EState.EState_VSA.VSA_EState6(ligand)
    ligand_VSA_EState7 = EState.EState_VSA.VSA_EState7(ligand)
    ligand_VSA_EState8 = EState.EState_VSA.VSA_EState8(ligand)
    ligand_VSA_EState9 = EState.EState_VSA.VSA_EState9(ligand)
    ligand_VSA_EState10 = EState.EState_VSA.VSA_EState10(ligand)

    whole_ligand_features = [ligand_logp, ligand_tpsa, ligand_asphericity, ligand_chi0n, ligand_chi0v, ligand_chi1n,
                             ligand_chi1v, ligand_chi2n, ligand_chi2v, ligand_chi3n, ligand_chi3v, ligand_chi4n,
                             ligand_chi4v, ligand_eccentricity, ligand_hall_kier_alpha, ligand_inertial_shape_factor,
                             ligand_kappa1, ligand_kappa2, ligand_kappa3, ligand_labute_asa, ligand_npr1, ligand_npr2,
                             ligand_rotatable_bonds, ligand_pbf, ligand_pmi1, ligand_pmi2, ligand_pmi3, ligand_phi,
                             ligand_radius_gyration, ligand_aliphatic_carbocycles,
                             ligand_aliphatic_heterocycles, ligand_aliphatic_rings, ligand_aromatic_heterocycles,
                             ligand_aromatic_carbocycles, ligand_aromatic_rings, ligand_NHOH, ligand_NO, ligand_CSP3,
                             ligand_saturated_carbocycles, ligand_saturated_heterocycles, ligand_saturated_rings,
                             ligand_rings, ligand_MolMR, ligand_qed, ligand_PEOE_VSA1, ligand_PEOE_VSA2,
                             ligand_PEOE_VSA3, ligand_PEOE_VSA4, ligand_PEOE_VSA5, ligand_PEOE_VSA6, ligand_PEOE_VSA7,
                             ligand_PEOE_VSA8, ligand_PEOE_VSA9, ligand_PEOE_VSA10, ligand_PEOE_VSA11,
                             ligand_PEOE_VSA12, ligand_PEOE_VSA13, ligand_PEOE_VSA14, ligand_SMR_VSA1, ligand_SMR_VSA2,
                             ligand_SMR_VSA3, ligand_SMR_VSA4, ligand_SMR_VSA5, ligand_SMR_VSA6, ligand_SMR_VSA7,
                             ligand_SMR_VSA8, ligand_SlogP_VSA1, ligand_SlogP_VSA2, ligand_SlogP_VSA3,
                             ligand_SlogP_VSA4, ligand_SlogP_VSA5, ligand_SlogP_VSA6, ligand_SlogP_VSA7,
                             ligand_SlogP_VSA8, ligand_SlogP_VSA9, ligand_SlogP_VSA10, ligand_SlogP_VSA11,
                             ligand_SlogP_VSA12, ligand_VSA_EState1, ligand_VSA_EState2, ligand_VSA_EState3,
                             ligand_VSA_EState4, ligand_VSA_EState5, ligand_VSA_EState6, ligand_VSA_EState7,
                             ligand_VSA_EState8, ligand_VSA_EState9, ligand_VSA_EState10]
    print("Done with ligand features")

    # Calculating protein topological and structural features
    complex_graph.graph["ligand_attr"] = whole_ligand_features
    pocket_logp = Descriptors.MolLogP(pocket)
    pocket_tpsa = rdMolDescriptors.CalcTPSA(pocket)
    pocket_asphericity = rdMolDescriptors.CalcAsphericity(pocket)
    pocket_chi0n = rdMolDescriptors.CalcChi0n(pocket)
    pocket_chi0v = rdMolDescriptors.CalcChi0v(pocket)
    pocket_chi1n = rdMolDescriptors.CalcChi1n(pocket)
    pocket_chi1v = rdMolDescriptors.CalcChi1v(pocket)
    pocket_chi2n = rdMolDescriptors.CalcChi2n(pocket)
    pocket_chi2v = rdMolDescriptors.CalcChi2v(pocket)
    pocket_chi3n = rdMolDescriptors.CalcChi3n(pocket)
    pocket_chi3v = rdMolDescriptors.CalcChi3v(pocket)
    pocket_chi4n = rdMolDescriptors.CalcChi4n(pocket)
    pocket_chi4v = rdMolDescriptors.CalcChi4v(pocket)
    pocket_eccentricity = rdMolDescriptors.CalcEccentricity(pocket)
    pocket_hall_kier_alpha = rdMolDescriptors.CalcHallKierAlpha(pocket)
    pocket_inertial_shape_factor = rdMolDescriptors.CalcInertialShapeFactor(pocket)
    pocket_kappa1 = rdMolDescriptors.CalcKappa1(pocket)
    pocket_kappa2 = rdMolDescriptors.CalcKappa2(pocket)
    pocket_kappa3 = rdMolDescriptors.CalcKappa3(pocket)
    pocket_labute_asa = rdMolDescriptors.CalcLabuteASA(pocket)
    pocket_npr1 = rdMolDescriptors.CalcNPR1(pocket)
    pocket_npr2 = rdMolDescriptors.CalcNPR2(pocket)
    pocket_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(pocket)
    pocket_pbf = rdMolDescriptors.CalcPBF(pocket)
    pocket_pmi1 = rdMolDescriptors.CalcPMI1(pocket)
    pocket_pmi2 = rdMolDescriptors.CalcPMI2(pocket)
    pocket_pmi3 = rdMolDescriptors.CalcPMI3(pocket)
    pocket_phi = rdMolDescriptors.CalcPhi(pocket)
    pocket_radius_gyration = rdMolDescriptors.CalcRadiusOfGyration(pocket)
    pocket_MolMR = Crippen.MolMR(pocket)
    pocket_PEOE_VSA1 = MolSurf.PEOE_VSA1(pocket)
    pocket_PEOE_VSA2 = MolSurf.PEOE_VSA2(pocket)
    pocket_PEOE_VSA3 = MolSurf.PEOE_VSA3(pocket)
    pocket_PEOE_VSA4 = MolSurf.PEOE_VSA4(pocket)
    pocket_PEOE_VSA5 = MolSurf.PEOE_VSA5(pocket)
    pocket_PEOE_VSA6 = MolSurf.PEOE_VSA6(pocket)
    pocket_PEOE_VSA7 = MolSurf.PEOE_VSA7(pocket)
    pocket_PEOE_VSA8 = MolSurf.PEOE_VSA8(pocket)
    pocket_PEOE_VSA9 = MolSurf.PEOE_VSA9(pocket)
    pocket_PEOE_VSA10 = MolSurf.PEOE_VSA10(pocket)
    pocket_PEOE_VSA11 = MolSurf.PEOE_VSA11(pocket)
    pocket_PEOE_VSA12 = MolSurf.PEOE_VSA12(pocket)
    pocket_PEOE_VSA13 = MolSurf.PEOE_VSA13(pocket)
    pocket_PEOE_VSA14 = MolSurf.PEOE_VSA14(pocket)
    pocket_SMR_VSA1 = MolSurf.SMR_VSA1(pocket)
    pocket_SMR_VSA2 = MolSurf.SMR_VSA2(pocket)
    pocket_SMR_VSA3 = MolSurf.SMR_VSA3(pocket)
    pocket_SMR_VSA4 = MolSurf.SMR_VSA4(pocket)
    pocket_SMR_VSA5 = MolSurf.SMR_VSA5(pocket)
    pocket_SMR_VSA6 = MolSurf.SMR_VSA6(pocket)
    pocket_SMR_VSA7 = MolSurf.SMR_VSA7(pocket)
    pocket_SMR_VSA8 = MolSurf.SMR_VSA8(pocket)
    pocket_SlogP_VSA1 = MolSurf.SlogP_VSA1(pocket)
    pocket_SlogP_VSA2 = MolSurf.SlogP_VSA2(pocket)
    pocket_SlogP_VSA3 = MolSurf.SlogP_VSA3(pocket)
    pocket_SlogP_VSA4 = MolSurf.SlogP_VSA4(pocket)
    pocket_SlogP_VSA5 = MolSurf.SlogP_VSA5(pocket)
    pocket_SlogP_VSA6 = MolSurf.SlogP_VSA6(pocket)
    pocket_SlogP_VSA7 = MolSurf.SlogP_VSA7(pocket)
    pocket_SlogP_VSA8 = MolSurf.SlogP_VSA8(pocket)
    pocket_SlogP_VSA9 = MolSurf.SlogP_VSA9(pocket)
    pocket_SlogP_VSA10 = MolSurf.SlogP_VSA10(pocket)
    pocket_SlogP_VSA11 = MolSurf.SlogP_VSA11(pocket)
    pocket_SlogP_VSA12 = MolSurf.SlogP_VSA12(pocket)
    pocket_VSA_EState1 = EState.EState_VSA.VSA_EState1(pocket)
    pocket_VSA_EState2 = EState.EState_VSA.VSA_EState2(pocket)
    pocket_VSA_EState3 = EState.EState_VSA.VSA_EState3(pocket)
    pocket_VSA_EState4 = EState.EState_VSA.VSA_EState4(pocket)
    pocket_VSA_EState5 = EState.EState_VSA.VSA_EState5(pocket)
    pocket_VSA_EState6 = EState.EState_VSA.VSA_EState6(pocket)
    pocket_VSA_EState7 = EState.EState_VSA.VSA_EState7(pocket)
    pocket_VSA_EState8 = EState.EState_VSA.VSA_EState8(pocket)
    pocket_VSA_EState9 = EState.EState_VSA.VSA_EState9(pocket)
    pocket_VSA_EState10 = EState.EState_VSA.VSA_EState10(pocket)
    whole_pocket_features = [pocket_logp, pocket_tpsa, pocket_asphericity, pocket_chi0n, pocket_chi0v, pocket_chi1n,
                             pocket_chi1v, pocket_chi2n, pocket_chi2v, pocket_chi3n, pocket_chi3v, pocket_chi4n,
                             pocket_chi4v, pocket_eccentricity, pocket_hall_kier_alpha, pocket_inertial_shape_factor,
                             pocket_kappa1, pocket_kappa2, pocket_kappa3, pocket_labute_asa, pocket_npr1, pocket_npr2,
                             pocket_rotatable_bonds, pocket_pbf, pocket_pmi1, pocket_pmi2, pocket_pmi3, pocket_phi,
                             pocket_radius_gyration, pocket_MolMR, pocket_PEOE_VSA1, pocket_PEOE_VSA2,
                             pocket_PEOE_VSA3, pocket_PEOE_VSA4, pocket_PEOE_VSA5, pocket_PEOE_VSA6, pocket_PEOE_VSA7,
                             pocket_PEOE_VSA8, pocket_PEOE_VSA9, pocket_PEOE_VSA10, pocket_PEOE_VSA11,
                             pocket_PEOE_VSA12, pocket_PEOE_VSA13, pocket_PEOE_VSA14, pocket_SMR_VSA1, pocket_SMR_VSA2,
                             pocket_SMR_VSA3, pocket_SMR_VSA4, pocket_SMR_VSA5, pocket_SMR_VSA6, pocket_SMR_VSA7,
                             pocket_SMR_VSA8, pocket_SlogP_VSA1, pocket_SlogP_VSA2, pocket_SlogP_VSA3,
                             pocket_SlogP_VSA4, pocket_SlogP_VSA5, pocket_SlogP_VSA6, pocket_SlogP_VSA7,
                             pocket_SlogP_VSA8, pocket_SlogP_VSA9, pocket_SlogP_VSA10, pocket_SlogP_VSA11,
                             pocket_SlogP_VSA12, pocket_VSA_EState1, pocket_VSA_EState2, pocket_VSA_EState3,
                             pocket_VSA_EState4, pocket_VSA_EState5, pocket_VSA_EState6, pocket_VSA_EState7,
                             pocket_VSA_EState8, pocket_VSA_EState9, pocket_VSA_EState10]
    print("Done with pocket features")
    complex_graph.graph["pocket_attr"] = whole_pocket_features
    for atom in ligand.GetAtoms():
        atom_index = atom.GetIdx()
        atom_features = [atom.GetAtomicNum()]
        atom_features.extend(get_residue_number(atom))
        atom_features.extend(get_hybridization_number(atom))
        atom_features.extend([atom.GetDegree(), atom.GetIsAromatic(), atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(), (atom.GetMass() - masses_mean) / masses_sd, atom.GetFormalCharge(), (get_gasteiger_charge(atom) - gasteiger_mean) / gasteiger_sd])
        coordinates = (float(ligand_positions[atom_index][0]), float(ligand_positions[atom_index][1]),
                       float(ligand_positions[atom_index][2]))
        normalized_coordinates = (coordinates - mean_coords) / std_coords
        atom_features += list(normalized_coordinates)
        complex_graph.add_node(atom_index, x=torch.tensor(atom_features, dtype=torch.float))

    # Adding ligand bonds to graph
    count = 0
    for bond in ligand.GetBonds():
        ligand_edge_features = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        start_atom = bond.GetBeginAtomIdx()
        start_atom_coordinates = ligand_positions[start_atom]
        end_atom = bond.GetEndAtomIdx()
        end_atom_coordinates = ligand_positions[end_atom]
        distance = np.linalg.norm(end_atom_coordinates - start_atom_coordinates)
        normalized_distance = (distance - min_distance) / (max_distance - min_distance)
        bond_type = bond.GetBondTypeAsDouble()
        ligand_edge_features[9] = bond_type
        ligand_edge_features += [normalized_distance]
        complex_graph.add_edge(start_atom, end_atom, edge_attr=torch.tensor(ligand_edge_features, dtype=torch.float))
        count += 1

    count = 0
    # Adding pocket atoms to graph
    for l_atom, p_atom, latom_position, patom_position, normalized_distance in close_pairs:
        atom = pocket.GetAtomWithIdx(p_atom)
        atom_index = p_atom + ligand.GetNumAtoms()
        atom_features = [atom.GetAtomicNum()]
        atom_features.extend(get_residue_number(atom))
        atom_features.extend(get_hybridization_number(atom))
        atom_features.extend([atom.GetDegree(), atom.GetIsAromatic(), atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(), (atom.GetMass() - masses_mean) / masses_sd, atom.GetFormalCharge(), (get_gasteiger_charge(atom) - gasteiger_mean) / gasteiger_sd])
        coordinates = tuple(patom_position)
        normalized_coordinates = (coordinates - mean_coords) / std_coords
        atom_features += list(normalized_coordinates)
        complex_graph.add_node(atom_index, x=torch.tensor(atom_features, dtype=torch.float))

        # Adding interaction edge features to graph between protein and ligand atoms:
        # [ee, hab, hyd, hypc, mc, pp, t, sb, cp, bond type]
        interaction_edge_features = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for feature, pairs in edge_features_dict.items():
            for pair in pairs:
                ligand_coordinate = pair[0]
                pocket_coordinate = pair[1]
                if feature == "Electrostatic Energies":
                    if ligand_coordinate == tuple(latom_position):
                        if pocket_coordinate == tuple(patom_position):
                            energy = pair[2]
                            interaction_edge_features[0] = energy
                            break
                if feature == "Halogen Bonds":
                    if ligand_coordinate == tuple(latom_position):
                        if pocket_coordinate == tuple(patom_position):
                            interaction_edge_features[1] = 1
                            break
                if feature == "Hydrogen Bonds":
                    if ligand_coordinate == tuple(latom_position):
                        if pocket_coordinate == tuple(patom_position):
                            interaction_edge_features[2] = 1
                            break
                if feature == "Hydrophobic Contacts":
                    if ligand_coordinate == tuple(latom_position):
                        if pocket_coordinate == tuple(patom_position):
                            interaction_edge_features[3] = 1
                            break
                if feature == "Metal Contacts":
                    if ligand_coordinate == tuple(latom_position):
                        if pocket_coordinate == tuple(patom_position):
                            interaction_edge_features[4] = 1
                            break
                if feature == "Pi-Pi Stacking":
                    if ligand_coordinate == tuple(latom_position):
                        if pocket_coordinate == tuple(patom_position):
                            interaction_edge_features[5] = 1
                            break
                if feature == "T-stacking":
                    if ligand_coordinate == tuple(latom_position):
                        if pocket_coordinate == tuple(patom_position):
                            interaction_edge_features[6] = 1
                            break
                if feature == "Salt Bridges":
                    if ligand_coordinate == tuple(latom_position):
                        if pocket_coordinate == tuple(patom_position):
                            interaction_edge_features[7] = 1
                            break
                if feature == "Cation-Pi Interactions":
                    if ligand_coordinate == tuple(latom_position):
                        if pocket_coordinate == tuple(patom_position):
                            interaction_edge_features[8] = 1
                            break

        edge_features = interaction_edge_features + [normalized_distance]
        complex_graph.add_edge(atom_index, l_atom, edge_attr=torch.tensor(edge_features, dtype=torch.float))
        count += 1

    return complex_graph, num_ligand_atoms
