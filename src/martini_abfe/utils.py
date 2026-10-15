"""Helper functions for the simulation setup and execution."""

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FileUpdater:
    """
    Context manager for reading and updating files.
    Automatically closes the file after exiting the context.
    """

    def __init__(self, file_path: Path):
        self.file = open(file_path, "r+", encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def __getattr__(self, name):
        return getattr(self.file, name)

    def __iter__(self):
        return iter(self.file)

    def __repr__(self):
        return f"<FileUpdater file={self.file.name!r}>"

    def __bool__(self):
        return not self.file.closed

    def overwrite(self, content: str | list[str]):
        """
        Overwrite the file with the given content.
        :param content: Content to write to the file. If a list is provided, each element
            is written as a separate line.
        """
        self.file.seek(0)
        if isinstance(content, list):
            self.file.writelines(content)
        else:
            self.file.write(content)
        self.file.truncate()


def check_file_exists(*file_paths: str | Path, message: str = None):
    """
    Check if files exist or raise an error.
    :param file_paths: List of file paths to check.
    :param message: Optional custom error message. If not provided, a default message is used.
    :raises FileNotFoundError: If any of the files do not exist.
    """
    for file_path in file_paths:
        if not Path(file_path).exists():
            if message is None:
                message = f"File {file_path} does not exist."
            raise FileNotFoundError(message)


def read_json_file(json_file: Path) -> dict:
    """
    Read a JSON file and return its contents as a dictionary.
    :param json_file: Path to the JSON file.
    :return: Contents of the JSON file as a dictionary.
    """
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


class CompactListEncoder(json.JSONEncoder):
    """JSON encoder writing lists of numbers on a single line."""

    def _encode(self, obj, level=0):
        indent = " " * (self.indent or 4) * level
        if isinstance(obj, list) and all(isinstance(i, (int, float, str)) for i in obj):
            items = [round(i, 6) if isinstance(i, float) else i for i in obj]
            return json.dumps(items, separators=(",", ":"))
        elif isinstance(obj, dict):
            inner = []
            for k, v in obj.items():
                inner.append(
                    " " * (self.indent or 4) * (level + 1)
                    + json.dumps(k)
                    + ": "
                    + self._encode(v, level + 1)
                )
            return "{\n" + ",\n".join(inner) + "\n" + indent + "}"
        else:
            return json.dumps(obj)

    def iterencode(self, obj, _one_shot=False):
        yield self._encode(obj)


def write_json_file(data: dict, json_file: Path, compact_lists: bool = False):
    """
    Write a dictionary to a JSON file.
    :param data: Dictionary to write to the JSON file.
    :param json_file: Path to the JSON file.
    :param compact_lists: If True, use compact list encoding.
    """
    with open(json_file, "w", encoding="utf-8") as file:
        if compact_lists:
            json.dump(data, file, cls=CompactListEncoder, indent=4)
        else:
            json.dump(data, file, indent=4)
    logger.debug("Wrote data to JSON file %s", json_file)


def local_path(path: str | Path) -> str:
    """
    Shorten a path for messages, relative to the working directory if it lies below it.
    :param path: Path to shorten.
    :return: Relative path, or the path unchanged if it lies outside the working directory.
    """
    relative = os.path.relpath(path)
    return str(path) if relative.startswith("..") else relative


def replace_in_file(file_path: Path, replacements: dict):
    """
    Replace strings in a file based on a dictionary of replacements.
    :param file_path: Path to the file.
    :param replacements: Dictionary where keys are strings to be replaced and values are the
        replacement strings.
    """
    with FileUpdater(file_path) as file:
        content = file.read()
        for old, new in replacements.items():
            content = content.replace(old, new)
        file.overwrite(content)
    logger.debug("Replaced strings in file %s", file_path)


def clean_pdb_structure(
    input_file: Path,
    output_clean: Path,
    chain: str | list[str] = None,
    keep: list[str] = None,
) -> list[str]:
    """
    Remove all non-ATOM, non-TER, and non-END lines from a pdb file. Only the first model
    and the first alternate location of every atom are kept.
    :param input_file: Path to the input pdb file.
    :param output_clean: Path to the output cleaned pdb file.
    :param chain: If specified, only keep the atoms of this chain or list of chains,
        given as list or comma separated string.
    :param keep: List of line prefixes to keep, if None, only ATOM, TER, and END are kept.
    :return: Chain identifiers of the kept atoms, in order of appearance.
    """
    if keep is None:
        keep = ["ATOM", "TER", "END"]
    if isinstance(chain, str):
        chain = [c.strip() for c in chain.split(",")]
    with open(input_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()
    output_lines = []
    chains = []
    for line in lines:
        if line.startswith("ENDMDL"):
            break
        record = line[:6].strip()
        if record not in keep:
            continue
        if record == "ATOM":
            if chain is not None and line[21].strip() not in chain:
                continue
            # Keep only the first alternate location and drop its label
            if line[16] not in (" ", "A", "1"):
                continue
            line = line[:16] + " " + line[17:]
            if line[21] not in chains:
                chains.append(line[21])
        elif record == "TER" and (
            len(output_lines) == 0 or output_lines[-1].startswith("TER")
        ):
            continue
        elif record == "END":
            break
        output_lines.append(line)
    output_lines.append("END\n")
    with open(output_clean, "w", encoding="utf-8") as outfile:
        outfile.writelines(output_lines)
    logger.debug("Cleaned up pdb structure and saved to %s", output_clean)
    return chains


def get_charge_from_itp(itp_file: Path) -> float:
    """
    Get the total charge from a GROMACS itp file.
    :param itp_file: Path to the itp file.
    :return: Total charge.
    """
    total_charge = 0.0
    top_file_reading_status = False
    with open(itp_file, "r", encoding="utf-8") as top_file:
        for line in top_file:
            line = line.strip()
            if len(line) > 0 and line[0] in [";", "#"]:
                continue
            splitted_line = line.split()
            if top_file_reading_status and len(splitted_line) > 6:
                total_charge += float(splitted_line[6])
            elif "atoms" in line and "[" in line:
                top_file_reading_status = True
            else:
                top_file_reading_status = False
    if not np.isclose(total_charge, np.round(total_charge)):
        raise ValueError(f"Total charge is not an integer: {total_charge}")
    return int(np.round(total_charge))


def get_molecule_name_from_itp(itp_file: Path) -> str:
    """
    Get the molecule name from a GROMACS itp file.
    :param itp_file: Path to the itp file.
    :return: Molecule name.
    """
    with open(itp_file, "r", encoding="utf-8") as top_file:
        read_next_line = False
        for line in top_file:
            line = line.strip()
            if len(line) > 0 and line[0] in [";", "#"]:
                continue
            if "moleculetype" in line:
                read_next_line = True
            elif read_next_line:
                if len(line.split()) > 0:
                    return line.split()[0]
    raise ValueError(f"Could not find molecule name in {itp_file}")


def get_residue_com_from_pdb(pdb_file: Path, residue: str | int) -> np.ndarray:
    """
    Get the center of mass of a residue from a pdb file.
    :param pdb_file: Path to the pdb file.
    :param residue: Residue name or number.
    :return: Center of mass as a numpy array in nm.
    """
    coords = []
    with open(pdb_file, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                res_id = line[22:26].strip()
                res_name = line[17:20].strip()
                if str(residue) == res_id or str(residue) == res_name:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coords.append([x, y, z])
    if len(coords) == 0:
        raise ValueError(f"Residue {residue} not found in {pdb_file}")
    return np.mean(np.array(coords), axis=0) / 10  # Convert to nm


def get_n_lambda_steps_from_mdp(mdp_file: str | Path) -> int:
    """
    Get the number of lambda steps from a GROMACS MDP file.
    :param mdp_file: Path to the MDP file.
    :return: Number of lambda steps.
    """
    with open(mdp_file, "r", encoding="utf-8") as file:
        for line in file:
            if "lambdas" in line:
                return len(line.split("=")[1].strip().split())
    raise ValueError(f"Could not find lambdas in {mdp_file}")


def create_gromacs_index(
    gro_filename: str | Path,
    index_filename: str | Path,
    selections: dict[str, Callable[[pd.DataFrame], pd.DataFrame | list[pd.DataFrame]]],
):
    """
    Create an index file for the system coordinate file.
    Similar to the 'make_ndx' command in GROMACS, but programmable with python.
    :param gro_filename: Name of the coordinate file in .gro format.
    :param index_filename: Name of the index file to create.
    :param selections: Selection name to a function selecting rows of the coordinate frame
        with the columns 'resid', 'resname', 'atomname', 'atomid', 'x', 'y' and 'z'. A
        function returning a list or groupby gets a 1-based index appended to its name.
    """
    # Parse coordinate file
    with open(gro_filename, "r", encoding="utf-8") as coord_file:
        lines = [line for line in coord_file.readlines() if len(line.strip()) > 0]
    data = []
    for line in lines[2:-1]:
        data.append(
            [
                int(line[:5]),
                line[5:10].strip(),
                line[10:15].strip(),
                int(line[15:20]),
                float(line[20:28]),
                float(line[28:36]),
                float(line[36:44]),
            ]
        )
    data = pd.DataFrame(
        data, columns=["resid", "resname", "atomname", "atomid", "x", "y", "z"]
    )
    # Apply selections to data and add to dict. If the selection returns a groupby object
    # or list, iterate over the groups and add them to the dict with an index.
    selection_data = {}
    for selection_name, selection in selections.items():
        selection_group = selection(data)
        if isinstance(selection_group, pd.core.groupby.generic.DataFrameGroupBy):
            for i, (_, sg) in enumerate(selection_group):
                selection_data[f"{selection_name}{i + 1}"] = sg
        elif isinstance(selection_group, list):
            for i, sg in enumerate(selection_group):
                selection_data[f"{selection_name}{i + 1}"] = sg
        else:
            selection_data[selection_name] = selection_group
    # Write selections to index file
    with open(index_filename, "w", encoding="utf-8") as index_file:
        for selection_name, selection_group in selection_data.items():
            atom_indices = selection_group["atomid"].values
            index_file.write(f"[ {selection_name} ]\n")
            for i, atom_index in enumerate(atom_indices):
                index_file.write(f"{atom_index:>4}")
                if i % 15 == 14 or i == len(atom_indices) - 1:
                    index_file.write("\n")
                else:
                    index_file.write(" ")
