'''
Library containing utility functions we use frequently.
'''
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'preprocessing')))

import pandas as pd
import warnings
from pathlib import Path
from typing import List
import xic_to_csd_utils
from concurrent.futures import ProcessPoolExecutor
import re

def infer_seq_col_name(df: pd.DataFrame, substr='seq') -> str:
    '''
    ## Input:
    `pandas.DataFrame` with a column whose name contains `'seq'` (or whatever `substr` is).
    ## Output:
    The name of the column containing peptide sequences.
    '''
    seq_col_name = df.columns[df.columns.str.contains(substr, case=False)]

    if len(seq_col_name) < 1:
        raise ValueError("No peptide column found. Make sure the column containing the sequence has 'seq' in its name.")
    elif len(seq_col_name) > 1:
        warnings.warn("More than one column potentially contains peptides. Using first one.")
        seq_col_name = seq_col_name[0]

    return seq_col_name[0]

def infer_charge_col_name(df: pd.DataFrame, substr='charge') -> str:
    '''
    ## Input:
    `pandas.DataFrame` with a column whose name contains `'charge'` (or whatever `substr` is).
    ## Output:
    The name of the column containing precursor charge states.
    '''
    charge_col_name = df.columns[df.columns.str.contains(substr, case=False)]

    if len(charge_col_name) < 1:
        raise ValueError("No charge column found. Make sure the column containing the charge has 'charge' in its name.")
    elif len(charge_col_name) > 1:
        warnings.warn("More than one column potentially contains peptides. Using first one.")
        charge_col_name = charge_col_name[0]

    return charge_col_name[0]

def infer_intensity_col_name(df: pd.DataFrame, substr='intens') -> str:
    '''
    ## Input:
    `pandas.DataFrame` with a column whose name contains `'intens'` (or whatever `substr` is).
    ## Output:
    The name of the column containing precursor intensities.
    '''
    intensity_col_name = df.columns[df.columns.str.contains(substr, case=False)]

    if len(intensity_col_name) < 1:
        raise ValueError("No intensity column found. Make sure the column containing the precursor intensity has 'intens' in its name.")
    elif len(intensity_col_name) > 1:
        warnings.warn("More than one column potentially contains precursor intensities. Using first one.")
        intensity_col_name = intensity_col_name[0]

    return intensity_col_name[0]

def infer_rawfile_col_name(df: pd.DataFrame, substr='raw') -> str:
    '''
    ## Input:
    `pandas.DataFrame` with a column whose name contains `'raw'` (or whatever `substr` is).
    ## Output:
    The name of the column containing raw file names.
    '''
    rawfile_col_name = df.columns[df.columns.str.contains(substr, case=False)]

    if len(rawfile_col_name) < 1:
        raise ValueError("No rawfile column found. Make sure the column containing the rawfile name has 'raw' in its name.")
    elif len(rawfile_col_name) > 1:
        warnings.warn("More than one column potentially contains rawfile names. Using first one.")
        rawfile_col_name = rawfile_col_name[0]

    return rawfile_col_name[0]

def infer_fragmentation_col_name(df: pd.DataFrame, substr='fragm') -> str:
    '''
    ## Input:
    `pandas.DataFrame` with a column whose name contains `'seq'` (or whatever `substr` is).
    ## Output:
    The name of the column containing the fragmentation method.
    '''
    frag_col_name = df.columns[df.columns.str.contains(substr, case=False)]

    if len(frag_col_name) < 1:
        raise ValueError("No fragmentation column found. Make sure the column containing the sequence has 'fragm' in its name.")
    elif len(frag_col_name) > 1:
        warnings.warn("More than one column potentially contains the fragmentation method. Using first one.")
        frag_col_name = frag_col_name[0]

    return frag_col_name[0]

def get_cs_distribution_row_count_sum_norm(df: pd.DataFrame, legal_charges: list=[]) -> pd.DataFrame:
    '''
    ## Input:
    `pandas.DataFrame` object containing a peptide (must contain `'seq'`) and a charge (must contain `'charge'`) column.
    An optional list of charges. Is empty by default, meaning all charges in the dataframe are included. Rows containing illegal charges are discarded prior to calculating the distribution.

    ## Output:
    A `pandas.DataFrame` where sequences are unique und their charge state distribution based on counting rows and sum normalising these counts.
    '''
    seq_col_name = infer_seq_col_name(df)

    charge_col_name = infer_charge_col_name(df)

    if legal_charges:
        df = df[df[charge_col_name].isin(legal_charges)]

    pivot_df = df.pivot_table(index=seq_col_name, columns=charge_col_name, aggfunc='size', fill_value=0)
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)
    pivot_df.columns = ['charge_' + str(col) for col in pivot_df.columns]

    if legal_charges:
        for legal_charge in legal_charges:
            col_name = f'charge_{legal_charge}'
            if col_name not in pivot_df.columns:
                pivot_df[col_name] = 0.0

    pivot_df = pivot_df.reset_index()

    return pivot_df

def process_chunk_wrapper(rawfile_chunk, colnames, legal_charges, rawfile_dir, mz_tolerance, ThermoRawFileParser_exe):
    if rawfile_chunk.empty:
        return pd.DataFrame()
    
    rawfile_chunk = rawfile_chunk.reset_index(drop=True)
    raw_file_path = os.path.join(rawfile_dir, f"{rawfile_chunk[colnames['rawfile']].iloc[0]}.raw")

    if not os.path.exists(raw_file_path):
        warnings.warn(f'Missing rawfile: {raw_file_path} does not exist.')
        result = pd.DataFrame()
    else:
        result = xic_to_csd_utils.process_chunk(rawfile_chunk, colnames, legal_charges, rawfile_dir, mz_tolerance, ThermoRawFileParser_exe)

    return result


def get_cs_distribution_from_xics(df: pd.DataFrame, rawfile_dir: str, mz_tolerance: float, num_workers: int=None, legal_charges: list=[1, 2, 3, 4, 5, 6, 7], rt_filter: float=5.0, ThermoRawFileParser_exe='ThermoRawFileParser.exe') -> pd.DataFrame:
    '''
    ## Deprecation notice
    This function is not the best way to extract CSDs.
    The gorshkov method is the preferred XIC based CSD
    extraction method.

    Compute CSDs from XICs as described in the plan:
    `journal/12_prospect_xic.md`.
    Don't apply an rt filter if `rt_filter` is `None`

    ## Attention
    This function, despite working in parallel, can take very long, especially for many rawfiles. 
    '''
    colnames = {
        'seq': infer_seq_col_name(df),
        'charge': infer_charge_col_name(df),
        'intensity': infer_intensity_col_name(df),
        'rawfile': infer_rawfile_col_name(df),
        'mz': 'mz', # hard coded for now bc theres theoretical mz, maybe to param
        'rt': 'retention_time'
    }

    if rt_filter:
        # Get rt window for every peptide + rawfile
        print("Computing RT windows...")
        df = xic_to_csd_utils.get_rt_window(df, colnames, rt_filter)

    # Group by `raw_file` + `modified_sequence` + `precursor_charge`
    # Compute average of `m/z` and `intensity` for this group
    print("Computing max average intensities for groups...")
    df = xic_to_csd_utils.get_max_avg_intense_entry(df, colnames)

    # Based on this (max intense) charge entry, compute hypothetical 
    # `m/z` for all legal charges
    print("Computing theoretical m/z ratios...")
    df = xic_to_csd_utils.get_theoretical_mzs(df, colnames, legal_charges)

    # Split dataset into chunks by `raw_file`
    print("Splitting data by raw_file...")
    rawfile_chunks = xic_to_csd_utils.split_by(df, colnames['rawfile'])

    # Multithreaded processing per rawfile
    print("Multiprocessing chunks...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        processed_rawfile_chunks = list(executor.map(
            process_chunk_wrapper, 
            rawfile_chunks,
            [colnames] * len(rawfile_chunks),
            [legal_charges] * len(rawfile_chunks),
            [rawfile_dir] * len(rawfile_chunks),
            [mz_tolerance] * len(rawfile_chunks),
            [ThermoRawFileParser_exe] * len(rawfile_chunks)
        ))
    
    # Group by `modified_sequence`, 
    # sum up peak intensity XIC vectors
    # for different raw files
    print("Summing up peak XIC intensities...")
    df = pd.concat(processed_rawfile_chunks)
    df = df.reset_index(drop=True)
    df = xic_to_csd_utils.sum_peak_intensities(df, colnames, legal_charges)

    # Sum normalize the vectors
    print("Normalizing...")
    df = xic_to_csd_utils.sum_normalize_xic_vec(df, legal_charges)

    df = df.reset_index()

    return df

def get_cs_distribution_sum_avg_intensity_sum_norm(df: pd.DataFrame, legal_charges: list=[]) -> pd.DataFrame:
    '''
    ## Input:
    `pandas.DataFrame` object containing the following columns:
    - peptide (must contain `'seq'`)
    - precursor charge (must contain `'charge'`)
    - precursor intensity (must contain `'intens'`)
    - rawfile (must contain `'raw'`)
    - fragmentation (must contain `'fragm'`)
    
    An optional list of charges. Is empty by default, meaning all charges in the dataframe are included. Rows containing illegal charges are discarded prior to calculating the distribution.

    ## Output:
    A `pandas.DataFrame` where sequences are unique and their charge state distribution based on intensities summed by peptide + charge + rawfile + fragmentation, then averaged and sum normalized by peptide.
    '''
    seq_col_name = infer_seq_col_name(df)

    charge_col_name = infer_charge_col_name(df)

    intensity_col_name = infer_intensity_col_name(df)

    rawfile_col_name = infer_rawfile_col_name(df)

    frag_col_name = infer_fragmentation_col_name(df)

    df = df.dropna(subset=[intensity_col_name])

    if legal_charges:
        df = df[df[charge_col_name].isin(legal_charges)]

    df.groupby([
        seq_col_name,
        charge_col_name,
        rawfile_col_name, 
        frag_col_name
    ]).sum(numeric_only=True)
    
    df = df.groupby([seq_col_name, charge_col_name]).mean(numeric_only=True)

    df = df.reset_index()

    pivot_df = df.pivot_table(
        index=[seq_col_name],
        columns=charge_col_name, 
        values=intensity_col_name,
        fill_value=0
    )

    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)
    pivot_df.columns = ['charge_' + str(col) for col in pivot_df.columns]
    
    if legal_charges:
        for legal_charge in legal_charges:
            col_name = f'charge_{legal_charge}'
            if col_name not in pivot_df.columns:
                pivot_df[col_name] = 0.0
    
    pivot_df = pivot_df.reset_index()

    return pivot_df

def read_parquet(parquet: str) -> pd.DataFrame:
    '''
    ## Input: 
    A path to either a single parquet file or a directory containing parquet files.
    If a directory, find parquet file recursively.
    ## Output:
    A single `pandas.DataFrame` object with the content from the parquet file(s).
    '''
    if os.path.isdir(parquet):
        parquet = Path(parquet)
        df = pd.concat(
            pd.read_parquet(parquet_file)
            for parquet_file in parquet.rglob('*.parquet')
        )
    else:
        df = pd.read_parquet(parquet)
    
    return df

def get_chunks_by_seq(df: pd.DataFrame) -> List[pd.DataFrame]:
    '''
    ## Input:
    A `pandas.DataFrame` that contains a peptide column (must contain `'seq'` in )
    ## Output:
    A list of `pandas.DataFrame`s, each containing all entries of one sequence,
    i.e. all entries in `df` that share the same sequence.
    '''
    seq_col_name = infer_seq_col_name(df)
    chunks = [group.reset_index() for _, group in df.groupby(seq_col_name)]
    return chunks

def filter_andromeda(df: pd.DataFrame, cutoff=70) -> pd.DataFrame:
    '''
    ## Input:
    Dataframe with column 'andromeda_score' and `cutoff` (default 70)
    ## Output:
    Dataframe where all entries have andromeda score >= 70
    '''
    return df.loc[df['andromeda_score'] >= cutoff]

def filter_rawfiles(df: pd.DataFrame, blacklist: str) -> pd.DataFrame:
    '''
    ## Input:
    Dataframe with a column containing rawfile names, 
    path to a `blacklist` text file containing rawfile names
    ## Output:
    Dataframe with entries from `blacklist`ed rawfiles removed
    '''
    with open(blacklist, 'r') as f:
        blacklisted_rawfiles = [rawfile.strip() for rawfile in f]
    return df.loc[~df['raw_file'].isin(blacklisted_rawfiles)]

def find_filepaths(root_dir: str, file_extension: str, case_insensitive = True) -> list:
    '''
    ## Input:
    Root directory to start search from,
    file extension, e.g. `.raw` or `.mzML`
    ## Output:
    List with all filepaths ending in `file_extension`
    '''
    files = []
    
    if not os.path.exists(root_dir):
        print(f"{root_dir} is not a valid directory.")
        return files
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if case_insensitive and filename.lower().endswith(file_extension.lower()):
                files.append(os.path.join(dirpath, filename))
            elif filename.endswith(file_extension):
                files.append(os.path.join(dirpath, filename))
    return files


def compute_monoisotopic_mass(modified_sequence: str):
    '''
    ## Input:
    Amino acid sequence with modifications in UNIMOD format.
    Only modifications found in the Seitz dataset are supported.
    ## Output:
    Monoisotopic mass of the modified sequence.
    '''
    # initialize other masses
    PARTICLE_MASSES = {"PROTON": 1.007276467, "ELECTRON": 0.00054858}

    # masses of different atoms
    ATOM_MASSES = {
        "H": 1.007825035,
        "C": 12.0,
        "O": 15.9949146,
        "N": 14.003074,
    }

    MASSES = {**PARTICLE_MASSES, **ATOM_MASSES}
    MASSES["N_TERMINUS"] = MASSES["H"]
    MASSES["C_TERMINUS"] = MASSES["H"] + MASSES["O"]


    AA_MASSES = {
        "A": 71.037114,
        "R": 156.101111,
        "N": 114.042927,
        "D": 115.026943,
        "C": 103.009185,
        "E": 129.042593,
        "Q": 128.058578,
        "G": 57.021464,
        "H": 137.058912,
        "I": 113.084064,
        "L": 113.084064,
        "K": 128.094963,
        "M": 131.040485,
        "F": 147.068414,
        "P": 97.052764,
        "S": 87.032028,
        "T": 101.047679,
        "U": 150.95363,
        "W": 186.079313,
        "Y": 163.063329,
        "V": 99.068414,
        "[]-": MASSES["N_TERMINUS"],
        "-[]": MASSES["C_TERMINUS"],
    }

    MOD_MASSES = {
        "[UNIMOD:737]": 229.162932,  # TMT_6
        "[UNIMOD:1342]": 329.226595, # other TMT
        "[]": 0.0,
        "[UNIMOD:1]": 42.010565,  # Acetylation
        "[UNIMOD:21]": 79.966331,  # Phosphorylation
        "[UNIMOD:35]": 15.994915,  # Hydroxylation
        "[UNIMOD:4]": 57.021464,  # Carbamidomethyl
        "[UNIMOD:5]": 43.005814,  # Carbamyl
        "[UNIMOD:7]": 0.984016,  # Deamidation
        "[UNIMOD:312]": 119.004099, # Cysteinylation
        "[UNIMOD:425]": 31.989829, # dihydroxy
        "[UNIMOD:27]": -18.010565 # Pyro-glu from E
    }

    total_mass = 0.0
    pattern = re.compile(r'(\[UNIMOD:\d+\]|\[\]-|-\[]|[A-Z])')
    parts = pattern.findall(modified_sequence)

    for part in parts:
        if part in AA_MASSES:
            total_mass += AA_MASSES[part]
        elif part in MOD_MASSES:
            total_mass += MOD_MASSES[part]

    return total_mass
