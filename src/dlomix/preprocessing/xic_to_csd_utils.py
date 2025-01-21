'''
Collection of utility functions for extracting ion chromatograms and computing charge state distributions from them
'''

import os
import numpy as np
import pandas as pd
import json
from typing import List
import threading
import subprocess
from sklearn.metrics import auc

def get_max_avg_intense_entry(df: pd.DataFrame, colnames: dict, dropna=False) -> pd.DataFrame:
    '''
    ## Input:
    `df`: `pandas.DataFrame` to transform
    `colnames`: dictionary with keys `['seq', 'charge', 'rawfile', 'intensity']` and values the names of the corresponding columns in `df`
    `dropna`: optional boolean whether to discard peptides that only show up with `intensity` = `nan`
    ## Output:
    Transformed `DataFrame` with one row per `rawfile + sequence`
    '''
    df = df.groupby([colnames[name] for name in ['rawfile', 'seq', 'charge']]).mean(numeric_only=True)

    if not dropna:
        df[colnames['intensity']].fillna(-1, inplace=True)

    idx = df.groupby([colnames['rawfile'], colnames['seq']])[colnames['intensity']].transform('max') == df[colnames['intensity']]
    df = df[idx]
    df = df.reset_index()

    return df

def get_rt_window(df: pd.DataFrame, colnames: dict, plusminus=5) -> pd.DataFrame:
    '''
    Computes for each sequence+rawfile in `df` the min and max 
    retention time and store it in columns `min_rt` and `max_rt`
    '''
    rt_stats = df.groupby([colnames['seq'], colnames['rawfile']])[colnames['rt']].agg(['min', 'max']).reset_index()
    rt_stats = rt_stats.rename({'min': 'min_rt', 'max': 'max_rt'}, axis=1)
    df = pd.merge(df, rt_stats, on=[colnames['seq'], colnames['rawfile']])
    df['min_rt'] -= plusminus
    df['max_rt'] += plusminus
    return df

def get_theoretical_mzs(df: pd.DataFrame, colnames: dict, legal_charges: list) -> pd.DataFrame:
    '''
    Given a dataframe with a `mass` column, compute the theoretical mz values for all `legal_charges`
    '''
    proton_mass = 1.00727646688

    df['mass'] = df[colnames['mz']] * df[colnames['charge']] - df[colnames['charge']] * proton_mass
    
    for z in legal_charges:
        df[f'mz{z}'] = (df['mass'] + z * proton_mass) / z
    
    return df

def split_by(df: pd.DataFrame, colname) -> List[pd.DataFrame]:
    '''
    Split dataframe into chunks grouped by one or multiple columns
    '''
    return [group for _, group in df.groupby(colname)]

def create_pipe(rawfile: str) -> str:
    '''
    Create a pipe (file that purely resides in memory). `rawfile` serves as an identifier.
    Returns the path to the created pipe
    '''
    pipe_path = f"/tmp/input_pipe_{rawfile}"

    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)

    return pipe_path

def write_to_pipe(pipe_path: str, data: str):
    '''
    Write `data` to the pipe in `pipe_path`
    '''
    with open(pipe_path, 'w') as pipe:
        pipe.write(data)

def get_peak_intensities(chunk: pd.DataFrame, output_data: list, legal_charges: list):
    '''
    Given the json output from `ThermoRawFileParser`, get the peak value for each XIC.
    '''
    charges_num = len(legal_charges)

    peak_intensities_list = []

    xic_chunks = [output_data['Content'][i:i + charges_num] for i in range(0, len(output_data['Content']), charges_num)]

    for xic_chunk in xic_chunks:
        chunk_peak_intensities = [auc(xic['RetentionTimes'], xic['Intensities']) for xic in xic_chunk]
        peak_intensities_list.append(chunk_peak_intensities)

    peak_intensities_list_np = np.array(peak_intensities_list).T

    xic_charge_df = pd.DataFrame({f'xic_charge_{z}': peak_intensities_list_np[i] for i, z in enumerate(legal_charges, 0)})
    chunk = pd.concat([chunk, xic_charge_df], axis=1)

    return chunk

def process_chunk(chunk: pd.DataFrame, colnames: dict, legal_charges: list, rawfile_dir: str, mz_tolerance: float, ThermoRawFileParser_exe='ThermoRawFileParser.exe') -> pd.DataFrame:
    '''
    Extract ion chromatograms for different mzs of a `chunk` and add peak intensities
    '''
    table = chunk[[f'mz{i}' for i in legal_charges]].to_dict(orient='records')
    rts = chunk[['min_rt', 'max_rt']].to_dict(orient='records')

    xic_input = [{'mz': v, 'tolerance': mz_tolerance, 'rt_start': rt['min_rt'], 'rt_end': rt['max_rt']} for d, rt in zip(table, rts) for k, v in d.items()]
    print(xic_input)
    xic_input_json = json.dumps(xic_input)

    rawfile_name = chunk[colnames['rawfile']].iloc[0]

    pipe_path = create_pipe(rawfile_name)

    raw_file_path = os.path.join(rawfile_dir, f"{rawfile_name}.raw")

    print(f"Extracting XICs for {rawfile_name}")

    try:
        writer_thread = threading.Thread(target=write_to_pipe, args=(pipe_path, xic_input_json))
        writer_thread.start()

        cmd = ['bash', '-c', f'''
        module load thermorawfileparser/1.4.3 &&
        mono {ThermoRawFileParser_exe} xic -s -i={raw_file_path} -j={pipe_path} &&
        module unload thermorawfileparser/1.4.3
        ''']
        
        process = subprocess.run(cmd, 
                                stdin=subprocess.PIPE, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                shell=False)
        
        # If the subprocess fails and does not read from
        # the pipe for some reason, read here so that
        # the writer thread terminates
        if process.returncode != 0:
            with open(pipe_path) as f:
                f.read()

        writer_thread.join()
    finally:
        os.remove(pipe_path)
    
    std_out = process.stdout.decode('utf-8')
    std_err = process.stderr.decode('utf-8')

    if process.returncode == 0:
        output_data = json.loads(std_out)
    else:
        print(std_err)
        return pd.DataFrame()

    chunk = get_peak_intensities(chunk, output_data, legal_charges)

    return chunk

def sum_peak_intensities(df: pd.DataFrame, colnames: dict, legal_charges: list) -> pd.DataFrame:
    '''
    Group by sequence and aggregate by summing up peak intensities. For other columns compute the mean
    '''
    peak_intensity_cols = [f'xic_charge_{i}' for i in legal_charges]
    other_cols = [col for col in df.columns if col not in peak_intensity_cols]

    agg_dict = {col: 'sum' for col in peak_intensity_cols}
    agg_dict.update({col: 'mean' for col in other_cols if pd.api.types.is_numeric_dtype(df[col])})

    df = df.groupby(colnames['seq']).agg(agg_dict)

    return df

def sum_normalize_xic_vec(df: pd.DataFrame, legal_charges: list) -> pd.DataFrame:
    '''
    For `df` containing columns with the sum peak intensities, sum normalise to get a distribution for each row / sequence
    '''
    xic_charge_cols = [f'xic_charge_{i}' for i in legal_charges]

    df[xic_charge_cols] = df[xic_charge_cols].div(df[xic_charge_cols].sum(numeric_only=True, axis=1), axis=0)

    return df
