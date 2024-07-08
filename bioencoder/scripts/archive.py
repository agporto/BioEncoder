#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Mar  7 11:22:48 2024

@author: mlurig@ad.ufl.edu
"""
import argparse
from datetime import datetime
import os
import zipfile

from bioencoder import config, utils

#%%

def archive(
    data=False,
    logs=True,
    plots=True,
    runs=True,
    weights=False,
    export_dir=None,
    **kwargs,
):
    """
    Creates a zip archive containing selected components of a BioEncoder project, such as
    logs, plots, runs, and model weights. This function is configurable to selectively
    include these components based on boolean flags passed as parameters.

    Parameters
    ----------
    data : bool, optional
        If True, includes the 'data' directory corresponding to the current run in the archive.
        Default is False.
    logs : bool, optional
        If True, includes the 'logs' directory corresponding to the current run in the archive.
        Default is True.
    plots : bool, optional
        If True, includes the 'plots' directory in the archive.
        Default is True.
    runs : bool, optional
        If True, includes the 'runs' directory corresponding to the current run in the archive.
        Default is True.
    weights : bool, optional
        If True, includes the 'weights' directory corresponding to the current run in the archive.
        Default is False.
    export_dir : str, optional
        Path to the directory where the archive will be saved. If not specified, the archive will be
        saved in a default 'archive' directory within the root directory of the configuration.
        Default is None.

    Raises
    ------
    FileNotFoundError
        If any specified directory does not exist and is expected to be archived.
    OSError
        If there are issues writing the zip file to disk, perhaps due to permissions or disk space.

    Examples
    --------
    To archive logs, plots, and runs directories:
        bioencoder.archive(logs=True, plots=True, runs=True, weights=False)

    To create a full archive with a custom export directory:
        bioencoder.archive(data=True, logs=True, plots=True, runs=True, weights=True, export_dir='/path/to/custom/dir')

    Notes
    -----
    This function is part of a larger system intended for managing machine learning experiments.
    It assumes that directory paths are set up according to the 'bioencoder' project structure.
    """
    
    ## load bioencoer config
    root_dir = config.root_dir
    run_name = config.run_name
    
    if not export_dir:
        export_dir = os.path.join(root_dir, "archive")
        os.makedirs(export_dir,exist_ok=True)
    
    ## construct save path
    save_suffix = datetime.today().strftime( "%Y-%m-%d_%H:%M:%S")
    save_path = os.path.join(export_dir, run_name + "_" + save_suffix + ".zip")
    print(f"saving to {save_path}:")

    ## start zip process
    with zipfile.ZipFile(save_path, "w", zipfile.ZIP_DEFLATED) as zip:
        
        if data:
            print("- archiving data")
            utils.zip_directory(
                os.path.join(root_dir, "data", run_name), root_dir, zip)    
            
        if logs:
            print("- archiving logs")
            utils.zip_directory(os.path.join(root_dir, "logs", run_name), root_dir, zip)    

        if plots:
            print("- archiving plots")
            utils.zip_directory(os.path.join(root_dir, "plots"), root_dir, zip)   
            
        if runs:
            print("- archiving runs")
            utils.zip_directory(os.path.join(root_dir, "runs", run_name), root_dir, zip)    
            
        if weights:
            print("- archiving weights")
            utils.zip_directory(os.path.join(root_dir, "weights", run_name), root_dir, zip)    
            
def cli():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=bool, default=False, help="Include the 'data' directory in the archive (default: False)")
    parser.add_argument("--logs", type=bool, default=True, help="Include the 'logs' directory in the archive (default: True)")
    parser.add_argument("--plots", type=bool, default=True, help="Include the 'plots' directory in the archive (default: True)")
    parser.add_argument("--runs", type=bool, default=True, help="Include the 'runs' directory in the archive (default: True)")
    parser.add_argument("--weights", type=bool, default=False, help="Include the 'weights' directory in the archive (default: False)")
    parser.add_argument("--export-dir", type=str, help="Directory to save the zipped archive")
    args = parser.parse_args()

    archive_cli = utils.restore_config(archive)
    archive_cli(
        data=args.data,
        logs=args.logs,
        plots=args.plots,
        runs=args.runs,
        weights=args.weights,
        export_dir=args.export_dir
    )

    
if __name__ == "__main__":
    
    cli()

