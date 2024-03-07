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

from bioencoder import utils

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
    

    Parameters
    ----------
    data : TYPE, optional
        DESCRIPTION. The default is False.
    logs : TYPE, optional
        DESCRIPTION. The default is True.
    plots : TYPE, optional
        DESCRIPTION. The default is True.
    runs : TYPE, optional
        DESCRIPTION. The default is True.
    weights : TYPE, optional
        DESCRIPTION. The default is False.
    export_dir : TYPE, optional
        DESCRIPTION. The default is None.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    ## load bioencoer config
    config = utils.load_config(kwargs.get("bioencoder_config_path"))
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
    parser.add_argument(
        "--data",
        type=str,
    )
    parser.add_argument(
        "--logs",
        type=str,
    )
    parser.add_argument(
        "--plots",
        type=str,
    )
    parser.add_argument(
        "--runs",
        type=str,
    )
    parser.add_argument(
        "--weights",
        type=str,
    )
    parser.add_argument(
        "--export-dir",
        type=str,
    )
    args = parser.parse_args()
    
    archive(args.config_path)


if __name__ == "__main__":
    
    cli()

