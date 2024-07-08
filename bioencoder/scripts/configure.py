#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% imports

import argparse
import os

from bioencoder import config, utils

#%% function

def configure(
        root_dir, 
        run_name,
        create=False,
        **kwargs
        ):
    """
    This function allows specifying a root directory and run name bioencoder for 
    the current sessions. 

    Parameters
    ----------
    root_dir : str
        Path to the root directory where settings are to be saved. If the directory does not exist,
        it will be created if `create` is set to True; otherwise, an error message will be printed.
    run_name : str
        Name of the run to be saved in the configuration.
    create : bool, optional
        If set to True, the specified root directory will be created if it does not exist.
        Defaults to False.

    Examples
    --------
    To create or update a configuration with a specific root directory and run name:

        configure(root_dir="/path/to/data", run_name="experiment1")

    This will update the global configuration with the specified `root_dir` and `run_name`,
    and print out the full path where the run directory will be located based on the current
    working directory.

    """
    
    if not os.path.isdir(root_dir) and not create:
        print(f"{root_dir} does not exist (use create = True)")
        return
    elif not os.path.isdir(root_dir) and create:
        os.makedirs(root_dir)
        print(f"created root_dir: {root_dir}")
    else:
        print(f"found root_dir: {root_dir}")
        
    config.root_dir = root_dir
    config.run_name = run_name
                
    print(f"Given your Python WD ({os.getcwd()}), the current BioEncoder run directory will be:")
    print(f"- {os.path.join(os.getcwd(), config.root_dir, config.run_name)}")       
    
    utils.save_yaml({"root_dir": config.root_dir,
                     "run_name": config.run_name}, os.path.expanduser("~/.bioencoder.yaml"))

def cli():  

    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--create", type=bool, default=False)
    args = parser.parse_args()

    configure(**vars(args))
    
if __name__ == "__main__":
    cli()

