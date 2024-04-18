#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import yaml

#%%

def configure(
        config_path=None, 
        **kwargs
        ):
    """
    Configures and saves settings for an application, storing them in a YAML file.
    This function allows specifying a configuration file path and updates or creates
    configuration settings based on the provided keyword arguments. If no configuration
    file is specified, it defaults to a '.bioencoder' file in the user's home directory.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file where settings are to be saved. If not specified,
        defaults to ~/.bioencoder. If the file does not exist, it will be created.

    Raises
    ------
    IOError
        If there are issues reading from or writing to the configuration file.

    Examples
    --------
    To create or update a configuration with a specific root directory and run name:

        configure(root_dir="/path/to/data", run_name="experiment1")

    This will update the default configuration file with the specified `root_dir` and `run_name`,
    and print out the full path where the run directory will be located based on the current
    working directory.

    """
    
    if not config_path:
        config_path = os.path.join(os.path.expanduser("~"), ".bioencoder")
        
    if not os.path.isfile(config_path):
        config = {}
    else:
        with open(config_path, "r") as file:
            config = yaml.full_load(file)
        
    if kwargs.get("root_dir"):
        config["root_dir"] = kwargs.get("root_dir")
        config["root_dir_abs"] = os.path.abspath(kwargs.get("root_dir"))
    if kwargs.get("run_name"):
        config["run_name"] = kwargs.get("run_name")

    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
                
    if not config["root_dir"].__class__.__name__ == "NoneType":
        print("BioEncoder config:")
        {print(f"- {k}: {v}") for k,v in config.items()}
        print(f"Given your Python WD ({os.getcwd()}), the current BioEncoder run directory will be:")
        print(f"- {os.path.join(os.getcwd(), config['root_dir'], config['run_name'])}")
    else:
        print("No root-dir or run-name provided - doing nothing.")
        
    if not os.path.isdir(config["root_dir_abs"]):
        rd = config["root_dir_abs"]
        print(f"{rd} does not exist but will be created when adding data!")
        
        

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    configure(**vars(args))
    
    
    
if __name__ == "__main__":
    cli()
    
