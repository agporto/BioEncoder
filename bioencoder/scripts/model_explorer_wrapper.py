# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:03:05 2023

@author: mluerig
"""

import argparse
import os 
import subprocess  

def model_explorer_wrapper(config_path):
    
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_explorer.py")
    process = ["streamlit", "run", script_path , "--", "--config-path", config_path]
    subprocess.run(process)
    


    
def cli():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
    )
    args = parser.parse_args()
        
    model_explorer_wrapper(args.config_path)
    
    
    
if __name__ == "__main__":

    cli()
    

