import argparse
import os

from bioencoder import config

def configure(**kwargs):

    if kwargs.get("root_dir"):
        config.root_dir = kwargs.get("root_dir")
    if kwargs.get("run_name"):
        config.run_name = kwargs.get("run_name")
        
    if not config.root_dir.__class__.__name__ == "NoneType":
        print("BioEncoder config:")
        { print(f"- {k}: {v}") for k,v in vars(config).items() if not k.startswith('__') }
        print(f"Given the current WD ({os.getcwd()}), BioEncoder root directory will be:")
        print(f"- {os.path.join(os.getcwd(), config.root_dir, config.run_name)}")
    
if __name__ in {"__main__","bioencoder.scripts.configure"}:
       
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
    
