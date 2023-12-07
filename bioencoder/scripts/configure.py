import argparse
import os
import yaml



def configure(config_path=None, **kwargs):
    
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
    
