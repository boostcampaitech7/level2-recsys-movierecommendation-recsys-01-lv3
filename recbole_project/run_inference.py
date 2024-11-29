import sys
import os
from recbole.config import Config
import argparse
from src.inference import inference

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", "-mt", type=str, default="general", help="name of model type (e.g., general, context, sequential)")
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    args, _ = parser.parse_known_args()
    
    
    model = str(args.model)
    model_type = str(args.model_type)
    config_file = [f"configs/{model_type.lower()}/{model.lower()}_config.yaml"]
    config = Config(model=model, dataset=None, config_file_list=config_file)
    
    inference(config)