# src/train_and_evaluate.py
import sys
import os
import argparse
from src.mlflow_train_and_evaluate import run 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", "-dt", type=str, default="dataset", help="name of data type (e.g., dataset, dataset_context, small)")
    parser.add_argument("--model_type", "-mt", type=str, default="general", help="name of model type (e.g., general, context, sequential)")
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    args, _ = parser.parse_known_args()
    
    run(args.data_type, args.model_type, args.model)    