# src/train.py

import argparse
import sys
from recbole.quick_start import run_recbole

def main():
    parser = argparse.ArgumentParser(description="Train a RecBole model.")
    parser.add_argument('--config_file', type=str, required=True, help='Path to the config YAML file.')
    args = parser.parse_args()
    
    # sys.argv 초기화하여 RecBole이 추가 인자를 인식하지 못하게 함
    sys.argv = [sys.argv[0]]
    
    # RecBole 실행
    run_recbole(config_file_list=[args.config_file], config_dict=None, saved=True)

if __name__ == '__main__':
    main()
