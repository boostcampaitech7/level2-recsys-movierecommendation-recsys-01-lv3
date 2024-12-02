# src/train_and_evaluate.py

import pandas as pd
import os
import glob
from tqdm import tqdm
import time
from recbole.utils.case_study import full_sort_topk
from recbole.quick_start import load_data_and_model
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.utils import get_model, get_trainer
import warnings
warnings.filterwarnings('ignore')
    
def run(data_type, model_type, model):
    
    data_type = str(data_type)
    model_type = str(model_type)
    model = str(model)
    
    # configurations initialization
    config_file = [f"configs/{model_type.lower()}/{model.lower()}_config.yaml"]
    
    config = Config(model=model, dataset=data_type, config_file_list=config_file)
            
    # init random seed
    init_seed(config['seed'], config['reproducibility'])
    
    # logger initialization
    init_logger(config)
    logger = getLogger()
    
    # write config info into log
    logger.info(config) 
    
    # dataset creating and filtering
    dataset = create_dataset(config)
    # logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # model loading and initialization
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=True, show_progress=config["show_progress"])