# src/mlflow_train_and_evaluate.py

import mlflow
import mlflow.pytorch
import pandas as pd
import os
import glob
import time
from tqdm import tqdm
from recbole.utils.case_study import full_sort_topk
from recbole.quick_start import load_data_and_model
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.utils import get_model, get_trainer, get_local_time
import warnings
warnings.filterwarnings('ignore')
    
def run(data_type, model_type, model):
    
    data_type = str(data_type)
    model_type = str(model_type)
    model = str(model)
    
    # configurations initialization
    config_file = [f"configs/{model_type.lower()}/{model.lower()}_config.yaml"]
    
    config = Config(model=model, dataset=data_type, config_file_list=config_file)
    
    # MLflow experiment 초기화
    mlflow.set_tracking_uri("http://10.28.224.95:30696")
    mlflow.set_experiment("RecBole_model_ALL")
    
    with mlflow.start_run(run_name=f"{model_type}_{model}") as run:

        mlflow.log_params({key: config[key] for key in config.final_config_dict.keys()})

        # init random seed
        init_seed(config['seed'], config['reproducibility'])

        # logger initialization
        init_logger(config)
        logger = getLogger()
    
        # write config info into log
        # logger.info(config)

        # dataset creating and filtering
        print("현재: 데이터 구성")
        dataset = create_dataset(config)
        # logger.info(dataset)

        # dataset splitting
        train_data, valid_data, test_data = data_preparation(config, dataset)
        
        # model loading and initialization
        print("현재: 모델 구성")
        model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
        # logger.info(model)

        # trainer loading and initialization
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

        current_time = get_local_time()
        
        # 모델 저장 경로 초기화
        first_checkpoint_path = os.path.join(config["checkpoint_dir"], config['model'], f"{config['model']}-{current_time}.pth")
        model_dir = os.path.join(config["checkpoint_dir"], config['model'])
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # model training
        best_valid_score, best_valid_result = None, None
        best_epoch = None
        early_stop_epoch = None
        for epoch in range(config['epochs']):
            # 1. Train Epoch 실행
            train_loss = trainer._train_epoch(train_data, epoch_idx=epoch, show_progress=True)
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            # 현재 에폭이 eval_step의 배수인지 확인
            if (epoch + 1) % config['eval_step'] == 0:
                # 2. Validation 실행
                valid_score, valid_result = trainer._valid_epoch(valid_data, show_progress=True)
                for metric, value in valid_result.items():
                    sanitized_metric = metric.replace("@", "_")  # '@'를 '_'로 변경
                    mlflow.log_metric(f"{sanitized_metric}", value, step=epoch)

                metric_log = ""
                for metric in config['metrics']:
                    for topk in config['topk']:
                        # Validation 결과 Key 생성
                        metric_name = f"{metric}@{topk}"
                        metric_value = valid_result.get(metric_name.lower())  # Key는 소문자로 저장됨
                        # 결과 출력 문자열 생성
                        if metric_value is not None:
                            metric_log += f"{metric_name}: {metric_value:.4f}, "

                # 마지막 쉼표 제거 및 출력
                metric_log = metric_log.rstrip(", ")
                print(f"Epoch {epoch} Validation Results: {metric_log}")

                # 3. 최적 모델 갱신 및 Early Stopping 체크
                if best_valid_score is None or valid_score > best_valid_score:
                    best_valid_score = valid_score
                    best_valid_result = valid_result
                    best_epoch = epoch
                    stopping_counter = 0  # 개선이 되었으므로 카운터 리셋
                else:
                    stopping_counter += 1  # 개선이 없었으므로 카운터 증가       
                    
                # 4. Early Stopping 조건 체크
                if stopping_counter >= config['stopping_step']:
                    early_stop_epoch = epoch + 1  # Early stopping이 발생한 에폭 번호 저장
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break 
     
        # Early stopping이 발생하지 않았다면, 전체 에폭 수 저장
        if early_stop_epoch is None:
            early_stop_epoch = config['epochs'] 
        trainer._save_checkpoint(best_epoch, saved_model_file=first_checkpoint_path)
        # early_stop_epoch를 MLflow에 로그 기록
        mlflow.log_param("early_stop_epoch", early_stop_epoch) 
        mlflow.log_param("best recall_10", best_valid_score)
        # MLflow로 모델 로깅
        # mlflow.pytorch.log_model(model, "best_model")  # Best 모델 저장 