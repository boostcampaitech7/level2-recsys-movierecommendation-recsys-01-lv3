# src/mlflow_train_and_evaluate.py

import mlflow
import mlflow.pytorch
import pandas as pd
import os
import glob
from tqdm import tqdm
from recbole.utils.case_study import full_sort_topk
from recbole.quick_start import load_data_and_model
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.utils import get_model, get_trainer, get_local_time
import warnings
warnings.filterwarnings('ignore')
    
def run(model_type, model):
    
    # configurations initialization
    config_file = [f"configs/{model_type.lower()}/{model.lower()}_config.yaml"]
    if model_type.lower() == 'context': 
        config = Config(model=model, dataset='dataset_context', config_file_list=config_file)
    else:
        config = Config(model=model, dataset='dataset', config_file_list=config_file)
    
    # MLflow experiment 초기화
    mlflow.set_tracking_uri("http://10.28.224.95:30696")
    mlflow.set_experiment("rebole_project_experiment")
    with mlflow.start_run(run_name=f"{model_type}_{model}") as run:
        mlflow.log_params({key: config[key] for key in config.final_config_dict.keys()})
        
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

        current_time = get_local_time()
        # 모델 저장 경로 초기화
        first_checkpoint_path = os.path.join(config["checkpoint_dir"], config['model'], f"{config['model']}-{current_time}.pth")
        model_dir = os.path.join(config["checkpoint_dir"], config['model'])
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # model training
        best_valid_score, best_valid_result = None, None
        
        for epoch in range(config['epochs']):
            # 1. Train Epoch 실행
            train_loss = trainer._train_epoch(train_data, epoch_idx=epoch, show_progress=True)
            mlflow.log_metric("train_loss", train_loss, step=epoch)

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

            # 3. 최적 모델 갱신
            if best_valid_score is None or valid_score > best_valid_score:
                best_valid_score = valid_score
                trainer._save_checkpoint(epoch, saved_model_file=first_checkpoint_path)                

        # MLflow로 모델 로깅
        mlflow.pytorch.log_model(model, "best_model")  # Best 모델 저장   
    
        # Inference
        sample_submission = pd.read_csv(os.path.join(config['eval_path'], 'sample_submission.csv'))
        test_data = sample_submission.copy()
        checkpoint_dir = os.path.join(config['checkpoint_dir'], config['model'])
        model_name = config['model']
        checkpoint_pattern = os.path.join(checkpoint_dir, f"{model_name}-*.pth")
        checkpoint_files = glob.glob(checkpoint_pattern)
        if not checkpoint_files:
            print(f"Checkpoint files not found in {checkpoint_dir} with pattern {checkpoint_pattern}")

        # 최신 체크포인트 파일 선택
        checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
        print(f"Loading model from {checkpoint_path}")

        general_hyper_params, best_model, dataset, train_data_loader, valid_data_loader, test_data_loader = load_data_and_model(model_file=checkpoint_path)
        best_model.to(config['device'])

        test_data.columns = ['user_id', 'item_id']
        test_users = test_data['user_id'].unique().tolist()
        test_users = [str(user) for user in test_users]
        uid_series = dataset.token2id(dataset.uid_field, test_users)

        batch_size = 256

        recommended_df = pd.DataFrame(columns=['user', 'item'])
        for i in tqdm(range(0, len(uid_series), batch_size)):
            batch_indices = uid_series[i:i+batch_size]
            batch_users = test_users[i:i+batch_size]

            topk_iid_list_batch = full_sort_topk(batch_indices, best_model, valid_data_loader, k=10, device=config['device'])
            last_topk_iid_list = topk_iid_list_batch.indices
            recommended_item_list = dataset.id2token(dataset.iid_field, last_topk_iid_list.cpu()).tolist()
            temp_df = pd.DataFrame({'user': batch_users, 'item': recommended_item_list})
            recommended_df = pd.concat([recommended_df, temp_df], ignore_index=True)

        recommended_df = recommended_df.explode('item').reset_index(drop=True)
        output_dir = os.path.join(config['output_data_path'], config['model'])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        recommended_df.to_csv(
            os.path.join(output_dir, f"output_E{config['epochs']}_{checkpoint_path.split('/')[-1][:-4]}.csv"), 
            index=False
        )
        
        # 추천 결과 MLflow에 저장
        mlflow.log_artifact(
            os.path.join(output_dir, f"output_E{config['epochs']}_{checkpoint_path.split('/')[-1][:-4]}.csv")
        )
        