import os 
import glob
import numpy as np
import pandas as pd
from collections import defaultdict


def latest_checkpoint(model):
    checkpoint_dir = f'model/{model}'
    model_name = model
    checkpoint_pattern = os.path.join(checkpoint_dir, f"{model_name}-*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    # 최신 체크포인트 파일 선택
    checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
    return checkpoint_path

def create_ground_truth(interaction):
    # 데이터 추출
    user_ids = interaction['user_id'].detach().numpy() 
    item_ids = interaction['item_id'].detach().numpy() 
    
    ground_truth = defaultdict(set)
    
    for user, item in zip(user_ids, item_ids):
        ground_truth[user].add(item)  # 사용자 ID를 키로, 아이템 ID를 값으로 추가
    
    return dict(ground_truth)  # defaultdict을 일반 dict로 변환

def calculate_recall_at_k(final_ensemble_topk, ground_truth, k=10):
    # 추천 결과에서 상위 K개 아이템 선택
    topk_recommendations = (
        final_ensemble_topk[final_ensemble_topk['rank'] < k]
        .groupby('user')['item']
        .apply(list)
    )
    
    recalls = []
    
    for user, recommended_items in topk_recommendations.items():
        relevant_items = ground_truth.get(user, set())  # ground truth에서 사용자별 실제 선호 아이템 가져오기
        if len(relevant_items) == 0:  # ground truth가 비어 있는 경우 recall = 0
            continue
        recall = len(set(recommended_items) & relevant_items) / len(relevant_items)
        recalls.append(recall)
    
    return np.mean(recalls) if recalls else 0.0

def min_max_scale_excluding_inf(df):
    # NumPy 배열로 변환
    data = df.to_numpy()
    
    # 유효한 값(-inf, inf 제외)을 기준으로 min, max 계산
    valid_mask = np.isfinite(data)  # 유효한 값만 True
    col_min = np.where(valid_mask, data, np.inf).min(axis=0)  # 유효한 값 중 최소값
    col_max = np.where(valid_mask, data, -np.inf).max(axis=0)  # 유효한 값 중 최대값

    # 스케일링: (x - min) / (max - min)
    range_ = col_max - col_min  # 각 열의 범위
    scaled_data = (data - col_min) / range_  # broadcasting
    
    # 유효하지 않은 값(-inf, inf)은 원래 값 유지
    scaled_data[~valid_mask] = data[~valid_mask]

    # 결과를 데이터프레임으로 반환
    return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

def get_top_k_indices(dataframe, k):
    values = dataframe.to_numpy()  # NumPy 배열로 변환
    topk_indices = np.argpartition(-values, k, axis=1)[:, :k]  # 상위 k개 위치 추출
    sorted_topk_indices = np.argsort(-values[np.arange(values.shape[0])[:, None], topk_indices], axis=1)  # 정렬
    topk_indices_sorted = topk_indices[np.arange(values.shape[0])[:, None], sorted_topk_indices]
    return topk_indices_sorted

def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    import torch

    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data
