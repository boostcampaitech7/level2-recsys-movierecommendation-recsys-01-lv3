# run.py

import os
import pandas as pd
import torch
from recbole.quick_start import run
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed, get_model, set_color
from recbole.data.interaction import Interaction
from recbole.config import Config
from tqdm import tqdm
import datetime
import pytz
import glob
import warnings
warnings.filterwarnings('ignore')

def generate_recommendations(config, model, dataset, device, eval_path, output_path, k=10):
    """
    sample_submission.csv에 있는 사용자들에 대해 상위 k개의 아이템을 추천하고, 결과를 CSV 파일로 저장합니다.

    Args:
        config (dict): RecBole 설정.
        model (torch.nn.Module): 학습된 모델.
        dataset (Dataset): 데이터셋 객체.
        device (torch.device): 사용 중인 디바이스.
        eval_path (str): sample_submission.csv 파일이 위치한 경로.
        output_path (str): 추천 결과를 저장할 디렉토리 경로.
        k (int): 추천할 아이템의 개수.
    """
    # 테스트에 사용할 사용자 목록 로드
    sample_submission_path = os.path.join(eval_path, 'sample_submission.csv')
    if not os.path.exists(sample_submission_path):
        raise FileNotFoundError(f"'{sample_submission_path}' 파일이 존재하지 않습니다.")
    sample_submission = pd.read_csv(sample_submission_path)
    sample_submission.rename(columns={'user': 'user_id'}, inplace=True)
    sample_submission['user_id'] = sample_submission['user_id'].astype(str)
    
    # 테스트 사용자 ID 리스트
    test_users = sample_submission['user_id'].tolist()
    uid_series = dataset.token2id(dataset.uid_field, test_users)
    
    # 추천 결과 저장용 리스트
    recommendations = []

    # 아이템 메타데이터 로드 (모든 필드 포함)
    item_meta = dataset.get_item_feature().to(config['device'])
    
    # 배치 크기 설정
    batch_size = 128
    
    # 아이템 메타데이터 로드 (필요한 경우)
    # 여기서는 모든 아이템에 대해 추천하므로, item_num만 사용
    item_num = dataset.item_num
    
    # 모든 사용자에 대해 점수 예측 수행
    model.eval()
    with torch.no_grad():
        for start_idx in tqdm(range(0, len(uid_series), batch_size), desc="Predicting for users in batches"):
            end_idx = min(start_idx + batch_size, len(uid_series))
            batch_uids = uid_series[start_idx:end_idx]
            batch_size_actual = len(batch_uids)
    
            # 배치 사용자 ID와 모든 아이템 ID 생성
            user_ids = torch.tensor(batch_uids, device=device).repeat_interleave(item_num)
            item_ids = torch.arange(dataset.item_num, device=device).repeat(batch_size_actual)

            # 아이템 메타데이터를 반복 및 확장
            item_meta_repeated = {}
            for field in item_meta.columns:
                data = item_meta[field]
                if len(data.shape) == 1:  # 1차원 필드
                    item_meta_repeated[field] = data.repeat(len(batch_uids))
                elif len(data.shape) == 2:  # 2차원 시퀀스 필드
                    item_meta_repeated[field] = data.repeat(len(batch_uids), 1)
                else:
                    raise ValueError(f"Unexpected dimension for field {field}: {data.shape}")
            
            # 사용자-아이템 상호작용 생성
            batch_interactions = Interaction({
                config['USER_ID_FIELD']: user_ids,
                config['ITEM_ID_FIELD']: item_ids,
                **item_meta_repeated,  # 모든 필드 추가
            })
    
            # 점수 계산
            scores = model.predict(batch_interactions).view(batch_size_actual, -1)
    
            # 각 사용자에 대해 상위 k개 아이템 선택
            # topk_scores, topk_items = torch.topk(scores, k=k, dim=1)
            topk_items = torch.topk(scores, k=10, dim=1).indices.cpu().numpy()
    
            # 추천 결과 저장
            for idx, uid in enumerate(batch_uids):
                user_id = dataset.id2token(dataset.uid_field, uid)
                item_ids = dataset.id2token(dataset.iid_field, topk_items[idx])

                for item_id in item_ids:
                    recommendations.append({
                        'user_id': user_id,
                        'item': item_id
                    })
    
    # 추천 결과를 DataFrame으로 변환
    recommendations_df = pd.DataFrame(recommendations)
    
    # 현재 시간으로 파일명 생성
    korea_timezone = pytz.timezone('Asia/Seoul')
    current_time = datetime.datetime.now(korea_timezone).strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_path, f"output_{model.__class__.__name__}_{current_time}.csv")
    
    # 추천 결과 저장
    recommendations_df.to_csv(output_file, index=False)
    print(f"추천 결과가 {output_file}에 저장되었습니다.")


def main():
    # RecBole 설정 파일 경로
    config_files = ["recbole/properties/dataset/movie.yaml"]
    
    # 모델과 데이터셋 설정
    result = run(
        model='EASE',
        dataset='movie',
        config_file_list=config_files,
    )
    
    # 학습 및 평가 결과 출력
    print("Training and Evaluation Results:")
    print(result)
    
    # 학습된 모델과 데이터셋 로드
    config = Config(model='EASE', dataset='movie', config_file_list=config_files)
    dataset = create_dataset(config)
    # train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # 모델 클래스와 객체 생성
    model_class = get_model(config['model'])  # 'WideDeep' 모델 클래스 가져오기
    model = model_class(config, dataset)       # 모델 객체 생성
    
    # 모델 저장 디렉토리 설정
    save_dir = config['save_dir']
    # dataset_name = config['dataset']
    model_name = config['model']

    model_save_path_pattern = os.path.join(save_dir, f"{model_name}-*.pth")
    model_save_files = glob.glob(model_save_path_pattern)
    # 가장 최근에 저장된 모델 파일 선택
    best_model_path = max(model_save_files, key=os.path.getctime)
    checkpoint = torch.load(best_model_path, map_location=config.device)

    # 'state_dict' 키가 존재하는지 확인
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise KeyError(f"'state_dict' 키가 {best_model_path}에 존재하지 않습니다.")

    # 모델에 state_dict 로드
    model.load_state_dict(state_dict)
    model.to(config.device)
    
    # 평가에 사용할 경로 설정
    eval_path = config['eval_path']  
    output_path = "./dataset/data/output/"
    
    # 출력 디렉토리 생성
    os.makedirs(output_path, exist_ok=True)
    torch.cuda.empty_cache()
    # 추천 생성 및 저장
    generate_recommendations(config, model, dataset, config.device, eval_path, output_path, k=10)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()