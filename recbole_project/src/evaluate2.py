# src/evaluate.py

import argparse
import sys
from recbole.quick_start import load_data_and_model
from recbole.config import Config
from recbole.utils import init_seed
import pandas as pd
from recbole.data.interaction import Interaction
import torch
from tqdm import tqdm
import os
import glob

def main():
    parser = argparse.ArgumentParser(description="Evaluate a RecBole model.")
    parser.add_argument('--config_file', type=str, required=True, help='Path to the config YAML file.')
    args = parser.parse_args()
    
    # sys.argv 초기화하여 RecBole이 추가 인자를 인식하지 못하게 함
    sys.argv = [sys.argv[0]]
    
    # Config 설정
    config = Config(model=None, config_file_list=[args.config_file])
    
    # 시드 초기화
    init_seed(config['seed'], config['reproducibility'])
   
    # 학습된 모델 로드
    # Checkpoint 디렉토리에서 최신 체크포인트 파일 찾기
    checkpoint_dir = config['checkpoint_dir']
    model_name = config['model']
    checkpoint_pattern = os.path.join(checkpoint_dir, f"{model_name}*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        print(f"Checkpoint files not found in {checkpoint_dir} with pattern {checkpoint_pattern}")
        return
    
    # 최신 체크포인트 파일 선택
    checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
    print(f"Loading model from {checkpoint_path}")
    
    # 데이터 및 모델 로드 (model_file 지정)
    general_hyper_params, model, dataset, train_data_loader, valid_data_loader, test_data_loader = load_data_and_model(model_file=checkpoint_path)

    # 모델은 이미 load_data_and_model에서 로드되었으므로 별도로 로드할 필요 없음
    model.to(config['device'])
    
    # 테스트 사용자 목록 로드
    sample_submission = pd.read_csv('./data/raw/eval/sample_submission.csv')
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
    batch_size = 256
    
    # 모든 사용자에 대해 점수 예측 수행
    model.eval()
    with torch.no_grad():
        for start_idx in tqdm(range(0, len(uid_series), batch_size), desc="Predicting for users in batches"):
            end_idx = min(start_idx + batch_size, len(uid_series))
            batch_uids = uid_series[start_idx:end_idx]

            # 배치 사용자 ID와 모든 아이템 메타데이터 결합
            user_ids = torch.tensor(batch_uids, device=config['device']).repeat_interleave(dataset.item_num)
            item_ids = torch.arange(dataset.item_num, device=config['device']).repeat(len(batch_uids))

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

            # 배치 내 모든 사용자와 아이템에 대해 점수 계산
            scores = model.predict(batch_interactions).view(len(batch_uids), -1)

            # 각 사용자에 대해 상위 k개 아이템 선택
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
    
    # 결과 저장
    output_file = f'./data/output/output_{model_name}.csv'
    os.makedirs('./data/output/', exist_ok=True)
    recommendations_df.rename(columns={'user_id': 'user'}, inplace=True)
    recommendations_df.to_csv(output_file, index=False)
    print(f"Recommendations saved to {output_file}")

if __name__ == '__main__':
    main()
