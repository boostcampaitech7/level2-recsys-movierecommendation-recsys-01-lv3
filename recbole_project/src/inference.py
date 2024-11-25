import pandas as pd
import os
import glob
from tqdm import tqdm
import torch
from recbole.utils.case_study import full_sort_topk
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed
from recbole.utils import get_model, get_trainer
import warnings
warnings.filterwarnings('ignore')

def inference(config): 
    
    # Inference
    sample_submission = pd.read_csv(os.path.join(config['eval_path'], 'sample_submission.csv'))
    checkpoint_dir = os.path.join(config['checkpoint_dir'], config['model'])
    model_name = config['model']
    checkpoint_pattern = os.path.join(checkpoint_dir, f"{model_name}-*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    if not checkpoint_files:
        print(f"Checkpoint files not found in {checkpoint_dir} with pattern {checkpoint_pattern}")

    # 최신 체크포인트 파일 선택
    checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
    print(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    general_hyper_params = checkpoint["config"]
    general_hyper_params['eval_args'] = {
        'split': {'RS': [1, 0, 0]},
        'order': 'TO',
        'group_by': 'user',
        'mode': {'valid': 'full', 'test': 'full'}
    } 
    
    # 전체 학습 데이터셋 구성
    print("현재: 전체 데이터셋으로 구성")
    full_dataset = create_dataset(general_hyper_params)
    full_train_data, _, _ = data_preparation(general_hyper_params, full_dataset)
    dataset = create_dataset(config)
    _, valid_data, _ = data_preparation(config, dataset)
    
    init_seed(config["seed"], config["reproducibility"])
    best_model = get_model(config["model"])(config, full_train_data._dataset).to(config["device"])
    best_model.to(config['device'])
    
    # Early stopping epoch 가져오기
    if checkpoint.get('epoch', config['epochs']) == 0:
        early_stop_epoch = 1
    else: 
        early_stop_epoch = checkpoint.get('epoch', config['epochs'])
    print(f"Early stopping epoch: {early_stop_epoch}")
    
    # 전체 데이터셋으로 재학습
    print("현재: 전체 데이터셋으로 재학습 시작")

    # 트레이너 초기화
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, best_model)

    # 전체 데이터로 재학습
    for epoch in range(early_stop_epoch):
        train_loss = trainer._train_epoch(full_train_data, epoch_idx=epoch, show_progress=False)
        print(f"Epoch {epoch+1}/{early_stop_epoch}, Train Loss: {train_loss:.4f}")
    
    # Inference 시작
    sample_submission.columns = ['user_id', 'item_id']
    test_users = sample_submission['user_id'].unique().tolist()
    test_users = [str(user) for user in test_users]
    uid_series = full_dataset.token2id(full_dataset.uid_field, test_users)

    batch_size = 256
    recommended_df = pd.DataFrame(columns=['user', 'item'])
    for i in tqdm(range(0, len(uid_series), batch_size)):
        batch_indices = uid_series[i:i+batch_size]
        batch_users = test_users[i:i+batch_size]

        topk_iid_list_batch = full_sort_topk(batch_indices, best_model, valid_data, k=10, device=config['device'])
        last_topk_iid_list = topk_iid_list_batch.indices
        recommended_item_list = full_dataset.id2token(full_dataset.iid_field, last_topk_iid_list.cpu()).tolist()
        temp_df = pd.DataFrame({'user': batch_users, 'item': recommended_item_list})
        recommended_df = pd.concat([recommended_df, temp_df], ignore_index=True)

    recommended_df = recommended_df.explode('item').reset_index(drop=True)
    output_dir = os.path.join(config['output_data_path'], config['model'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if early_stop_epoch == config['epochs']:
        recommended_df.to_csv(
            os.path.join(output_dir, f"output_{checkpoint_path.split('/')[-1][:-4]}_epoch{early_stop_epoch}.csv"), 
            index=False
        )
    else:
        recommended_df.to_csv(
            os.path.join(output_dir, f"output_{checkpoint_path.split('/')[-1][:-4]}_early_stop{early_stop_epoch}.csv"), 
            index=False
        )