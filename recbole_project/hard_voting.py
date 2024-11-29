import argparse
import pandas as pd
import numpy as np
import os
import torch
from itertools import combinations
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def count_similiarity(df1, df2):
    df = pd.concat([df1,df2])
    df = df.groupby(['user','item']).size().reset_index(name='counts')
    df = df[df['counts'] > 1]
    return len(df) / len(df1) * 100

def hard_voting(args):
    model_outputs = torch.load("./data/output/Ensemble/Ensemble_Candidates.pth")['model_recommendations']
    # user와 item 데이터 유형 int로 변환
    for recommended in model_outputs:
        recommended['user'] = recommended['user'].astype(int)
        recommended['item'] = recommended['item'].astype(int)
    
    # weight: 앙상블 모델 가중치    
    weights = [
        # General
        {'model': 'EASE',     'weight': 1,   'index': 0},     # valid: 0.2366
        {'model': 'ADMMSLIM', 'weight': 1,   'index': 1},     # valid: 0.2309
        {'model': 'CDAE',     'weight': 1,   'index': 2},     # valid: 0.2203
        {'model': 'RecVAE',   'weight': 1,   'index': 3},     # valid: 0.2048
        {'model': 'MultiVAE', 'weight': 1,   'index': 4},     # valid: 0.2016
        {'model': 'MultiDAE', 'weight': 1,   'index': 5},     # valid: 0.2013
        {'model': 'LightGCN', 'weight': 1,   'index': 6},     # valid: 0.1896
    
        # Context
        {'model': 'DCNV2',    'weight': 1,   'index': 7},     # valid: 0.1937
        {'model': 'DeepFM',   'weight': 1,   'index': 8},     # valid: 0.1785

        # Sequential
        {'model': 'BERT4Rec', 'weight': 1, 'index': 9},     # valid: 0.1847
        {'model': 'GRU4Rec',  'weight': 1, 'index': 10},    # valid: 0.1437
        {'model': 'GRU4RecF', 'weight': 1, 'index': 11},    # valid: 0.1567
        {'model': 'SASRec',   'weight': 1,   'index': 12},    # valid: 0.1161
    ]
    
    candidate_names = args.models.split(' ')
    candidate_weight = list(map(float, args.weights.split(' ')))
    candidate_indices = [a['index'] for a in weights if a['model'] in candidate_names]
    candidates = [model_outputs[i] for i in candidate_indices]
    
    # 입력된 inference 가중치로 weights의 각 모델 가중치 변경
    for name, new_weight in zip(candidate_names, candidate_weight):
        for item in weights:
            if item['model'] == name:
                item['weight'] = new_weight
    print(f"선택된 모델: {candidate_names}, 가중치: {candidate_weight}")
    
    # 선택된 후보 inference 간 유사도  
    index_combinations = list(combinations(range(len(candidates)), 2))
    
    similarity_matrix = pd.DataFrame(
        np.zeros((len(candidates), len(candidates))), 
        index=args.models.split(' '), 
        columns=args.models.split(' ')
    )
    
    for i, j in index_combinations:
        df1 = candidates[i]
        df2 = candidates[j]
        sim = count_similiarity(df1, df2)
        df1_name = candidate_names[i]
        df2_name = candidate_names[j]
        # print(f'{df1_name} and {df2_name}: {sim:.2f}% similarity')
    
        similarity_matrix.loc[df1_name, df2_name] = sim
        similarity_matrix.loc[df2_name, df1_name] = sim
    
    print(f"{args.models} inference 결과 유사도: \n{similarity_matrix.round(2)}")
    
    # 추천 결과를 저장할 딕셔너리 {user_id: {item_id: weighted_score}}
    weighted_recommendations = defaultdict(lambda: defaultdict(float))

    # 가중 합 계산: inference별 순위(rank)의 가중치를 반영
    print("Calculating weighted recommendations with rank aggregation...")
    for model_idx, df in tqdm(enumerate(candidates), total=len(candidates)):
        weight = weights[model_idx]['weight']
        grouped = df.groupby("user")

        # 각 사용자별로 순위를 기반으로 점수 계산
        for user, group in grouped:
            # 아이템별 순위를 계산
            group["rank"] = range(1, len(group) + 1)

            # 순위를 점수에 반영 (역순위 사용: 높은 순위 -> 높은 점수)
            for _, row in group.iterrows():
                item = row["item"]
                rank = row["rank"]
                # 점수 계산: 모델 가중치 + 역순위 가중치
                rank_weight = 1 / (rank+60)  # 예: 1등 = 1, 2등 = 0.5, ...
                weighted_recommendations[user][item] += weight * rank_weight

    # top@10 계산 및 DataFrame 생성
    top_k = 10
    recommendation_list = []

    print("\nGenerating top@10 recommendations for each user...")
    for user, item_scores in tqdm(weighted_recommendations.items(), total=len(weighted_recommendations)):
        # 아이템별 점수를 기준으로 정렬하여 top@10 추출
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        for item, _ in sorted_items:
            recommendation_list.append({"user": user, "item": item})

    # 최종 추천 결과 DataFrame
    final_recommendations_df = pd.DataFrame(recommendation_list)

    # 결과 저장
    output_dir = "./data/output/Ensemble/"
    output_name = f"output_ensemble_of_{'_'.join(candidate_names)}.csv"
    final_recommendations_df.to_csv(os.path.join(output_dir, output_name), index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", "-m", type=str, default="EASE CDAE", help="select candidate models for ensemble")
    parser.add_argument("--weights", "-w", help="input model weights of hard voting")
    args, _ = parser.parse_known_args()
    hard_voting(args)