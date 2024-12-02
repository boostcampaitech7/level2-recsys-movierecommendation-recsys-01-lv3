# src/data_preparation.py

import os
import pandas as pd
import json

def prepare_data(base_path, output_path):
    # 사용자-아이템 상호작용 데이터 로드
    inter_df = pd.read_csv(os.path.join(base_path, "train_ratings.csv"))
    
    # 아이템 메타데이터 로드
    with open(os.path.join(base_path, "Ml_item2attributes.json"), 'r') as f:
        item2attributes = json.load(f)
    item2attributes = {str(k): v for k, v in item2attributes.items()}
    
    # 아이템 메타데이터(side-information) 로드
    directors_df = pd.read_csv(os.path.join(base_path, 'directors.tsv'), sep='\t')
    genres_df = pd.read_csv(os.path.join(base_path, 'genres.tsv'), sep='\t')
    titles_df = pd.read_csv(os.path.join(base_path, 'titles.tsv'), sep='\t')
    writers_df = pd.read_csv(os.path.join(base_path, 'writers.tsv'), sep='\t')
    years_df = pd.read_csv(os.path.join(base_path, 'years.tsv'), sep='\t')
    
    # Atomic Files로 변환: 상호작용 데이터
    # 컬럼 이름 변경
    inter_df.rename(columns={'user': 'user_id', 'item': 'item_id', 'time': 'timestamp'}, inplace=True)


    # 데이터 타입 변환
    inter_df['user_id'] = inter_df['user_id'].astype(str)
    inter_df['item_id'] = inter_df['item_id'].astype(str)
    inter_df['timestamp'] = inter_df['timestamp'].astype(int)

    # 필드 이름과 데이터 타입 지정
    inter_df.columns = [
        'user_id:token',
        'item_id:token',
        'timestamp:float',
    ]
    
    # Atomic Files로 변환: Item Profile 데이터
    directors_df.rename(columns={'item': 'item_id'}, inplace=True)
    genres_df.rename(columns={'item': 'item_id'}, inplace=True)
    titles_df.rename(columns={'item': 'item_id'}, inplace=True)
    writers_df.rename(columns={'item': 'item_id'}, inplace=True)
    years_df.rename(columns={'item': 'item_id'}, inplace=True)

    # 아이템 ID를 문자열로 변환
    directors_df['item_id'] = directors_df['item_id'].astype(str)
    genres_df['item_id'] = genres_df['item_id'].astype(str)
    titles_df['item_id'] = titles_df['item_id'].astype(str)
    writers_df['item_id'] = writers_df['item_id'].astype(str)
    years_df['item_id'] = years_df['item_id'].astype(str)

    # 각 데이터프레임의 컬럼 이름 변경
    genres_df.rename(columns={'genre': 'genres'}, inplace=True)
    directors_df.rename(columns={'director': 'directors'}, inplace=True)
    writers_df.rename(columns={'writer': 'writers'}, inplace=True)

    genres_grouped = genres_df.groupby(['item_id'])['genres'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
    directors_grouped = directors_df.groupby(['item_id'])['directors'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
    writers_grouped = writers_df.groupby(['item_id'])['writers'].apply(lambda x: ' '.join(x.astype(str))).reset_index()

    # 아이템 데이터 병합
    item_df = titles_df.merge(genres_grouped, on='item_id', how='left')
    item_df = item_df.merge(directors_grouped, on='item_id', how='left')
    item_df = item_df.merge(writers_grouped, on='item_id', how='left')
    item_df = item_df.merge(years_df, on='item_id', how='left')

    # 결측치 처리: NaN을 빈 문자열로 대체
    item_df = item_df.fillna({'year': 0, 'directors': '', 'writers': '', 'genres': ''})

    # 데이터 타입 변환
    item_df['title'] = item_df['title'].astype(str)
    item_df['genres'] = item_df['genres'].astype(str)
    item_df['directors'] = item_df['directors'].astype(str)
    item_df['writers'] = item_df['writers'].astype(str)
    item_df['year'] = item_df['year'].astype(int)

    # 필드 이름과 데이터 타입 지정
    item_df.columns = [
        'item_id:token',
        'title:token_seq',
        'genres:token_seq',
        'directors:token_seq',
        'writers:token_seq',
        'year:float',
    ]
    
    # 결과 저장
    os.makedirs(output_path, exist_ok=True)
    inter_df.to_csv(os.path.join(output_path, 'dataset.inter'), index=False, sep='\t')
    item_df.to_csv(os.path.join(output_path, 'dataset.item'), index=False, sep='\t')

if __name__ == '__main__':
    base_path = './data/raw/train/'
    eval_path = './data/raw/eval/'
    output_path = './data/processed/dataset/'
    prepare_data(base_path, output_path)