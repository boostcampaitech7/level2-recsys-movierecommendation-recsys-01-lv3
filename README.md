![header](https://capsule-render.vercel.app/api?type=waving&color=0:EDDFE0,100:B7B7B7&width=max&height=250&section=header&text=Movie&nbsp;Recommendation&desc=RecSys05-오곡밥&fontSize=40&fontColor=4A4947&&fontAlignY=40)

## 🍚 팀원 소개

|문원찬|안규리|오소영|오준혁|윤건욱|황진욱|
|:---:|:---:|:---:|:---:|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/a29cbbd9-0cde-495a-bd7e-90f20759f3d1" width="100"/> | <img src="https://github.com/user-attachments/assets/c619ed82-03f3-4d48-9bba-dd60408879f9" width="100"/> | <img src="https://github.com/user-attachments/assets/1b0e54e6-57dc-4c19-97f5-69b7e6f3a9b4" width="100"/> | <img src="https://github.com/user-attachments/assets/67d19373-8cac-4676-bde1-b0637921cf7f" width="100"/> | <img src="https://github.com/user-attachments/assets/f91dd46e-9f1a-42e7-a939-db13692f4098" width="100"/> | <img src="https://github.com/user-attachments/assets/69bbb039-752e-4448-bcaa-b8a65015b778" width="100"/> |
| [![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/WonchanMoon)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/notmandarin)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/irrso)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/ojunhyuk99)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/YoonGeonWook)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/hw01931)|

</br>

## 💡 프로젝트 개요

### 프로젝트 소개
사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 영화를 예측하는 프로젝트입니다.

### 데이터 소개
데이터셋은 사용자 아이디, 아이템 아이디, 시점, 영화 제목, 개봉년도, 감독, 장르, 작가 등 다양한 특성으로 구성되어 있습니다.

### 데이터셋 구성
- **train_ratings.csv**: user id, item id, timestamp로 구성된 학습 데이터
- **MI_item2attributes.json** : 전처리에 의해 생성된 데이터
- **titles.tsv** : 영화 제목
- **years.tsv** : 영화 개봉년도
- **directors.tsv**: 영화 별 감독
- **genres.tsv**: 영화 장르
- **writers.tsv**: 영화 작가

### 평가 방식
- **평가 지표**: Recall@10을 사용하여 예측 성능을 평가합니다.

### 프로젝트 목표
단순한 평점 기반 추천이 아니라, 사용자 행동 패턴과 시청 이력을 반영한 implicit feedback을 중심으로 한 순차적 추천 문제를 해결하고자 합니다.

</br>

## 📂폴더구조
```
# level2-recsys-movierecommendation-recsys-01-lv3/
│
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── .keep 
│   └── PULL_REQUEST_TEMPLATE.md
│   
├── recbole_project/
│   ├── configs/                              # 모델 유형별 config yaml 파일                           
│   │   ├── context/
│   │   │   ├── dcnv2_config.yaml
│   │   │   ├── deepfm_config.yaml
│   │   │   ├── lightgbm_config.yaml
│   │   │   ├── widedeep_config.yaml
│   │   │   └── xgboost_config.yaml
│   │   │    
│   │   ├── general/
│   │   │   ├── admmslim_config.yaml
│   │   │   ├── bpr_config.yaml
│   │   │   ├── cdae_config.yaml
│   │   │   ├── ease_config.yaml
│   │   │   ├── lightgcn_config.yaml
│   │   │   ├── multidae_config.yaml
│   │   │   ├── multivae_config.yaml
│   │   │   ├── neumf_config.yaml
│   │   │   ├── pop_config.yaml
│   │   │   ├── recvae_config.yaml
│   │   │   └── slimelastic_config.yaml
│   │   │                        
│   │   └── sequential/
│   │   │   ├── ber4rec_config.yaml
│   │   │   ├── gru4rec_config.yaml
│   │   │   ├── gru4recf_config.yaml
│   │   │   ├── s3rec_config.yaml
│   │   │   ├── sasrec_config.yaml
│   │   │   ├── sasrecf_config.yaml
│   │   │   └── transrec_config.yaml
│   │                          
│   ├── mlflow_tracking_server/
│   │   ├── mlflow.db
│   │   └── readme.md
│   │                              
│   ├── src/                                  # 모델 학습/평가 및 추론 스크립트  
│   │   ├── __init__.py                                
│   │   ├── data_preparation.py
│   │   ├── inference.py
│   │   ├── mlflow_train_and_evaluate.py
│   │   └── train_and_evaluate.py
│   │    
│   ├── utils/
│   │   ├── __init__.py
│   │   └── utils.py
│   │            
│   ├── hard_voting.py                        # Hard Voting Ensemble 코드
│   ├── hard_voting_preparation.ipynb         # Hard voting을 위한 준비 코드
│   ├── requirements.txt
│   ├── run.py                                # train_and_evaluate.py 실행 코드
│   ├── run_inference.py                      # inference.py 실행 코드
│   └── run_mlflow.py                         # mlflow_train_and_evaluate.py 실행 코드
│
├── .gitignore
└── README.md
```
</br>

## ⚙️ 개발 환경
#### OS: Linux (5.4.0-99-generic, x86_64)
#### GPU: Tesla V100-SXM2-32GB (CUDA Version: 12.2)
#### CPU : Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz, 8 Cores
</br>

## 🔧 기술 스택

#### 프로그래밍 언어 <img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=python&logoColor=white"/>

#### 데이터 분석 및 전처리 <img src="https://img.shields.io/badge/pandas-150458.svg?style=flat-square&logo=pandas&logoColor=white"/> <img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat-square&logo=numpy&logoColor=white"/>

#### 모델 학습 및 평가 
  
#### 시각화 도구 <img src="https://img.shields.io/badge/Matplotlib-3F4F75.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/seaborn-221E68.svg?style=flat-square&logoColor=white"/>

#### 개발 환경 <img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat-square&logo=jupyter&logoColor=white"/>

#### 실험 관리 및 추적 <img src="https://img.shields.io/badge/MLflow-0194E2.svg?style=flat-square&logo=MLflow&logoColor=white"/>
