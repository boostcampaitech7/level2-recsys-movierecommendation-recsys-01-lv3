# 서버 실행 코드 (터미널)
*RecSys1 서버 전용*
*반드시 해당 폴더로 이동한 다음 실행시킬 것*

```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --serve-artifacts --host 0.0.0.0 --port 30696
```