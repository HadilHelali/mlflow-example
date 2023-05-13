import pandas as pd
import mlflow
import os 


with mlflow.start_run() as run:
    mlflow.sklearn.autolog()
    run_id = run.info.run_id

    
    
    
os.listdir(f"mlruns/0/{run_id}/artifacts/model")



import whylogs as why

profile_result = why.log(df)
profile_view = profile_result.view()


profile_view.to_pandas()


def log_profile(dataframe: pd.DataFrame) -> None:
    profile_result = why.log(dataframe)
    profile_result.writer("mlflow").write()
    
    
    
with mlflow.start_run() as run:
    mlflow.sklearn.autolog()

    df = get_data()
    train(dataframe=df)

    log_profile(dataframe=df)

    run_id = run.info.run_id

    mlflow.end_run()
    
    

from mlflow.tracking import MlflowClient

client = MlflowClient()

local_dir = "/tmp/artifact_downloads"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)
local_path = client.download_artifacts(run_id, "whylogs", local_dir)

os.listdir(local_path)



profile_name = os.listdir(local_path)[0]
result = why.read(path=f"{local_path}/{profile_name}")


result.view().to_pandas()


os.environ["WHYLABS_DEFAULT_ORG_ID"] = "org-Kw8Uby" # ORG-ID is case-sensitive
os.environ["WHYLABS_API_KEY"] = "knnKBJ5NKr.40fGkmQkMcCsgCUkgcyvjwZts4e5SmiiYPeoSg2eWDp1YRHoReKnU"
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "model-3" # The selected model project "MODEL-NAME" is "model-0"

results = why.log(df)

results.writer("whylabs").write()

