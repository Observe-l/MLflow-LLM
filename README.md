# MLflow-LLM
Demo of MLflow LLM

### MLflow LLM: Local model: Custom PyFunc
[Custom-pyfunc](custom-pyfunc-advanced-llm.ipynb)

### MLflow-Open AI: Prompt Engineering UI
```shell
export OPENAI_API_KEY=my_key
mlflow deployments start-server --config-path config.yaml --port 7000

export MLFLOW_DEPLOYMENTS_TARGET="http://127.0.0.1:7000"
mlflow server --port 5000

curl http://localhost:11434/api/generate -d '{
 "model": "deepseek-r1"
}'
```


|model|MSE|
|---|---|
|llama3.1:8b|4602.93|
|gemma3:12b|4919.89|
