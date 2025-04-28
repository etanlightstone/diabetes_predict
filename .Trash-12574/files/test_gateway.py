from mlflow.deployments import get_deploy_client
import os

client = get_deploy_client(os.environ['DOMINO_MLFLOW_DEPLOYMENTS'])

response = client.predict(
	endpoint="etan-basic-gpt-oai", 
	inputs={"prompt": "It's one small step for"}
)
print(response)