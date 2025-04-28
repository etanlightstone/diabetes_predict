import argparse
import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# Define model configurations to try
def get_model_configs():
    """
    Define different model configurations to evaluate
    """
    configs = [
        # Baseline config with default parameters
        {
            "name": "baseline",
            "params": {
                "epochs": 10,
                "batch_size": 64,
                "hidden_dim1": 32,
                "hidden_dim2": 16, 
                "hidden_dim3": 8,
                "learning_rate": 0.001
            }
        },
        # Configuration with larger network
        {
            "name": "larger_network",
            "params": {
                "epochs": 15,
                "batch_size": 64,
                "hidden_dim1": 64,
                "hidden_dim2": 32,
                "hidden_dim3": 16,
                "learning_rate": 0.001
            }
        },
        # Configuration with smaller learning rate
        {
            "name": "small_lr",
            "params": {
                "epochs": 15,
                "batch_size": 64,
                "hidden_dim1": 32,
                "hidden_dim2": 16,
                "hidden_dim3": 8, 
                "learning_rate": 0.0001
            }
        },
        # Configuration with larger learning rate
        {
            "name": "large_lr",
            "params": {
                "epochs": 15,
                "batch_size": 64,
                "hidden_dim1": 32,
                "hidden_dim2": 16,
                "hidden_dim3": 8,
                "learning_rate": 0.01
            }
        },
        # Configuration with different batch size
        {
            "name": "small_batch",
            "params": {
                "epochs": 15,
                "batch_size": 16,
                "hidden_dim1": 32,
                "hidden_dim2": 16,
                "hidden_dim3": 8,
                "learning_rate": 0.001
            }
        }
    ]
    return configs

def run_model_configuration(config):
    """Run the diabetes_trainer.py with the specific configuration"""
    print(f"Running configuration: {config['name']}")
    
    # Build command with all parameters
    cmd = "python diabetes_trainer.py"
    for param, value in config['params'].items():
        cmd += f" --{param} {value}"
    
    # Execute the training command
    print(f"Executing: {cmd}")
    return_code = os.system(cmd)
    
    if return_code != 0:
        print(f"Error running configuration {config['name']}")
        return False
    
    return True

def compare_results():
    """Compare results from MLflow and print best models"""
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Diabetes Prediction2")
    
    if experiment:
        experiment_id = experiment.experiment_id
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["metrics.val_accuracy DESC"]
        )
        
        # Create a DataFrame to display results
        results = []
        for run in runs:
            # Extract the parameters and metrics
            params = run.data.params
            metrics = run.data.metrics
            
            # Add to results list
            results.append({
                "run_id": run.info.run_id,
                "val_accuracy": metrics.get("val_accuracy", 0),
                "val_loss": metrics.get("val_loss", 0),
                "epochs": params.get("epochs", ""),
                "batch_size": params.get("batch_size", ""),
                "learning_rate": params.get("learning_rate", ""),
                "hidden_dim1": params.get("hidden_dim1", ""),
                "hidden_dim2": params.get("hidden_dim2", ""),
                "hidden_dim3": params.get("hidden_dim3", ""),
            })
        
        # Create DataFrame and sort by validation accuracy
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values("val_accuracy", ascending=False)
            print("\nModel Optimization Results:")
            print(results_df)
            
            # Get best run
            best_run = results_df.iloc[0]
            print(f"\nBest model run_id: {best_run['run_id']}")
            print(f"Best model accuracy: {best_run['val_accuracy']:.4f}")
            print(f"Best model configuration:")
            for param in ["epochs", "batch_size", "learning_rate", "hidden_dim1", "hidden_dim2", "hidden_dim3"]:
                print(f"  {param}: {best_run[param]}")
        else:
            print("No results found")
    else:
        print(f"Experiment 'Diabetes Prediction2' not found")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Optimize diabetes prediction model")
    parser.add_argument("--run-all", action="store_true", help="Run all configurations")
    parser.add_argument("--config", type=str, help="Run specific configuration by name")
    parser.add_argument("--compare", action="store_true", help="Compare results of previous runs")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_results()
        return
    
    configs = get_model_configs()
    
    if args.config:
        # Run specific configuration
        for config in configs:
            if config["name"] == args.config:
                run_model_configuration(config)
                break
        else:
            print(f"Configuration '{args.config}' not found")
    elif args.run_all:
        # Run all configurations
        for config in configs:
            run_model_configuration(config)
        
        # Compare results at the end
        compare_results()
    else:
        # Display available configurations
        print("Available configurations:")
        for config in configs:
            print(f"  - {config['name']}")
        print("\nUse --run-all to run all configurations or --config NAME to run a specific one")

if __name__ == "__main__":
    main() 