import json
import pickle

import boto3
from firebase_admin import firestore
import numpy as np
import wandb

from sparsepy.core.results.training_result import TrainingResult
from sparsepy.core.results.training_result import TrainingStepResult
from sparsepy.core.results.evaluation_result import EvaluationResult
from sparsepy.core.results.hpo_result import HPOResult
from sparsepy.core.results.hpo_step_result import HPOStepResult
from sparsepy.access_objects.models.model import Model

"""
DataStorer: Stores data to weights and biases
"""
class DataStorer:

    def __init__(self, config: dict):
        # configure saved metrics?
        self.saved_metrics = [metric["name"] for metric in config if metric["save"] is True]
        # create API client
        self.api = wandb.Api()
        # connect to Firestore
        self.firestore_db = firestore.client()
        # connect to DynamoDB
        self.db = boto3.resource('dynamodb')
        self.experiments_table = self.db.Table("sts_experiments")
        self.hpo_table = self.db.Table("sts_hpo_runs")
        
    
    def save_model(self, m: Model):
        # Implementation to save the model
        pass
    
    def save_training_step(self, parent: str, result: TrainingStepResult):

        summary_dict = {}
        full_dict = {}
        
        # gather the saved metrics for summary in W&B and full storage in the DB
        for metric_name, metric_val in result.get_metrics()[0].items():
            if metric_name in self.saved_metrics:
                summary_dict[metric_name] = self.average_nested_data(metric_val)
                full_dict[metric_name] = pickle.dumps(metric_val) # pickling
                #wandb.log({metric_name: self.average_nested_data(metric_val)}, commit=False)
        
        # save the summary to W&B
        # add this if we add step resolution back now that we don't need W&B concordance
        #summary_dict["resolution"] = result.resolution
        wandb.log(summary_dict)

        # save the full results to Firestore
        experiment_ref = self.firestore_db.collection("experiments").document(parent)

        if experiment_ref.get().exists:
            # add step to existing experiment
            experiment_ref.update(
                {
                    "saved_metrics.training": firestore.ArrayUnion([full_dict])
                }
            )
        #else:
        # raise exception
            
        # save results to DynamoDB
        self.experiments_table.update_item(
            Key={"id": parent},
            UpdateExpression="SET #tm = list_append(#tm, :v)",
            ExpressionAttributeNames={"#tm": "saved_metrics.training"},
            ExpressionAttributeValues={":v": [{(":"+k):v for k, v in full_dict.items()}]}
        )
    
    def create_experiment(self, experiment: TrainingResult):
        # create the DB entry in DynamoDB
        self.experiments_table.put_item(Item={
            "id": experiment.id,
            "start_time": experiment.start_time.isoformat(), # save as string since DynamoDB does not support datetime
            "saved_metrics": {
                "training": [], # DynamoDB requires the list to already exist to append to it
                "evaluation": [],
                "resolution": experiment.resolution
            },
            "completed": False
        })
        
        # create the DB entry for this experiment in Firestore
        experiment_ref = self.firestore_db.collection("experiments").document(experiment.id)
        experiment_ref.set({
            "id": experiment.id,
            "start_time": experiment.start_time.isoformat(),
            "saved_metrics": {
                "resolution": experiment.resolution
            },
            "completed": False
            })

    def save_training_result(self, result: TrainingResult):
        # Implementation to save the training result

        # do we even need to set anything in W&B here? time finished? but W&B should track that
        # is there something we need to do here to mark end of run?
        # FIXME save required data if any, else remove
        #run = self.api.run(wandb.run.path)

        experiment_ref = self.firestore_db.collection("experiments").document(result.id)

        # every invocation of this costs 1 read call; consider removing
        if experiment_ref.get().exists:
            experiment_ref.update(
                {
                    "end_time": result.end_time,
                    "completed": True
                }
            )

        # COMMENT THIS IN to test updated config saving
        #run.config = result.

    def save_evaluation_result(self, parent: str, result: EvaluationResult):
        # Implementation to save the evaluation result

        # access the current run's summary-level data with the API
        run = self.api.run(wandb.run.path)
        
        # gather the saved metrics for summary in W&B and full storage in the DB        
        eval_dict = {}

        for metric_name, metric_val in result.get_metrics()[0].items():
            if metric_name in self.saved_metrics:
                run.summary["Evaluation" + metric_name] = metric_val
                eval_dict[metric_name] = pickle.dumps(metric_val) # thank you, Firestore!

        # save summary to W&B
        run.summary.update()
        # save the full results to Firestore
        experiment_ref = self.firestore_db.collection("experiments").document(parent)
        if experiment_ref.get().exists:
            # add step to existing experiment
            experiment_ref.update(
                {
                    "saved_metrics.evaluation": firestore.ArrayUnion([eval_dict])
                }
            )

    def save_hpo_step(self, parent: str, result: HPOStepResult):
        # Implementation to save the HPO step result

        # WEIGHTS & BIASES
        # save the objective and HPO config to this experiment

        objective = result.get_objective()
        # FIXME correct duplicate logging, only log hpo_objective
        wandb.log(
                    {
                    'hpo_objective': objective["total"],
                    'objective_details': objective 
                    }
                )

        run = self.api.run(wandb.run.path)
        #run.summary["objective"] = result.get_objective()
        run.summary["hpo_configs"] = result.configs

        run.summary.update()
    
        # FIRESTORE
        # save the objective data into the experiment
        hpo_step_experiment_ref = self.firestore_db.collection("experiments").document(result.id)

        hpo_step_experiment_ref.update(
            {
                "hpo_objective": result.get_objective(),
                "parent_sweep": parent
            }
        )

        # mark this HPO step's experiment as belonging to the parent sweep
        parent_sweep_ref = self.firestore_db.collection("hpo_runs").document(parent)

        parent_sweep_ref.update(
            {
                "runs": firestore.ArrayUnion([result.id])
            }
        )

        

    def create_hpo_sweep(self, sweep: HPOResult):
        
        # set up the data
        new_sweep_data = {
            "id": sweep.id,
            "name": sweep.name,
            "start_time": sweep.start_time.isoformat(),
            "best_run_id": None,
            "runs": [],
            "completed": False,
            "configs": {
                conf_name:json.dumps(conf_json) for conf_name, conf_json in sweep.configs.items()
            }
        }

        self.hpo_table.put_item(Item=new_sweep_data)

        # create the DB entry for this experiment in Firestore
        sweep_ref = self.firestore_db.collection("hpo_runs").document(sweep.id)
        sweep_ref.set({
            "id": sweep.id,
            "name": sweep.name,
            "start_time": sweep.start_time,
            "best_run_id": None,
            "runs": [],
            "completed": False,
            "configs": {
                conf_name:json.dumps(conf_json) for conf_name, conf_json in sweep.configs.items()
            }
            })

        # and in DynamoDB


    def save_hpo_result(self, result: HPOResult):
        # Implementation to save the HPO result

        # WEIGHTS & BIASES automatically tracks this already

        # FIRESTORE
        # set the end time and best run ID and mark as completed
        sweep_ref = self.firestore_db.collection("hpo_runs").document(result.id)

        sweep_ref.update(
            {
                "end_time": result.end_time,
                "best_run_id": result.best_run.id,
                "completed": True
            }
        )

        # DynamoDB
        self.hpo_table.update_item(
            Key={"id": result.id},
            UpdateExpression="SET end_time=:et, best_run_id=:brid, completed=:c",
            ExpressionAttributeValues={
                ":et": result.end_time.isoformat(),
                ":brid": result.best_run.id,
                ":c": True
            }
        )
    
    def create_artifact(self, content: dict) -> wandb.Artifact:
        # Implementation to create and save a wandb.Artifact
        pass

    def average_nested_data(self, data):
        if isinstance(data, list):
            return np.mean(np.nan_to_num([self.average_nested_data(item) for item in data]))
        elif hasattr(data, 'tolist'):  # numpy array
            return np.mean(np.nan_to_num(data))
        else:
            # Scalar value
            return data