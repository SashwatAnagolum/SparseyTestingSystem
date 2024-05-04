"""
printer.py - contains the printer class for printing result and metric data to the console
"""

from pprint import pformat
import torch
from tqdm import tqdm

from sparseypy.core.results import TrainingResult, TrainingStepResult, HPOResult, HPOStepResult

class Printer:
    """
    Printer class to consolidate metric-printing logic in a single place 
    within the Sparsey Testing System.
    """


    @staticmethod
    def print_best_steps(results: TrainingResult, run_type: str = "training") -> None:
        """
        Summarizes the best steps of a single training/evaluation run.

        Args:
            results (TrainingResult): the results to summarize
            run_type (str): the type of run to summarize
        """
        tqdm.write(f"{run_type.capitalize()} completed.\n\n{run_type.upper()} - SUMMARY\n")
        tqdm.write("Best metric steps:")
        for metric, val in results.best_steps.items():
            tqdm.write(
                f"* {metric:>25}: step {val['best_index']:<5} (using {val['best_function'].__name__})"
            )

    @staticmethod
    def print_pre_hpo_summary(hpo_config: dict) -> None:
        """
        Prints a pre-execution summary of an HPO run.

        Args:
            hpo_config (dict): the validated HPO configuration that will be
                used for the run.
        """
        met_separator = "\n* "
        combination_item = "{mn:<25} (weight: {mw:.5f})"

        obj_vals = [
            combination_item.format(mn=x['metric']['name'], mw=x['weight'])
            for x in hpo_config['optimization_objective']['objective_terms']
            ]

        tqdm.write(f"""
HYPERPARAMETER OPTIMIZATION SUMMARY
          
W&B project name:      {hpo_config['project_name']}
W&B sweep name:        {hpo_config['hpo_run_name']}

Model family:          {hpo_config['model_family']}
Optimization strategy: {hpo_config['hpo_strategy']}
Number of runs:        {hpo_config['num_candidates']}

Selected metrics: 
* {met_separator.join([x["name"] for x in hpo_config["metrics"]])}

Objective calculation: {hpo_config['optimization_objective']['combination_method']} of
* {met_separator.join(obj_vals)}""")


    @staticmethod
    def print_post_hpo_summary(hpo_results: HPOResult, hpo_config: dict,
                               sweep_url: str, best_run_url: str) -> None:
        """
        Summarizes a completed HPO run.

        Args:
            hpo_results (HPOResult): the HPOResults object containing the HPO results
            hpo_config (dict): the current system HPO configuration
            hpo_run (HPORun): the HPORun that logged the current run; used to retrieve
                W&B information required to print the summary
        """
        # calculate total time elapsed
        delta = hpo_results.end_time - hpo_results.start_time
        # print the header
        tqdm.write("\n---------------------------------------------------------")
        tqdm.write("HYPERPARAMETER OPTIMIZATION COMPLETED")
        tqdm.write(f"\nCompleted {hpo_config['num_candidates']} runs in {str(delta.seconds)} seconds.")
        # print the breakdown of the best-performing run
        tqdm.write("\n---------------------------------------------------------")
        tqdm.write("BEST RUN\n")
        tqdm.write("")
        Printer.summarize_hpo_step(step_results=hpo_results.best_run, print_config=True)
        tqdm.write("\n---------------------------------------------------------")
        # print the summary directing users to Weights & Biases
        tqdm.write(f"""Review full results in Weights & Biases:

Project name:   {hpo_config['project_name']}

HPO sweep name: {hpo_config['hpo_run_name']}
HPO sweep URL:  {sweep_url}

Best run ID:    {hpo_results.best_run.id}
Best run URL:   {best_run_url}""")


    @staticmethod
    def print_hpo_exception(current_step: int, message: str):
        """
        Prints a message for an exception that occurs during HPO execution.

        Args:
            current_step (int): the current step number
            message (str): the traceback to print
        """
        tqdm.write(f"WARNING: EXCEPTION OCCURRED DURING HPO STEP {current_step}")
        tqdm.write("Exception traceback:")
        tqdm.write(message)


    @staticmethod
    def print_pre_training_summary(training_dataset_config: dict,
        evaluation_dataset_config: dict, trainer_config: dict,
        training_num_batches: int, eval_num_batches: int) -> None:
        """
        Prints a pre-execution summary of a single training run.

        Args:
            training_dataset_config (dict): the validated training dataset
                configuration that will be used for the run.
            evaluation_dataset_config (dict):  the validated evaluation dataset
                configuration that will be used for the run.
            trainer_config (dict): the validated trainer configuration that will be
                used for the run.
            training_num_batches (int): the number of batches passed to 
                the model during training.
            eval_num_batches (int): the number of batches passed to
                the model during evaluation.
        """
        # print training run summary
        met_separator = "\n* "
        tqdm.write(f"""
TRAINING RUN SUMMARY

Training:
* Dataset type:      {training_dataset_config['dataset_type']}
* Batch size:        {trainer_config['training']['dataloader']['batch_size']}
* Number of batches: {training_num_batches}

Evaluation:
* Dataset type:      {evaluation_dataset_config['dataset_type']}
* batch size:        {trainer_config['eval']['dataloader']['batch_size']}
* Number of batches: {eval_num_batches}

Selected metrics: 
* {met_separator.join([x["name"] for x in trainer_config["metrics"]])}
""")


    @staticmethod
    def print_run_start_message(run_name: str, run_url: str, phase: str = "training"):
        """
        Prints the "beginning run" summary to the console.

        Args:
            run_name (str): the name of the run in Weights & Biases
            run_url (str): the URL of the run in Weights & Biases
            phase (str): the current phase (training/validation/evaluation)
        """
        tqdm.write(f"{phase.upper()} STARTED")
        tqdm.write(f"Run name:          {run_name}")
        tqdm.write(f"View results live: {run_url}\n")


    @staticmethod
    def print_post_train_model_summary(model_name: str, run_group: str,
                                       train_url: str, eval_url: str):
        """
        Prints the final summary for the train_model task, including the
        model name and run URL.

        Args:
            model_name (str): the name of the model in Weights & Biases
            run_group (str): the name of the run group on W&B containing
                the training and evaluation runs for this training session
            train_url (str): the URL for this training run in
                Weights & Biases
            eval_url (str): the URL for this evaluation run in 
                Weights & Biases
        """
        tqdm.write(f"""
TRAIN MODEL COMPLETED
Review results in Weights & Biases:

Model name:           {model_name}
Group name:           {run_group}

Run URL (Training):   {train_url}
Run URL (Evaluation): {eval_url}""")


    @staticmethod
    def print_pre_evaluate_model_summary(dataset_config: dict, trainer_config: dict,
                                         model_name: str, num_batches: int):
        """
        Prints the initial summary for the evaluate_model task, including the
        model, dataset, and selected metrics.

        Args:
            dataset_config (str): the validated dataset configuration
            trainer_config (str): the validated trainer configuration
            model_name (str): the name of the model in Weights & Biases
            num_batches (str): the number of evaluation batches to be run
        """
        met_separator = "\n* "
        tqdm.write(f"""
EVALUATION RUN SUMMARY
Model:             {model_name}
Dataset type:      {dataset_config['dataset_type']}
Batch size:        {trainer_config['eval']['dataloader']['batch_size']}
Number of batches: {num_batches}

Selected metrics: 
* {met_separator.join([x["name"] for x in trainer_config["metrics"]])}
    """)


    @staticmethod
    def print_post_evaluate_model_summary(model_name: str, run_url: str, run_group: str):
        """
        Prints the final summary for the evaluate_model task, including the
        model name and run URL.

        Args:
            model_name (str): the name of the model in Weights & Biases
            run_url (str): the URL for this evaluation run in 
                Weights & Biases
            run_group (str): the group name for this run in
                Weights & Biases
        """
        tqdm.write("\nEVALUATE MODEL COMPLETED")
        tqdm.write("Review results in Weights & Biases:")
        tqdm.write(f"Model name: {model_name}")
        tqdm.write(f"Group name: {run_group}")
        tqdm.write(f"Run URL:    {run_url}")


    @staticmethod
    def print_step_metrics(step_data: TrainingStepResult, batch_number: int,
                            max_batch_size: int = 1, step_type: str = "training"):
        """
        Prints the metrics for a single training/evaluation step to the console.

        Args:
            step_data (TrainingStepResult): the metric data for this step
            batch_number (int): the current batch number for this input (e.g. input 50)
            max_batch_size (int): the maximum possible size of the current batch
            batch_type (str): whether the current step is training or evaluation
        """
        metric_data = step_data.get_metrics()
        # for each item in the batch
        for batch_index in range(step_data.batch_size):
            # calculate input number and print input header
            input_num = (batch_number - 1) * max_batch_size + (batch_index + 1)
            tqdm.write(
                f"\n\n{step_type.capitalize()} results - INPUT {input_num}\n--------------------"
            )
            # then write metric values
            # for each metric, slice the batch input for the current step
            # step output data is in the dimensions [layers][batch][MACs][...]
            # so to extract the batch data we need to slice across dimension 1
            # push it to the CPU to avoid cluttering the output with "device=cuda:0"
            # and assemble it into a dict so it looks the same as non-batched output
            # then pretty-format it and write it to the console
            tqdm.write(
                pformat(
                    {
                        metric: torch.select(data.cpu(), dim=1, index=batch_index)
                        for metric, data in metric_data.items()
                    }
                )
            )


    @staticmethod
    def summarize_hpo_trial(step_results: HPOStepResult, step_num: int, num_trials: int,
                        print_config: bool = False, new_best: bool = False):
        """
        Prints a summary for a single HPO trial.

        Args:
            step_results (HPOStepResult): the results for the step
            print_config (bool): whether to also print the model config for this step
            step_num (int): the number of this current step
            num_trials (int): the maximum number of steps in the HPO run
            new_best (bool): whether this step is the new best step
        """
        # print the remaining trial counts
        tqdm.write(f"\nCompleted trial {step_num} of {num_trials}")
        # if there is a previous best value, print the prior objective value
        if new_best:
            tqdm.write(
                "New best objective value!"
            )
        Printer.summarize_hpo_step(step_results=step_results, print_config=print_config)


    @staticmethod
    def summarize_hpo_step(step_results: HPOStepResult,
                           print_config: bool = False):
        """
        Prints a breakdown of a single HPO step.

        Args:
            step_results (HPOStepResult): the results for the step
            print_config (bool): whether to also print the model config for this step
        """
        objective_results = step_results.get_objective()
        # TODO enhance with summary of metrics
        tqdm.write(f"Objective value:            {objective_results['total']:.5f}")
        tqdm.write(f"Combination method:         {objective_results['combination_method']}")
        tqdm.write("Objective term breakdown:")
        for name, values in objective_results["terms"].items():
            tqdm.write(f"* {name:>24}: {values['value']:.5f} with weight {values['weight']}")

        if print_config:
            Printer.print_model_config(step_results.configs["model_config"])


    @staticmethod
    def print_model_config(model_config: dict):
        """
        Prints a model configuration file to the console.

        Args:
            model_config (dict): the model configuration to print
        """
        tqdm.write("\n---------------------------------------------------------")
        tqdm.write("Configuration:\n---------------------------------------------------------")
        layer_number = 1
        tqdm.write('INPUT DIMENSIONS ')
        tqdm.write(pformat(model_config["input_dimensions"]))
        tqdm.write("\n---------------------------------------------------------")
        for layer in model_config["layers"]:
            tqdm.write(f"LAYER {layer_number}")
            tqdm.write("\n---------------------------------------------------------")
            tqdm.write(pformat(layer))
            layer_number+=1


    @staticmethod
    def unnest_tensor(values: torch.Tensor):
        """
        If the input is a NestedTensor, unbinds the values, converts to NumPy, moves to the CPU,
        and returns a list.

        Args:
            values (torch.Tensor): the input to unbind

        Returns:
            (list | torch.Tensor): the original tensor (if not a NestedTensor) 
                or the unbound values as a list
        """
        if isinstance(values, torch.Tensor) and values.is_nested:
            return [
                x.numpy() for x in values.cpu().unbind()
            ]
        else:
            return values
