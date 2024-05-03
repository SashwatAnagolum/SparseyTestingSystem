# -*- coding: utf-8 -*-

"""
Evaluate Model: script to reload and evaluate models.
"""

import os
import shutil
import warnings

from tqdm import tqdm
import wandb

from sparseypy.access_objects.training_recipes.training_recipe_builder import TrainingRecipeBuilder
from sparseypy.core.data_storage_retrieval import DataFetcher, DataStorer
from sparseypy.core.printing import Printer

# Weights & Biases attempts to read tqdm updates from the console even after the last run
# in an HPO sweep finishes, causing an unnecessary UserWarning when it attempts to log data
# to a nonexistent run; this is a Weights & Biases issue that does not affect system
# functionality so we ignore this warning
warnings.filterwarnings(
    "ignore",
    message="Run (.*) is finished. The call to `_console_raw_callback` will be ignored."
    )
# PyTorch nested tensors are in beta and print a warning about API changes
warnings.filterwarnings(
    "ignore",
    message=r"The PyTorch API of nested tensors is in prototype stage and will change in the.+"
)

def evaluate_model(model_name: str, trainer_config: dict,
                preprocessing_config: dict, dataset_config: dict,
                system_config: dict):
    """
    Builds a model using the model_config, and trains
    it using the trainer built using trainer_config on 
    the dataset built using dataset_config, with preprocessing
    defined in preprocessing_config.
    Args:
        model_config (dict): config info to build the model.
        trainer_config (dict): config info to build the trainer.
        preprocessing_config (dict): config info to build the
            preprocessing stack.
        dataset_config (dict): config info to build the dataset
            to train on.
        system_config (dict): config info for the overall system
    """

    # silence WandB if requested by the user
    if system_config["wandb"]["silent"]:
        os.environ["WANDB_SILENT"] = "true"

    # check for match_accuracy
    for i, m in enumerate(trainer_config["metrics"]):
        if m["name"] == "match_accuracy":
            tqdm.write("WARNING: match_accuracy is not supported for reloaded models. Removing.")
            del trainer_config["metrics"][i]
            break

    # initialize the DataStorer (logs into W&B and Firestore)
    tqdm.write("Connecting to Weights & Biases...")
    DataStorer.configure(system_config)

    df = DataFetcher(system_config)

    # fetch the required group to associate this evaluation with its parent training run
    source_path = df.get_model_source_path(model_name)
    source_group = get_update_group(source_path)

    wandb.init(
        allow_val_change=True,
        dir=system_config['wandb']['local_log_directory'],
        group=source_group,
        job_type="eval",
        name=trainer_config["run_name"],
        notes=trainer_config['description'],
        project=system_config["wandb"]["project_name"]
    )

    model_config, model_weights = df.get_model_data(model_name)

    trainer = TrainingRecipeBuilder.build_training_recipe(
        model_config=model_config,
        evaluation_dataset=None,
        evaluation_dataset_config=dataset_config,
        preprocessing_config=preprocessing_config,
        training_dataset=None,
        training_dataset_config=None,
        train_config=trainer_config,
    )

    trainer.model.load_state_dict(model_weights)

    # print training run summary
    Printer.print_pre_evaluate_model_summary(dataset_config, trainer_config,
                                             model_name, trainer.eval_num_batches)

    Printer.print_run_start_message(
        run_name=wandb.run.name,
        run_url=wandb.run.url,
        phase="evaluation"
    )

    for epoch in tqdm(range(trainer_config['training']['num_epochs']), desc="Epochs", position=0):
        trainer.model.eval()
        is_epoch_done = False
        batch_number = 1

        # perform evaluation
        with tqdm(total=trainer.eval_num_batches, desc="Evaluation", leave=False, position=1) as pbar:
            while not is_epoch_done:
                output, is_epoch_done = trainer.step(training=False)
                # only print metric values to the console if explicitly requested by
                # the user (for performance reasons--metrics print a lot of data)
                if system_config['console'].get('print_metric_values', False):
                    Printer.print_step_metrics(
                        step_data=output,
                        batch_number=batch_number,
                        max_batch_size=trainer_config['eval']['dataloader']['batch_size'],
                        step_type="evaluation"
                    )
                batch_number+=1
                pbar.update(1)

        # get the summary results
        # (if not printing you still need to call this to finalize the results)
        tqdm.write("\nLogging evaluation results...")
        eval_summary = trainer.get_summary("evaluation")
        # then print them
        Printer.print_best_steps(results=eval_summary, run_type="evaluation")


    tqdm.write("\nFinalizing results...")
    run_url = wandb.run.get_url()
    model_name = model_config.get('model_name', wandb.run.id+'-model')

    wandb_run_dir = wandb.run.dir.removesuffix('files')

    wandb.finish()

    if system_config['wandb'].get('remove_local_files', False):
        shutil.rmtree(wandb_run_dir)
        tqdm.write("Removed local temporary files.")

    Printer.print_post_evaluate_model_summary(model_name, run_url, source_group)


def get_update_group(source_run_path: str) -> str:
    """
    Fetches the existing group of the indicated run, if any. If
    there is no existing group, creates a new one using the name
    of the source run and returns that.
    Args:
        source_run_path (str): the full path to the source run.
    Returns:
        str: the name of the group
    """
    api = wandb.Api()

    source_run = api.run(source_run_path)

    if source_run.group is None:
        source_run.group = source_run.name
        source_run.update()

    return source_run.group
