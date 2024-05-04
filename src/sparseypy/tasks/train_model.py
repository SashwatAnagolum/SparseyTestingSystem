# -*- coding: utf-8 -*-

"""
Train Model: script to train models.
"""

from copy import deepcopy
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

def train_model(model_config: dict, trainer_config: dict,
                preprocessing_config: dict, training_dataset_config: dict,
                evaluation_dataset_config: dict, system_config: dict) -> None:
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
        training_dataset_config (dict): config info used to
            build the training dataset object.
        evaluation_dataset_config (dict): config info used to
            build the evaluation dataset object.
        system_config (dict): config info for the overall system
    """

    # silence WandB if requested by the user
    if system_config["wandb"]["silent"]:
        os.environ["WANDB_SILENT"] = "true"

    # initialize the DataStorer (logs into W&B and Firestore)
    tqdm.write("Connecting to Weights & Biases...")
    DataStorer.configure(system_config)
    df = DataFetcher(system_config)

    # break out the individual model config layers for better hyperparameter
    # access on W&B (without this you can't use the layer HP in the visualizer)
    wandb_model_config = deepcopy(model_config)
    # convert from array of layers to dict of "layer_1", "layer_2", ...
    wandb_model_config["layers"] = {
        f"layer_{i+1}": layer
        for i, layer in enumerate(wandb_model_config["layers"])
    }

    wandb.init(
        allow_val_change=True,
        config={
            'dataset': training_dataset_config,
            'model': wandb_model_config,
            'training_recipe': trainer_config,
            'preprocessing': preprocessing_config
        },
        dir=system_config['wandb']['local_log_directory'],
        job_type="train",
        name=trainer_config["run_name"],
        notes=trainer_config['description'],
        project=system_config["wandb"]["project_name"],
    )

    reload_model = False

    if isinstance(model_config, str):
        model_config, model_weights = df.get_model_data(model_config)
        reload_model = True

    trainer = TrainingRecipeBuilder.build_training_recipe(
        model_config, training_dataset_config,
        evaluation_dataset_config, preprocessing_config,
        trainer_config
    )

    if reload_model:
        trainer.model.load_state_dict(model_weights)

    Printer.print_pre_training_summary(
        training_dataset_config=training_dataset_config,
        evaluation_dataset_config=evaluation_dataset_config,
        trainer_config=trainer_config,
        training_num_batches=trainer.training_num_batches,
        eval_num_batches=trainer.eval_num_batches
    )

    Printer.print_run_start_message(
        run_name=wandb.run.name,
        run_url=wandb.run.url,
        phase="training"
    )

    for epoch in tqdm(range(trainer_config['training']['num_epochs']), desc="Epochs", position=0):
        is_epoch_done = False
        trainer.model.train()
        batch_number = 1

        # perform training
        with tqdm(
            total=trainer.training_num_batches,
            desc="Training",
            leave=False,
            position=1,
            unit="input"if trainer_config['training']['dataloader']['batch_size'] == 1 else "batch",
            miniters=int(trainer.training_num_batches/100)
        ) as pbar:
            while not is_epoch_done:
                output, is_epoch_done = trainer.step(training=True)
                # only print metric values to the console if explicitly requested by
                # the user (for performance reasons--metrics print a lot of data)
                if system_config['console'].get('print_metric_values', False):
                    Printer.print_step_metrics(
                        step_data=output,
                        batch_number=batch_number,
                        max_batch_size=trainer_config['training']['dataloader']['batch_size'],
                        step_type="training"
                    )
                batch_number+=1
                pbar.update(1)

        # summarize the best training steps
        train_summary = trainer.get_summary("training")
        Printer.print_best_steps(results=train_summary, run_type="training")

        trainer.model.eval()
        is_epoch_done = False
        batch_number = 1

        # begin logging a new evaluation run
        # save the run id
        run_name = wandb.run.name
        run_group = get_update_group(wandb.run.path)
        train_url = wandb.run.url
        # end the current run
        tqdm.write("\nFinalizing training results...\n")
        wandb.finish()

        # start a new evaluation run
        wandb.init(
            allow_val_change=True,
            config={
                'dataset': evaluation_dataset_config,
                'model': wandb_model_config,
                'training_recipe': trainer_config,
                'preprocessing': preprocessing_config
            },
            dir=system_config['wandb']['local_log_directory'],
            group=run_group,
            job_type="eval",
            name=run_name + "-eval",
            notes=trainer_config['description'],
            project=system_config["wandb"]["project_name"],
        )

        Printer.print_run_start_message(
            run_name=wandb.run.name,
            run_url=wandb.run.url,
            phase="evaluation"
        )

        # perform evaluation
        with tqdm(
            total=trainer.eval_num_batches,
            desc="Evaluation",
            leave=False,
            position=1,
            unit="input"if trainer_config['eval']['dataloader']['batch_size'] == 1 else "batch",
            miniters=int(trainer.eval_num_batches/100)
        ) as pbar:
            while not is_epoch_done:
                # validate this logic VS the design of our EvaluationResult
                # this looks like old-style logic for which we should remove the "while"
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

        # get summary
        # if not printing you still need to call this to finalize the results
        eval_summary = trainer.get_summary("evaluation")
        # then print the best steps
        Printer.print_best_steps(eval_summary, run_type="evaluation")

    tqdm.write("\nFinalizing evaluation results...")
    eval_url = wandb.run.get_url()
    model_name = model_config.get('model_name', wandb.run.id+'-model')

    wandb_run_dir = wandb.run.dir.removesuffix('files')

    wandb.finish()

    if system_config['wandb'].get('remove_local_files', False):
        shutil.rmtree(wandb_run_dir)
        tqdm.write("Removed local temporary files.")

    Printer.print_post_train_model_summary(model_name, run_group, train_url, eval_url)


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
    else:
        return source_run.group
