# -*- coding: utf-8 -*-

"""
Train Model: script to train models.
"""


import pprint
from tqdm import tqdm
import warnings

import wandb

from sparseypy.access_objects.training_recipes.training_recipe_builder import TrainingRecipeBuilder
from sparseypy.core.data_storage_retrieval import DataFetcher, DataStorer

# Weights & Biases attempts to read tqdm updates from the console even after the last run
# in an HPO sweep finishes, causing an unnecessary UserWarning when it attempts to log data
# to a nonexistent run; this is a Weights & Biases issue that does not affect system
# functionality so we ignore this warning
warnings.filterwarnings(
    "ignore",
    message="Run (.*) is finished. The call to `_console_raw_callback` will be ignored."
    )


def train_model(model_config: dict, trainer_config: dict,
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

    # initialize the DataStorer (logs into W&B and Firestore)
    DataStorer.configure(system_config)

    wandb.init(
        project=system_config["wandb"]["project_name"],
        allow_val_change=True
    )

    reload_model = False

    if isinstance(model_config, str):
        model_config, model_weights = DataFetcher().get_model_data(model_config)
        reload_model = True

    trainer = TrainingRecipeBuilder.build_training_recipe(
        model_config, dataset_config, preprocessing_config,
        trainer_config
    )

    if reload_model:
        trainer.model.load_state_dict(model_weights)

    # print training run summary
    met_separator = "\n* "
    tqdm.write(f"""
TRAINING RUN SUMMARY
Dataset type: {dataset_config['dataset_type']}
Batch size: {trainer_config['dataloader']['batch_size']}
Number of batches: {trainer.num_batches}
Selected metrics: 
* {met_separator.join([x["name"] for x in trainer_config["metrics"]])}
""")

    for epoch in tqdm(range(trainer_config['training']['num_epochs']), desc="Epochs", position=0):
        is_epoch_done = False
        trainer.model.train()
        batch_number = 1

        # perform training
        with tqdm(total=trainer.num_batches, desc="Training", leave=False, position=1) as pbar:
            while not is_epoch_done:
                output, is_epoch_done = trainer.step(training=True)
                tqdm.write(f"\n\nTraining results - INPUT {batch_number}\n--------------------")
                metric_str = pprint.pformat(output.get_metrics())
                tqdm.write(metric_str)
                batch_number+=1
                pbar.update(1)
        # summarize the best training steps
        train_summary = trainer.get_summary("training")
        tqdm.write("\n\nTRAINING - SUMMARY\n")
        tqdm.write("Best metric steps:")
        for metric, val in train_summary.best_steps.items():
            tqdm.write(f"* {metric:>25}: step {val['best_index']:<5} (using {val['best_function'].__name__})")


        trainer.model.eval()
        is_epoch_done = False
        batch_number = 1

        # perform evaluation
        with tqdm(total=trainer.num_batches, desc="Evaluation", leave=False, position=1) as pbar:
            while not is_epoch_done:
                # validate this logic VS the design of our EvaluationResult
                # this looks like old-style logic for which we should remove the "while"
                output, is_epoch_done = trainer.step(training=False)
                tqdm.write(f"\n\nEvaluation results - INPUT {batch_number}\n--------------------")
                metric_str = pprint.pformat(output.get_metrics())
                tqdm.write(metric_str)
                batch_number+=1
                pbar.update(1)

        # print summary here in model script
        # if not printing you still need to call this to finalize the results
        eval_summary = trainer.get_summary("evaluation")

        tqdm.write("\n\nEVALUATION - SUMMARY\n")
        tqdm.write("Best metric steps:")
        for metric, val in eval_summary.best_steps.items():
            tqdm.write(f"* {metric:>25}: step {val['best_index']:<5} (using {val['best_function'].__name__})")

        wandb.finish()
