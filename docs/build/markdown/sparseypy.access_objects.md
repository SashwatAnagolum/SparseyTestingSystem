# sparseypy.access_objects package

## Subpackages

* [sparseypy.access_objects.datasets package](sparseypy.access_objects.datasets.md)
  * [Submodules](sparseypy.access_objects.datasets.md#submodules)
  * [sparseypy.access_objects.datasets.built_in_dataset module](sparseypy.access_objects.datasets.md#module-sparseypy.access_objects.datasets.built_in_dataset)
    * [`BuiltInDataset`](sparseypy.access_objects.datasets.md#sparseypy.access_objects.datasets.built_in_dataset.BuiltInDataset)
  * [sparseypy.access_objects.datasets.dataset module](sparseypy.access_objects.datasets.md#module-sparseypy.access_objects.datasets.dataset)
    * [`Dataset`](sparseypy.access_objects.datasets.md#sparseypy.access_objects.datasets.dataset.Dataset)
  * [sparseypy.access_objects.datasets.dataset_factory module](sparseypy.access_objects.datasets.md#module-sparseypy.access_objects.datasets.dataset_factory)
    * [`DatasetFactory`](sparseypy.access_objects.datasets.md#sparseypy.access_objects.datasets.dataset_factory.DatasetFactory)
      * [`DatasetFactory.allowed_modules`](sparseypy.access_objects.datasets.md#sparseypy.access_objects.datasets.dataset_factory.DatasetFactory.allowed_modules)
      * [`DatasetFactory.create_dataset()`](sparseypy.access_objects.datasets.md#sparseypy.access_objects.datasets.dataset_factory.DatasetFactory.create_dataset)
      * [`DatasetFactory.get_dataset_class()`](sparseypy.access_objects.datasets.md#sparseypy.access_objects.datasets.dataset_factory.DatasetFactory.get_dataset_class)
  * [sparseypy.access_objects.datasets.image_dataset module](sparseypy.access_objects.datasets.md#module-sparseypy.access_objects.datasets.image_dataset)
    * [`ImageDataset`](sparseypy.access_objects.datasets.md#sparseypy.access_objects.datasets.image_dataset.ImageDataset)
  * [sparseypy.access_objects.datasets.preprocessed_dataset module](sparseypy.access_objects.datasets.md#module-sparseypy.access_objects.datasets.preprocessed_dataset)
    * [`PreprocessedDataset`](sparseypy.access_objects.datasets.md#sparseypy.access_objects.datasets.preprocessed_dataset.PreprocessedDataset)
      * [`PreprocessedDataset.preprocessed_dir`](sparseypy.access_objects.datasets.md#sparseypy.access_objects.datasets.preprocessed_dataset.PreprocessedDataset.preprocessed_dir)
      * [`PreprocessedDataset.preprocessing_stack`](sparseypy.access_objects.datasets.md#sparseypy.access_objects.datasets.preprocessed_dataset.PreprocessedDataset.preprocessing_stack)
      * [`PreprocessedDataset.preprocessed_flags`](sparseypy.access_objects.datasets.md#sparseypy.access_objects.datasets.preprocessed_dataset.PreprocessedDataset.preprocessed_flags)
  * [sparseypy.access_objects.datasets.sparsey_dataset module](sparseypy.access_objects.datasets.md#module-sparseypy.access_objects.datasets.sparsey_dataset)
    * [`SparseyDataset`](sparseypy.access_objects.datasets.md#sparseypy.access_objects.datasets.sparsey_dataset.SparseyDataset)
  * [Module contents](sparseypy.access_objects.datasets.md#module-sparseypy.access_objects.datasets)
* [sparseypy.access_objects.hpo_runs package](sparseypy.access_objects.hpo_runs.md)
  * [Submodules](sparseypy.access_objects.hpo_runs.md#submodules)
  * [sparseypy.access_objects.hpo_runs.hpo_run module](sparseypy.access_objects.hpo_runs.md#module-sparseypy.access_objects.hpo_runs.hpo_run)
    * [`HPORun`](sparseypy.access_objects.hpo_runs.md#sparseypy.access_objects.hpo_runs.hpo_run.HPORun)
      * [`HPORun.num_steps_to_perform`](sparseypy.access_objects.hpo_runs.md#sparseypy.access_objects.hpo_runs.hpo_run.HPORun.num_steps_to_perform)
      * [`HPORun.check_is_value_constraint()`](sparseypy.access_objects.hpo_runs.md#sparseypy.access_objects.hpo_runs.hpo_run.HPORun.check_is_value_constraint)
      * [`HPORun.close_tqdm()`](sparseypy.access_objects.hpo_runs.md#sparseypy.access_objects.hpo_runs.hpo_run.HPORun.close_tqdm)
      * [`HPORun.construct_sweep_config()`](sparseypy.access_objects.hpo_runs.md#sparseypy.access_objects.hpo_runs.hpo_run.HPORun.construct_sweep_config)
      * [`HPORun.extract_hyperparams()`](sparseypy.access_objects.hpo_runs.md#sparseypy.access_objects.hpo_runs.hpo_run.HPORun.extract_hyperparams)
      * [`HPORun.generate_model_config()`](sparseypy.access_objects.hpo_runs.md#sparseypy.access_objects.hpo_runs.hpo_run.HPORun.generate_model_config)
      * [`HPORun.generate_trainer_config()`](sparseypy.access_objects.hpo_runs.md#sparseypy.access_objects.hpo_runs.hpo_run.HPORun.generate_trainer_config)
      * [`HPORun.run_sweep()`](sparseypy.access_objects.hpo_runs.md#sparseypy.access_objects.hpo_runs.hpo_run.HPORun.run_sweep)
      * [`HPORun.step()`](sparseypy.access_objects.hpo_runs.md#sparseypy.access_objects.hpo_runs.hpo_run.HPORun.step)
      * [`HPORun.tqdm_bar`](sparseypy.access_objects.hpo_runs.md#sparseypy.access_objects.hpo_runs.hpo_run.HPORun.tqdm_bar)
  * [Module contents](sparseypy.access_objects.hpo_runs.md#module-sparseypy.access_objects.hpo_runs)

## Module contents

Init: initialization for the access_objects module
