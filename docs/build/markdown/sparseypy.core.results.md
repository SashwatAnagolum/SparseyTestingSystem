# sparseypy.core.results package

## Submodules

## sparseypy.core.results.hpo_result module

### *class* sparseypy.core.results.hpo_result.HPOResult(configs: dict, id: str, name: str)

Bases: [`Result`](#sparseypy.core.results.result.Result)

#### add_step(step: [HPOStepResult](#sparseypy.core.results.hpo_step_result.HPOStepResult))

#### get_steps()

#### get_top_k_steps(k: int)

## sparseypy.core.results.hpo_step_result module

### *class* sparseypy.core.results.hpo_step_result.HPOStepResult(parent_run: str, id: str, configs: dict)

Bases: [`Result`](#sparseypy.core.results.result.Result)

#### get_eval_results()

#### get_objective()

#### get_training_results()

#### populate(objective: dict, training_results: [TrainingResult](#sparseypy.core.results.training_result.TrainingResult), eval_results: [TrainingResult](#sparseypy.core.results.training_result.TrainingResult))

## sparseypy.core.results.result module

### *class* sparseypy.core.results.result.Result

Bases: `object`

#### mark_finished()

## sparseypy.core.results.training_result module

### *class* sparseypy.core.results.training_result.TrainingResult(id: str, result_type: str, resolution: str, metrics: list[[Metric](sparseypy.core.metrics.md#sparseypy.core.metrics.metrics.Metric)], configs: dict | None = None)

Bases: [`Result`](#sparseypy.core.results.result.Result)

#### add_config(name, config)

#### add_step(step: [TrainingStepResult](#sparseypy.core.results.training_step_result.TrainingStepResult))

#### get_best_step(metric: str)

#### get_configs()

#### get_step(index: int)

#### get_steps()

## sparseypy.core.results.training_step_result module

### *class* sparseypy.core.results.training_step_result.TrainingStepResult(resolution: str)

Bases: [`Result`](#sparseypy.core.results.result.Result)

#### add_metric(name: str, values: list)

#### get_metric(name: str)

#### get_metrics()

## Module contents
