# sparseypy.core.hpo_objectives package

## Submodules

## sparseypy.core.hpo_objectives.hpo_objective module

### *class* sparseypy.core.hpo_objectives.hpo_objective.HPOObjective(hpo_config: dict)

Bases: `object`

#### average_nested_data(data)

#### combine_metrics(results: [TrainingResult](sparseypy.core.results.md#sparseypy.core.results.training_result.TrainingResult))

Combines multiple metric results into a single scalar value using a specified operation and weights,
averaging values at different levels within each metric. Only metrics specified in the HPO configuration are used.

* **Parameters:**
  **results** -- A list of dictionaries containing metric results.
* **Returns:**
  A single scalar value representing the combined result.

## Module contents
