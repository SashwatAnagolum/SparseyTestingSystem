# sparseypy.cli.config_validation.saved_schemas package

## Subpackages

* [sparseypy.cli.config_validation.saved_schemas.dataset package](sparseypy.cli.config_validation.saved_schemas.dataset.md)
  * [Submodules](sparseypy.cli.config_validation.saved_schemas.dataset.md#submodules)
  * [sparseypy.cli.config_validation.saved_schemas.dataset.built_in module](sparseypy.cli.config_validation.saved_schemas.dataset.md#module-sparseypy.cli.config_validation.saved_schemas.dataset.built_in)
    * [`BuiltInDatasetSchema`](sparseypy.cli.config_validation.saved_schemas.dataset.md#sparseypy.cli.config_validation.saved_schemas.dataset.built_in.BuiltInDatasetSchema)
      * [`BuiltInDatasetSchema.build_schema()`](sparseypy.cli.config_validation.saved_schemas.dataset.md#sparseypy.cli.config_validation.saved_schemas.dataset.built_in.BuiltInDatasetSchema.build_schema)
      * [`BuiltInDatasetSchema.check_if_dataset_exists()`](sparseypy.cli.config_validation.saved_schemas.dataset.md#sparseypy.cli.config_validation.saved_schemas.dataset.built_in.BuiltInDatasetSchema.check_if_dataset_exists)
      * [`BuiltInDatasetSchema.check_if_transform_exists()`](sparseypy.cli.config_validation.saved_schemas.dataset.md#sparseypy.cli.config_validation.saved_schemas.dataset.built_in.BuiltInDatasetSchema.check_if_transform_exists)
      * [`BuiltInDatasetSchema.convert_transform_name()`](sparseypy.cli.config_validation.saved_schemas.dataset.md#sparseypy.cli.config_validation.saved_schemas.dataset.built_in.BuiltInDatasetSchema.convert_transform_name)
      * [`BuiltInDatasetSchema.extract_schema_params()`](sparseypy.cli.config_validation.saved_schemas.dataset.md#sparseypy.cli.config_validation.saved_schemas.dataset.built_in.BuiltInDatasetSchema.extract_schema_params)
      * [`BuiltInDatasetSchema.transform_schema()`](sparseypy.cli.config_validation.saved_schemas.dataset.md#sparseypy.cli.config_validation.saved_schemas.dataset.built_in.BuiltInDatasetSchema.transform_schema)
  * [sparseypy.cli.config_validation.saved_schemas.dataset.image module](sparseypy.cli.config_validation.saved_schemas.dataset.md#module-sparseypy.cli.config_validation.saved_schemas.dataset.image)
    * [`ImageDatasetSchema`](sparseypy.cli.config_validation.saved_schemas.dataset.md#sparseypy.cli.config_validation.saved_schemas.dataset.image.ImageDatasetSchema)
      * [`ImageDatasetSchema.build_schema()`](sparseypy.cli.config_validation.saved_schemas.dataset.md#sparseypy.cli.config_validation.saved_schemas.dataset.image.ImageDatasetSchema.build_schema)
      * [`ImageDatasetSchema.extract_schema_params()`](sparseypy.cli.config_validation.saved_schemas.dataset.md#sparseypy.cli.config_validation.saved_schemas.dataset.image.ImageDatasetSchema.extract_schema_params)
      * [`ImageDatasetSchema.transform_schema()`](sparseypy.cli.config_validation.saved_schemas.dataset.md#sparseypy.cli.config_validation.saved_schemas.dataset.image.ImageDatasetSchema.transform_schema)
  * [sparseypy.cli.config_validation.saved_schemas.dataset.sparsey module](sparseypy.cli.config_validation.saved_schemas.dataset.md#module-sparseypy.cli.config_validation.saved_schemas.dataset.sparsey)
    * [`SparseyDatasetSchema`](sparseypy.cli.config_validation.saved_schemas.dataset.md#sparseypy.cli.config_validation.saved_schemas.dataset.sparsey.SparseyDatasetSchema)
      * [`SparseyDatasetSchema.build_schema()`](sparseypy.cli.config_validation.saved_schemas.dataset.md#sparseypy.cli.config_validation.saved_schemas.dataset.sparsey.SparseyDatasetSchema.build_schema)
      * [`SparseyDatasetSchema.extract_schema_params()`](sparseypy.cli.config_validation.saved_schemas.dataset.md#sparseypy.cli.config_validation.saved_schemas.dataset.sparsey.SparseyDatasetSchema.extract_schema_params)
      * [`SparseyDatasetSchema.transform_schema()`](sparseypy.cli.config_validation.saved_schemas.dataset.md#sparseypy.cli.config_validation.saved_schemas.dataset.sparsey.SparseyDatasetSchema.transform_schema)
  * [Module contents](sparseypy.cli.config_validation.saved_schemas.dataset.md#module-sparseypy.cli.config_validation.saved_schemas.dataset)
* [sparseypy.cli.config_validation.saved_schemas.db_adapter package](sparseypy.cli.config_validation.saved_schemas.db_adapter.md)
  * [Submodules](sparseypy.cli.config_validation.saved_schemas.db_adapter.md#submodules)
  * [sparseypy.cli.config_validation.saved_schemas.db_adapter.firestore module](sparseypy.cli.config_validation.saved_schemas.db_adapter.md#module-sparseypy.cli.config_validation.saved_schemas.db_adapter.firestore)
    * [`FirestoreDbAdapterSchema`](sparseypy.cli.config_validation.saved_schemas.db_adapter.md#sparseypy.cli.config_validation.saved_schemas.db_adapter.firestore.FirestoreDbAdapterSchema)
      * [`FirestoreDbAdapterSchema.build_schema()`](sparseypy.cli.config_validation.saved_schemas.db_adapter.md#sparseypy.cli.config_validation.saved_schemas.db_adapter.firestore.FirestoreDbAdapterSchema.build_schema)
  * [Module contents](sparseypy.cli.config_validation.saved_schemas.db_adapter.md#module-sparseypy.cli.config_validation.saved_schemas.db_adapter)
* [sparseypy.cli.config_validation.saved_schemas.hpo package](sparseypy.cli.config_validation.saved_schemas.hpo.md)
  * [Submodules](sparseypy.cli.config_validation.saved_schemas.hpo.md#submodules)
  * [sparseypy.cli.config_validation.saved_schemas.hpo.default module](sparseypy.cli.config_validation.saved_schemas.hpo.md#module-sparseypy.cli.config_validation.saved_schemas.hpo.default)
    * [`DefaultHpoSchema`](sparseypy.cli.config_validation.saved_schemas.hpo.md#sparseypy.cli.config_validation.saved_schemas.hpo.default.DefaultHpoSchema)
      * [`DefaultHpoSchema.build_precheck_schema()`](sparseypy.cli.config_validation.saved_schemas.hpo.md#sparseypy.cli.config_validation.saved_schemas.hpo.default.DefaultHpoSchema.build_precheck_schema)
      * [`DefaultHpoSchema.build_schema()`](sparseypy.cli.config_validation.saved_schemas.hpo.md#sparseypy.cli.config_validation.saved_schemas.hpo.default.DefaultHpoSchema.build_schema)
      * [`DefaultHpoSchema.check_if_metric_exists()`](sparseypy.cli.config_validation.saved_schemas.hpo.md#sparseypy.cli.config_validation.saved_schemas.hpo.default.DefaultHpoSchema.check_if_metric_exists)
      * [`DefaultHpoSchema.check_if_model_family_exists()`](sparseypy.cli.config_validation.saved_schemas.hpo.md#sparseypy.cli.config_validation.saved_schemas.hpo.default.DefaultHpoSchema.check_if_model_family_exists)
      * [`DefaultHpoSchema.check_optimized_hyperparams_validity()`](sparseypy.cli.config_validation.saved_schemas.hpo.md#sparseypy.cli.config_validation.saved_schemas.hpo.default.DefaultHpoSchema.check_optimized_hyperparams_validity)
      * [`DefaultHpoSchema.extract_schema_params()`](sparseypy.cli.config_validation.saved_schemas.hpo.md#sparseypy.cli.config_validation.saved_schemas.hpo.default.DefaultHpoSchema.extract_schema_params)
      * [`DefaultHpoSchema.get_max_num_layers()`](sparseypy.cli.config_validation.saved_schemas.hpo.md#sparseypy.cli.config_validation.saved_schemas.hpo.default.DefaultHpoSchema.get_max_num_layers)
      * [`DefaultHpoSchema.has_enough_layer_configs()`](sparseypy.cli.config_validation.saved_schemas.hpo.md#sparseypy.cli.config_validation.saved_schemas.hpo.default.DefaultHpoSchema.has_enough_layer_configs)
      * [`DefaultHpoSchema.validate_metrics_in_order()`](sparseypy.cli.config_validation.saved_schemas.hpo.md#sparseypy.cli.config_validation.saved_schemas.hpo.default.DefaultHpoSchema.validate_metrics_in_order)
  * [Module contents](sparseypy.cli.config_validation.saved_schemas.hpo.md#module-sparseypy.cli.config_validation.saved_schemas.hpo)
* [sparseypy.cli.config_validation.saved_schemas.metric package](sparseypy.cli.config_validation.saved_schemas.metric.md)
  * [Submodules](sparseypy.cli.config_validation.saved_schemas.metric.md#submodules)
  * [sparseypy.cli.config_validation.saved_schemas.metric.basis_average module](sparseypy.cli.config_validation.saved_schemas.metric.md#module-sparseypy.cli.config_validation.saved_schemas.metric.basis_average)
    * [`BasisAverageMetricSchema`](sparseypy.cli.config_validation.saved_schemas.metric.md#sparseypy.cli.config_validation.saved_schemas.metric.basis_average.BasisAverageMetricSchema)
      * [`BasisAverageMetricSchema.build_schema()`](sparseypy.cli.config_validation.saved_schemas.metric.md#sparseypy.cli.config_validation.saved_schemas.metric.basis_average.BasisAverageMetricSchema.build_schema)
  * [sparseypy.cli.config_validation.saved_schemas.metric.basis_set_size module](sparseypy.cli.config_validation.saved_schemas.metric.md#module-sparseypy.cli.config_validation.saved_schemas.metric.basis_set_size)
    * [`BasisSetSizeMetricSchema`](sparseypy.cli.config_validation.saved_schemas.metric.md#sparseypy.cli.config_validation.saved_schemas.metric.basis_set_size.BasisSetSizeMetricSchema)
      * [`BasisSetSizeMetricSchema.build_schema()`](sparseypy.cli.config_validation.saved_schemas.metric.md#sparseypy.cli.config_validation.saved_schemas.metric.basis_set_size.BasisSetSizeMetricSchema.build_schema)
  * [sparseypy.cli.config_validation.saved_schemas.metric.basis_set_size_increase module](sparseypy.cli.config_validation.saved_schemas.metric.md#module-sparseypy.cli.config_validation.saved_schemas.metric.basis_set_size_increase)
    * [`BasisSetSizeIncreaseMetricSchema`](sparseypy.cli.config_validation.saved_schemas.metric.md#sparseypy.cli.config_validation.saved_schemas.metric.basis_set_size_increase.BasisSetSizeIncreaseMetricSchema)
      * [`BasisSetSizeIncreaseMetricSchema.build_schema()`](sparseypy.cli.config_validation.saved_schemas.metric.md#sparseypy.cli.config_validation.saved_schemas.metric.basis_set_size_increase.BasisSetSizeIncreaseMetricSchema.build_schema)
  * [sparseypy.cli.config_validation.saved_schemas.metric.feature_coverage module](sparseypy.cli.config_validation.saved_schemas.metric.md#module-sparseypy.cli.config_validation.saved_schemas.metric.feature_coverage)
    * [`FeatureCoverageMetricSchema`](sparseypy.cli.config_validation.saved_schemas.metric.md#sparseypy.cli.config_validation.saved_schemas.metric.feature_coverage.FeatureCoverageMetricSchema)
      * [`FeatureCoverageMetricSchema.build_schema()`](sparseypy.cli.config_validation.saved_schemas.metric.md#sparseypy.cli.config_validation.saved_schemas.metric.feature_coverage.FeatureCoverageMetricSchema.build_schema)
  * [sparseypy.cli.config_validation.saved_schemas.metric.match_accuracy module](sparseypy.cli.config_validation.saved_schemas.metric.md#module-sparseypy.cli.config_validation.saved_schemas.metric.match_accuracy)
    * [`MatchAccuracyMetricSchema`](sparseypy.cli.config_validation.saved_schemas.metric.md#sparseypy.cli.config_validation.saved_schemas.metric.match_accuracy.MatchAccuracyMetricSchema)
      * [`MatchAccuracyMetricSchema.build_schema()`](sparseypy.cli.config_validation.saved_schemas.metric.md#sparseypy.cli.config_validation.saved_schemas.metric.match_accuracy.MatchAccuracyMetricSchema.build_schema)
  * [sparseypy.cli.config_validation.saved_schemas.metric.num_activations module](sparseypy.cli.config_validation.saved_schemas.metric.md#module-sparseypy.cli.config_validation.saved_schemas.metric.num_activations)
    * [`NumActivationsMetricSchema`](sparseypy.cli.config_validation.saved_schemas.metric.md#sparseypy.cli.config_validation.saved_schemas.metric.num_activations.NumActivationsMetricSchema)
      * [`NumActivationsMetricSchema.build_schema()`](sparseypy.cli.config_validation.saved_schemas.metric.md#sparseypy.cli.config_validation.saved_schemas.metric.num_activations.NumActivationsMetricSchema.build_schema)
      * [`NumActivationsMetricSchema.extract_schema_params()`](sparseypy.cli.config_validation.saved_schemas.metric.md#sparseypy.cli.config_validation.saved_schemas.metric.num_activations.NumActivationsMetricSchema.extract_schema_params)
      * [`NumActivationsMetricSchema.transform_schema()`](sparseypy.cli.config_validation.saved_schemas.metric.md#sparseypy.cli.config_validation.saved_schemas.metric.num_activations.NumActivationsMetricSchema.transform_schema)
  * [Module contents](sparseypy.cli.config_validation.saved_schemas.metric.md#module-sparseypy.cli.config_validation.saved_schemas.metric)
* [sparseypy.cli.config_validation.saved_schemas.model package](sparseypy.cli.config_validation.saved_schemas.model.md)
  * [Submodules](sparseypy.cli.config_validation.saved_schemas.model.md#submodules)
  * [sparseypy.cli.config_validation.saved_schemas.model.sparsey module](sparseypy.cli.config_validation.saved_schemas.model.md#module-sparseypy.cli.config_validation.saved_schemas.model.sparsey)
    * [`SparseyModelSchema`](sparseypy.cli.config_validation.saved_schemas.model.md#sparseypy.cli.config_validation.saved_schemas.model.sparsey.SparseyModelSchema)
      * [`SparseyModelSchema.build_schema()`](sparseypy.cli.config_validation.saved_schemas.model.md#sparseypy.cli.config_validation.saved_schemas.model.sparsey.SparseyModelSchema.build_schema)
      * [`SparseyModelSchema.check_if_hook_exists()`](sparseypy.cli.config_validation.saved_schemas.model.md#sparseypy.cli.config_validation.saved_schemas.model.sparsey.SparseyModelSchema.check_if_hook_exists)
      * [`SparseyModelSchema.compute_factor_pair()`](sparseypy.cli.config_validation.saved_schemas.model.md#sparseypy.cli.config_validation.saved_schemas.model.sparsey.SparseyModelSchema.compute_factor_pair)
      * [`SparseyModelSchema.compute_grid_size()`](sparseypy.cli.config_validation.saved_schemas.model.md#sparseypy.cli.config_validation.saved_schemas.model.sparsey.SparseyModelSchema.compute_grid_size)
      * [`SparseyModelSchema.extract_schema_params()`](sparseypy.cli.config_validation.saved_schemas.model.md#sparseypy.cli.config_validation.saved_schemas.model.sparsey.SparseyModelSchema.extract_schema_params)
      * [`SparseyModelSchema.transform_schema()`](sparseypy.cli.config_validation.saved_schemas.model.md#sparseypy.cli.config_validation.saved_schemas.model.sparsey.SparseyModelSchema.transform_schema)
  * [Module contents](sparseypy.cli.config_validation.saved_schemas.model.md#module-sparseypy.cli.config_validation.saved_schemas.model)
* [sparseypy.cli.config_validation.saved_schemas.optimizer package](sparseypy.cli.config_validation.saved_schemas.optimizer.md)
  * [Submodules](sparseypy.cli.config_validation.saved_schemas.optimizer.md#submodules)
  * [sparseypy.cli.config_validation.saved_schemas.optimizer.hebbian module](sparseypy.cli.config_validation.saved_schemas.optimizer.md#module-sparseypy.cli.config_validation.saved_schemas.optimizer.hebbian)
    * [`HebbianOptimizerSchema`](sparseypy.cli.config_validation.saved_schemas.optimizer.md#sparseypy.cli.config_validation.saved_schemas.optimizer.hebbian.HebbianOptimizerSchema)
      * [`HebbianOptimizerSchema.build_schema()`](sparseypy.cli.config_validation.saved_schemas.optimizer.md#sparseypy.cli.config_validation.saved_schemas.optimizer.hebbian.HebbianOptimizerSchema.build_schema)
      * [`HebbianOptimizerSchema.extract_schema_params()`](sparseypy.cli.config_validation.saved_schemas.optimizer.md#sparseypy.cli.config_validation.saved_schemas.optimizer.hebbian.HebbianOptimizerSchema.extract_schema_params)
      * [`HebbianOptimizerSchema.transform_schema()`](sparseypy.cli.config_validation.saved_schemas.optimizer.md#sparseypy.cli.config_validation.saved_schemas.optimizer.hebbian.HebbianOptimizerSchema.transform_schema)
  * [Module contents](sparseypy.cli.config_validation.saved_schemas.optimizer.md#module-sparseypy.cli.config_validation.saved_schemas.optimizer)
* [sparseypy.cli.config_validation.saved_schemas.plot package](sparseypy.cli.config_validation.saved_schemas.plot.md)
  * [Module contents](sparseypy.cli.config_validation.saved_schemas.plot.md#module-sparseypy.cli.config_validation.saved_schemas.plot)
* [sparseypy.cli.config_validation.saved_schemas.preprocessing_stack package](sparseypy.cli.config_validation.saved_schemas.preprocessing_stack.md)
  * [Submodules](sparseypy.cli.config_validation.saved_schemas.preprocessing_stack.md#submodules)
  * [sparseypy.cli.config_validation.saved_schemas.preprocessing_stack.default module](sparseypy.cli.config_validation.saved_schemas.preprocessing_stack.md#module-sparseypy.cli.config_validation.saved_schemas.preprocessing_stack.default)
    * [`DefaultPreprocessingStackSchema`](sparseypy.cli.config_validation.saved_schemas.preprocessing_stack.md#sparseypy.cli.config_validation.saved_schemas.preprocessing_stack.default.DefaultPreprocessingStackSchema)
      * [`DefaultPreprocessingStackSchema.build_precheck_schema()`](sparseypy.cli.config_validation.saved_schemas.preprocessing_stack.md#sparseypy.cli.config_validation.saved_schemas.preprocessing_stack.default.DefaultPreprocessingStackSchema.build_precheck_schema)
      * [`DefaultPreprocessingStackSchema.build_schema()`](sparseypy.cli.config_validation.saved_schemas.preprocessing_stack.md#sparseypy.cli.config_validation.saved_schemas.preprocessing_stack.default.DefaultPreprocessingStackSchema.build_schema)
      * [`DefaultPreprocessingStackSchema.check_if_transform_exists()`](sparseypy.cli.config_validation.saved_schemas.preprocessing_stack.md#sparseypy.cli.config_validation.saved_schemas.preprocessing_stack.default.DefaultPreprocessingStackSchema.check_if_transform_exists)
      * [`DefaultPreprocessingStackSchema.check_transform_schema_validity()`](sparseypy.cli.config_validation.saved_schemas.preprocessing_stack.md#sparseypy.cli.config_validation.saved_schemas.preprocessing_stack.default.DefaultPreprocessingStackSchema.check_transform_schema_validity)
      * [`DefaultPreprocessingStackSchema.extract_schema_params()`](sparseypy.cli.config_validation.saved_schemas.preprocessing_stack.md#sparseypy.cli.config_validation.saved_schemas.preprocessing_stack.default.DefaultPreprocessingStackSchema.extract_schema_params)
  * [Module contents](sparseypy.cli.config_validation.saved_schemas.preprocessing_stack.md#module-sparseypy.cli.config_validation.saved_schemas.preprocessing_stack)
* [sparseypy.cli.config_validation.saved_schemas.system package](sparseypy.cli.config_validation.saved_schemas.system.md)
  * [Submodules](sparseypy.cli.config_validation.saved_schemas.system.md#submodules)
  * [sparseypy.cli.config_validation.saved_schemas.system.default module](sparseypy.cli.config_validation.saved_schemas.system.md#module-sparseypy.cli.config_validation.saved_schemas.system.default)
    * [`DefaultSystemSchema`](sparseypy.cli.config_validation.saved_schemas.system.md#sparseypy.cli.config_validation.saved_schemas.system.default.DefaultSystemSchema)
      * [`DefaultSystemSchema.build_precheck_schema()`](sparseypy.cli.config_validation.saved_schemas.system.md#sparseypy.cli.config_validation.saved_schemas.system.default.DefaultSystemSchema.build_precheck_schema)
      * [`DefaultSystemSchema.build_schema()`](sparseypy.cli.config_validation.saved_schemas.system.md#sparseypy.cli.config_validation.saved_schemas.system.default.DefaultSystemSchema.build_schema)
      * [`DefaultSystemSchema.check_if_db_adapter_exists()`](sparseypy.cli.config_validation.saved_schemas.system.md#sparseypy.cli.config_validation.saved_schemas.system.default.DefaultSystemSchema.check_if_db_adapter_exists)
      * [`DefaultSystemSchema.extract_schema_params()`](sparseypy.cli.config_validation.saved_schemas.system.md#sparseypy.cli.config_validation.saved_schemas.system.default.DefaultSystemSchema.extract_schema_params)
      * [`DefaultSystemSchema.make_env_schema()`](sparseypy.cli.config_validation.saved_schemas.system.md#sparseypy.cli.config_validation.saved_schemas.system.default.DefaultSystemSchema.make_env_schema)
  * [Module contents](sparseypy.cli.config_validation.saved_schemas.system.md#module-sparseypy.cli.config_validation.saved_schemas.system)
* [sparseypy.cli.config_validation.saved_schemas.training_recipe package](sparseypy.cli.config_validation.saved_schemas.training_recipe.md)
  * [Submodules](sparseypy.cli.config_validation.saved_schemas.training_recipe.md#submodules)
  * [sparseypy.cli.config_validation.saved_schemas.training_recipe.sparsey module](sparseypy.cli.config_validation.saved_schemas.training_recipe.md#module-sparseypy.cli.config_validation.saved_schemas.training_recipe.sparsey)
    * [`SparseyTrainingRecipeSchema`](sparseypy.cli.config_validation.saved_schemas.training_recipe.md#sparseypy.cli.config_validation.saved_schemas.training_recipe.sparsey.SparseyTrainingRecipeSchema)
      * [`SparseyTrainingRecipeSchema.build_precheck_schema()`](sparseypy.cli.config_validation.saved_schemas.training_recipe.md#sparseypy.cli.config_validation.saved_schemas.training_recipe.sparsey.SparseyTrainingRecipeSchema.build_precheck_schema)
      * [`SparseyTrainingRecipeSchema.build_schema()`](sparseypy.cli.config_validation.saved_schemas.training_recipe.md#sparseypy.cli.config_validation.saved_schemas.training_recipe.sparsey.SparseyTrainingRecipeSchema.build_schema)
      * [`SparseyTrainingRecipeSchema.check_if_metric_exists()`](sparseypy.cli.config_validation.saved_schemas.training_recipe.md#sparseypy.cli.config_validation.saved_schemas.training_recipe.sparsey.SparseyTrainingRecipeSchema.check_if_metric_exists)
      * [`SparseyTrainingRecipeSchema.check_if_optimizer_exists()`](sparseypy.cli.config_validation.saved_schemas.training_recipe.md#sparseypy.cli.config_validation.saved_schemas.training_recipe.sparsey.SparseyTrainingRecipeSchema.check_if_optimizer_exists)
      * [`SparseyTrainingRecipeSchema.extract_schema_params()`](sparseypy.cli.config_validation.saved_schemas.training_recipe.md#sparseypy.cli.config_validation.saved_schemas.training_recipe.sparsey.SparseyTrainingRecipeSchema.extract_schema_params)
      * [`SparseyTrainingRecipeSchema.transform_schema()`](sparseypy.cli.config_validation.saved_schemas.training_recipe.md#sparseypy.cli.config_validation.saved_schemas.training_recipe.sparsey.SparseyTrainingRecipeSchema.transform_schema)
      * [`SparseyTrainingRecipeSchema.validate_metrics_in_order()`](sparseypy.cli.config_validation.saved_schemas.training_recipe.md#sparseypy.cli.config_validation.saved_schemas.training_recipe.sparsey.SparseyTrainingRecipeSchema.validate_metrics_in_order)
  * [Module contents](sparseypy.cli.config_validation.saved_schemas.training_recipe.md#module-sparseypy.cli.config_validation.saved_schemas.training_recipe)
* [sparseypy.cli.config_validation.saved_schemas.transform package](sparseypy.cli.config_validation.saved_schemas.transform.md)
  * [Submodules](sparseypy.cli.config_validation.saved_schemas.transform.md#submodules)
  * [sparseypy.cli.config_validation.saved_schemas.transform.binarize module](sparseypy.cli.config_validation.saved_schemas.transform.md#module-sparseypy.cli.config_validation.saved_schemas.transform.binarize)
    * [`BinarizeTransformSchema`](sparseypy.cli.config_validation.saved_schemas.transform.md#sparseypy.cli.config_validation.saved_schemas.transform.binarize.BinarizeTransformSchema)
      * [`BinarizeTransformSchema.build_schema()`](sparseypy.cli.config_validation.saved_schemas.transform.md#sparseypy.cli.config_validation.saved_schemas.transform.binarize.BinarizeTransformSchema.build_schema)
      * [`BinarizeTransformSchema.extract_schema_params()`](sparseypy.cli.config_validation.saved_schemas.transform.md#sparseypy.cli.config_validation.saved_schemas.transform.binarize.BinarizeTransformSchema.extract_schema_params)
      * [`BinarizeTransformSchema.transform_schema()`](sparseypy.cli.config_validation.saved_schemas.transform.md#sparseypy.cli.config_validation.saved_schemas.transform.binarize.BinarizeTransformSchema.transform_schema)
  * [sparseypy.cli.config_validation.saved_schemas.transform.skeletonize module](sparseypy.cli.config_validation.saved_schemas.transform.md#module-sparseypy.cli.config_validation.saved_schemas.transform.skeletonize)
    * [`SkeletonizeTransformSchema`](sparseypy.cli.config_validation.saved_schemas.transform.md#sparseypy.cli.config_validation.saved_schemas.transform.skeletonize.SkeletonizeTransformSchema)
      * [`SkeletonizeTransformSchema.build_schema()`](sparseypy.cli.config_validation.saved_schemas.transform.md#sparseypy.cli.config_validation.saved_schemas.transform.skeletonize.SkeletonizeTransformSchema.build_schema)
      * [`SkeletonizeTransformSchema.extract_schema_params()`](sparseypy.cli.config_validation.saved_schemas.transform.md#sparseypy.cli.config_validation.saved_schemas.transform.skeletonize.SkeletonizeTransformSchema.extract_schema_params)
      * [`SkeletonizeTransformSchema.transform_schema()`](sparseypy.cli.config_validation.saved_schemas.transform.md#sparseypy.cli.config_validation.saved_schemas.transform.skeletonize.SkeletonizeTransformSchema.transform_schema)
  * [Module contents](sparseypy.cli.config_validation.saved_schemas.transform.md#module-sparseypy.cli.config_validation.saved_schemas.transform)

## Submodules

## sparseypy.cli.config_validation.saved_schemas.abs_schema module

Abs Schema: file containing the base class for all Schemas.

### *class* sparseypy.cli.config_validation.saved_schemas.abs_schema.AbstractSchema

Bases: `object`

AbstractSchema: a base class for schemas.
: All schemas are used to vwalidate different config files
  passed in by the user to define model structures, training
  recipes, HPO runs, and create plots.

#### build_precheck_schema()

Builds the precheck schema for the config information
passed in by the user. This is used to verify that all parameters
can be collected in order to build the actual schema that will
be used to verify the entire configuration passed in by the
user.

* **Returns:**
   *(Schema)* -- the precheck schema.

#### *abstract* build_schema(schema_params: dict)

Builds a schema that can be used to validate the passed in
config info.

* **Parameters:**
  **schema_params** -- a dict containing all the required
  parameters to build the schema.
* **Returns:**
  a Schema that can be used to validate the config info.

#### extract_schema_params(config_info: dict)

Extracts the required schema parameters from the config info dict
in order to build the schema to validate against.

* **Parameters:**
  **config_info** -- a dict containing the config info from the
  user.
* **Returns:**
   *(dict)* --

  all the required parameters
  : to build the schema.

#### transform_schema(config_info: dict)

Transforms the config info passed in by the user to
construct the config information required by the model builder.

* **Parameters:**
  **config_info** -- dict containing the config information
* **Returns:**
   *(dict)* -- the transformed config info

#### validate(config_info: dict)

Validates a given configuration against the
schema defined by the class.

* **Parameters:**
  * **config_info** -- a dict containing all of the configuration
    information passed in by the user.
  * **schema** -- a Schema to be used for validation
* **Returns:**
  a dict (might be None) holding the validated
  : (and possibly transformed) user config info.

## sparseypy.cli.config_validation.saved_schemas.schema_utils module

Schema Utils: utility and helper functions for constructing schemas.

### sparseypy.cli.config_validation.saved_schemas.schema_utils.all_elements_are_same_type(x: list)

Returns whether all elements in a list are the same type or not.

* **Parameters:**
  **x** (*list*) -- the list to be checked.
* **Returns:**
   *(bool)* -- whether all elements are the same type or not.

### sparseypy.cli.config_validation.saved_schemas.schema_utils.all_elements_satisfy(x: list, cond: Callable)

Returns whether all elements in a list satisfy a condition or not.

* **Parameters:**
  **x** (*list*) -- the list to check.
* **Returns:**
   *(bool)* -- whether all elements satisfy the condtion or not.

### sparseypy.cli.config_validation.saved_schemas.schema_utils.is_between(x: int | float, range_start: int | float, range_end: int | float)

Returns whether a number is within a given range or not.

* **Parameters:**
  * **x** -- a float or int representing a number.
  * **range_start** -- a float or int representing the start of the range
    (inclusive).
  * **range_end** -- a float or int representing the end of the range
    (inclusive).
* **Returns:**
  a bool indicating whether x is in the given range or not.

### sparseypy.cli.config_validation.saved_schemas.schema_utils.is_expected_len(x: list, expected_len: int)

Returns whether a list is of the expected length or not.

* **Parameters:**
  * **x** -- a list.
  * **expected_len** -- an int representing the expected length
    of the list.
* **Returns:**
  a bool indicating whether x is the expected length
  : or not.

### sparseypy.cli.config_validation.saved_schemas.schema_utils.is_nonnegative(x: int | float)

Returns whether a number is nonnegative or not.

* **Parameters:**
  **x** -- a float or int representing a number.
* **Returns:**
  a bool indicating whether x is nonnegative or not.

### sparseypy.cli.config_validation.saved_schemas.schema_utils.is_positive(x: int | float)

Returns whether a number is positive or not.

* **Parameters:**
  **x** -- a float or int representing a number.
* **Returns:**
  a bool indicating whether x is positive or not.

## Module contents

Init: initialization for the Saved Schemas sub-package.
