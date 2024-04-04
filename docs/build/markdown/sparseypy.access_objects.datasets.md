# sparseypy.access_objects.datasets package

## Submodules

## sparseypy.access_objects.datasets.built_in_dataset module

Built In Dataset: file holding the built in dataset class.

### *class* sparseypy.access_objects.datasets.built_in_dataset.BuiltInDataset(name: str, root: str, download: bool, transform: str)

Bases: [`Dataset`](#sparseypy.access_objects.datasets.dataset.Dataset)

## sparseypy.access_objects.datasets.dataset module

Dataset: file holding the dataset class.

### *class* sparseypy.access_objects.datasets.dataset.Dataset

Bases: `Dataset`

## sparseypy.access_objects.datasets.dataset_factory module

Dataset Factory: file holding the Dataset Factory class.

### *class* sparseypy.access_objects.datasets.dataset_factory.DatasetFactory

Bases: `object`

#### allowed_modules *= {'BuiltInDataset', 'ImageDataset', 'PreprocessedDataset', 'SparseyDataset', 'built_in_dataset', 'dataset', 'image_dataset', 'preprocessed_dataset', 'sparsey_dataset'}*

#### *static* create_dataset(dataset_type: str, \*\*kwargs)

Creates a layer passed in based on the layer name and kwargs.

* **Parameters:**
  **dataset_type** (*str*)

#### *static* get_dataset_class(dataset_type: str)

Gets the class corresponding to the name passed in.
Throws an error if the name is not valid.

* **Parameters:**
  **dataset_type** (*str*) -- the type of dataset to create.

## sparseypy.access_objects.datasets.image_dataset module

IMage Dataset: file holding the image dataset class.

### *class* sparseypy.access_objects.datasets.image_dataset.ImageDataset(data_dir: str, image_format: str)

Bases: [`Dataset`](#sparseypy.access_objects.datasets.dataset.Dataset)

## sparseypy.access_objects.datasets.preprocessed_dataset module

Preprocessed Dataset: wrapper for datasets

### *class* sparseypy.access_objects.datasets.preprocessed_dataset.PreprocessedDataset(dataset: [Dataset](#sparseypy.access_objects.datasets.dataset.Dataset), preprocessing_stack: PreprocessingStack, preprocessed_dir: str = 'datasets/preprocessed_dataset')

Bases: [`Dataset`](#sparseypy.access_objects.datasets.dataset.Dataset)

A dataset wrapper class that applies preprocessing to another dataset and caches the results.
.. attribute:: dataset

> The original dataset to be preprocessed.

> * **type:**
>   Dataset

#### preprocessed_dir

Directory where preprocessed data is stored.

* **Type:**
  str

#### preprocessing_stack

The preprocessing operations to be applied.

* **Type:**
  PreprocessingStack

#### preprocessed_flags

A boolean list indicating whether an item has been preprocessed.

* **Type:**
  list[bool]

## sparseypy.access_objects.datasets.sparsey_dataset module

Sparsey Dataset: file holding the Sparsey dataset class to read data from Sparsey binary RAW files.

### *class* sparseypy.access_objects.datasets.sparsey_dataset.SparseyDataset(data_dir: str, width: int, height: int)

Bases: [`Dataset`](#sparseypy.access_objects.datasets.dataset.Dataset)

Dataset adapter class to read existing binary Sparsey datasets.

Requires the width and height of the dataset images be provided
in order to correctly interpret the raw files.

Note this adapter currently does NOT support reading
multiple items from the same binary file.

## Module contents
