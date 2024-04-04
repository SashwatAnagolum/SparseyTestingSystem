# sparseypy.core.transforms package

## Submodules

## sparseypy.core.transforms.abstract_transform module

### *class* sparseypy.core.transforms.abstract_transform.AbstractTransform

Bases: `Module`

An abstract class representing a transformation. All transformations
should inherit from this class and implement the forward method.

#### forward(sample)

Defines the computation performed at every call. Should be overridden by all subclasses.

* **Parameters:**
  **sample** -- The input data to be transformed.
* **Returns:**
  Transformed data.

## sparseypy.core.transforms.binarize_transform module

### *class* sparseypy.core.transforms.binarize_transform.BinarizeTransform(binarize_threshold=0.5)

Bases: [`AbstractTransform`](#sparseypy.core.transforms.abstract_transform.AbstractTransform)

A transform to first convert an image to grayscale and then
binarize it based on a threshold.

#### forward(sample)

Apply grayscale conversion and binarization to the input sample.

* **Parameters:**
  **sample** -- The input data to be transformed, assumed to be a PyTorch tensor.
* **Returns:**
  Binarized grayscale data.

## sparseypy.core.transforms.skeletonize_transform module

### *class* sparseypy.core.transforms.skeletonize_transform.SkeletonizeTransform(sigma=3.0)

Bases: [`AbstractTransform`](#sparseypy.core.transforms.abstract_transform.AbstractTransform)

A transform to apply Canny edge detection followed by skeletonization.

#### forward(sample)

Apply Canny edge detection and skeletonization to the input sample.

* **Parameters:**
  **sample** -- The input data to be transformed, assumed to be a PyTorch tensor.
* **Returns:**
  Skeletonized edge data.

## sparseypy.core.transforms.transform_factory module

Transform Factory: class to build transforms.

### *class* sparseypy.core.transforms.transform_factory.TransformFactory

Bases: `object`

TransformFactory: factory class for constructing built-in system and
PyTorch transforms.

#### allowed_modules

the names of the transform classes shipped as

* **Type:**
  list[string]

### part of the system that are allowed to be constructed with get_transform_class().

#### allowed_modules *= {'BinarizeTransform', 'SkeletonizeTransform', 'abstract_transform', 'binarize_transform', 'skeletonize_transform'}*

#### *static* create_transform(transform_name, \*\*kwargs)

Creates a transform passed in based on the transform name and kwargs.

* **Parameters:**
  * **transform_name** (*str*) -- the name of the transform to create.
  * **\*\*kwargs** -- arbitrary keyword arguments, passed to the transform
  * **class constructor.**

#### *static* get_transform_class(class_name: str)

Gets the class corresponding to the name passed in.
Throws an error if the name is not valid.

* **Parameters:**
  **class_name** (*str*) -- the transform class name to create

#### *static* get_transform_name(transform_name: str)

Converts a transform name from config file format ("transform_name") to
transform class name format ("TransformName").

* **Parameters:**
  **transform_name** (*str*) -- the config-style transform name
* **Returns:**
  *str* -- the class-style transform name

## Module contents
