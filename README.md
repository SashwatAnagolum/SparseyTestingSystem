![The project logo: a hierarchical machine learning model next to the word "sparesypy." Subtitle: "The Sparsey Testing System."](/readme_images/repository_logo_no_sponsor.png)
# sparseypy: The Sparsey Testing System
A full-featured framework for experimenting with the Sparsey model, including:
* A Python reimplementation of the original model
* Extensive hyperparameter optimization capabilities
* Highly customizable metrics, datasets, and transforms
* Freely extensible with custom classes
* Automatic experiment logging to Google Firestore
* Rich data visualization using Weights & Biases
* Built on PyTorch with GPU acceleration

# Sparsey

Sparsey is a biologically plausible machine learning model developed by Dr. Rod Rinkus and based on the structure of the neocortex. 

It represents information in a fundamentally different way than mainstream models using an extremely efficient, non-optimization-based learning method which can be viewed as a form of adaptive, locality-preserving hashing. 

In conjunction with the multi-level, hierarchical architecture of the memory, this algorithm results in a model that possesses lifelong learning capabilities.

# The Testing System

sparseypy makes getting started with the capabilities of Sparsey easy and accessible.

For our example, we will perform HPO to investigate the effects of different model parameters on the rate of basis set size growth, a key aspect of Sparsey's lifelong learning capabilities.

First, define your model and experiment parameters using the system's automatically validated YAML configuration files.

```yaml
hyperparameters:
  # the layers section defines the properties of individual layers in the Sparey model
  layers:
    - name: 
        value: sparsey
      params:
        # num_cms_per_mac: int > 0
        #     the number of coding modules that comprise each MAC in this layer
        num_cms_per_mac:
          value: 2
        # num_macs: int > 0
        #     the number of MACs in the layer
        #     if this is smaller than the layer size, not all rows will be filled
        num_macs:
          values: [4, 9]
        # num_neurons_per_cm: int > 0
        #     the number of neurons in each competitive module in each MAC in the layer
        num_neurons_per_cm:
          min: 2
          max: 10
          distribution: int_uniform
        # mac_receptive_field_size: float > 0
        #     the receptive field radius of each MAC, defined in terms of the side length of the current layer
        mac_receptive_field_size:
          values: [0.75, 1.0, 1.5]
      ...
```

Next, perform your experiments in the system CLI, with feedback on core performance indicators.

![The sparseypy CLI during a hyperparameter optimization run.](/readme_images/cli2.png)

Finally, review the full detail of your results on Weights & Biases.

![The results of a basis set size experiment in Weights & Biases.](/readme_images/wandb.png)

The experiment results show a promising trend--for most models, the basis set size does decrease over time. This particular experiment suggests that the size of the basis set (the number of unique codes learned by the model over time) is negatively correlated with the receptive field radius (the portion of the previous layer an individual MAC sees as part of its input).

# Installation

Requires **Python 3.11** and a **Weights & Biases** account.

User:  
`pip install sparseypy`

Developer:  
`git clone https://github.com/Neurithmic-Systems/SparseyTestingSystem.git`  
`./environment_setup`  
(installs sparseypy in a virtual environment in the project directory)

# Configuration

Create a Firebase account service key file following the project instructions.

Create a `.env` file in your work folder.
```
WANDB_API_KEY=<your Weights & Biases API key>
FIREBASE_ACCOUNT_SERVICE_FILE=<full path to the key file>
```

# Running The System

The Sparsey Testing System is accessed via a command-line interface. This command-line interface consists of **three scripts**, each corresponding to a different area of system functionality.
* `train_model` for training and performing evaluation on a single Sparsey model
* `evaluate_model` for performing additional evaluation on an existing trained Sparsey model
* `run_hpo` for performing hyperparameter optimization on a family of Sparsey models

Each script has its own command-line arguments (detailed below) and also makes use of a selection of the **six main configuration files**:
* `dataset.yaml`: Defines a dataset for use with the system. Includes the type and location of the dataset, optional preprocessing transforms, and performance options like lazy loading and in-memory caching.
  * Required for **all commands**
* `hpo.yaml`: Defines a set of candidate hyperparameters and metrics for use in HPO. Controls all the hyperparameters for the run, all metrics and training parameters, and the calculation of the objective function used to rank the success of experiments within an HPO run.
  * Required for `run_hpo`
* `network.yaml`: Defines the structure of a single Sparsey model, including all the model-related hyperparameters, the layer structure, the input size, and the model name and description.
  * Required for `train_model` (**unless** using an existing named model)
* `preprocessing.yaml`: Defines the sequence of transformations to be applied to input data loaded from the datasets.
  * Required for **all commands**
* `system.yaml`: Defines system-level settings for Weights & Biases and Firestore, such as the resolution of the data to be saved to the database.
  * Required for **all commands**
* `trainer.yaml`: Defines the training parameters and metrics to use for an individual training experiment.
  * Required for `train_model` and `evaluate_model`

The system also ships with **fully-commented reference configuration files**, including an explanation of the accepted values and purpose of every parameter, which can be found in the `demo` folder.


# Command-Line Reference

`train_model` - Train and evaluate a new (or existing) model in a single experiment.

**Required arguments:**  
`--training_dataset_config <path to training dataset.yaml>`  
The dataset to use for training.  
`--evaluation_dataset_config <path to evaluation dataset.yaml>`  
The dataset to use for evaluation.  
`--preprocessing_config <path to preprocessing.yaml>`  
The preprocessing stack (series of transformations) to use for this training run.  
`--system_config <path to system.yaml>`  
The system configuration (database and Weights & Biases settings) to use for the training run.  
`--training_recipe_config <path to trainer.yaml>`  

You must also select **exactly one** of the following two options to specify the model to use:  
`--model_config <path to network.yaml>`  
Train a new model with the configuration provided in the indicated file.  
**OR**  
`--model_name "example_model"`  
Reload the previously-trained model with name “example_model” from Weights & Biases for additional training.  

***

`evaluate_model` - Reload a chosen model from Weights & Biases and evaluate its performance on the provided dataset.  

**Required arguments:**  
`--dataset_config <path to dataset.yaml>`  
The dataset on which to evaluate the model.  
`--preprocessing_config <path to preprocessing.yaml>`  
The preprocessing transforms to apply to data loaded from the datasets.  
`--system_config <path to system.yaml>`  
The system configuration (database and Weights & Biases settings) to use for the evaluation run.  
`--training_recipe_config <path to trainer.yaml>`  
The training-related parameters (such as batch size) to use for the evaluation run.  
`--model_name "example_model"`  
The name of the model from the Weights & Biases model registry to use for this evaluation.  

***

`run_hpo` - Perform a hyperparameter optimization run to optimize a Sparsey model for a particular task.

**Required arguments:**  
`--training_dataset_config <path to dataset.yaml>`  
The dataset to use for training the models in this HPO run.  
`--evaluation_dataset_config <path to eval_dataset.yaml>`  
The dataset to use for evaluating the models in this HPO run.  
`--preprocessing_config <path to preprocessing.yaml>`  
The preprocessing transforms to apply to data loaded from the datasets.  
`--hpo_config <path to hpo.yaml>`  
The hyperparameter options, strategy, and metrics to evaluate in this HPO run.  
`--system_config <path to system.yaml>`  
The system configuration (database and Weights & Biases settings) to use for the HPO run.  

# Further Documentation

The project's API documentation and full manuals are available on this repository's GitHub Pages and Wiki, respectively.

# Contributing

Neurithmic Systems welcomes contributions from any interested developers--please feel free to get in touch or make a pull request.

# Acknowledgements

Sparsey, its existing Java implementation, and all associated research is the brainchild of **Dr. Rod Rinkus**.

The Sparsey Testing System was developed by **Sashwat Anagolum**, **JD Padrnos**, **Andy Klawa**, and **CJ Pereira** from The Pennsylvania State University as part of their capstone project.

The authors also wish to thank **Codie Petersen** for his extensive contributions to the project.

Sparsey (c) 2023 Neurithmic Systems 
