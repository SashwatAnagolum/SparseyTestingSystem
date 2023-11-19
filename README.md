# SparseyTestingSystem
A Python wrapper for the Sparsey Java implementation, with additional features such as result visualization, HPO, and more.


# Development notes
 

### CLI config validation
We have some in-built schemas for models, trainers, hpo runs.
	Models:
		Sparsey
		ResNeT
		ViT
		ConvNeXT

	Trainers:
		ASParsey trianer
		Any deep learning model trainer (most options are the same)

	HPO runs:
		Sparsey HPO runs (we need to cover all hyperparams)
		ResNet HPO runs (most options are the same)
		ViT HPO runs (most options are the same)
		ConvNeXT HPO runs (most options are the same)

Use a config factory to create config schemas to valid the configs input by the user against.
For custom schemas -> users need to use --custom_schema <FILEPATH_TO_PICKLED_CUSTOM_SCHEMA> option in all the scripts or the -disable_config_validation flag. 