# -*- coding: utf-8 -*-

"""
Named dataset schema: the schema for named dataset config files.

Does NOT correspond to a NamedDataset class.
"""

import inspect
import typing
import os

from schema import Schema, Optional, And, Use
from torchvision import datasets as torchvision_datasets
from torchvision.transforms import v2

from sparseypy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparseypy.cli.config_validation import schema_factory
from sparseypy.cli.config_validation.saved_schemas import preprocessing_stack


class BuiltInDatasetSchema(AbstractSchema):
    """
    BuiltInDatasetSchema: schema for built-in datasets (created by name, e.g. MNIST).
    """
    def convert_transform_name(self, transform_name: str) -> str:
        """
        Converts the transform name from the format used in
        the dataset config file to the naming format used by PyTorch.

        Args:
            transform_name (str): the name of the transform.

        Returns:
            (str): the converted transform name.
        """
        converted_transform_name = ''.join(
            [
                word[:1].upper() + word[1:]
                for word in transform_name.split('_')
            ]
        )

        return converted_transform_name


    def check_if_transform_exists(self, transform_name) -> bool:
        """
        Checks if a (Torchvision v2) transform with the name transform_name exists.

        Args:
            transform_name (str): the name of the transform to check.

        Returns:
            (bool): whether the transform exists or not
        """
        if transform_name in [
            cls[0] for cls in inspect.getmembers(v2, inspect.isclass)
        ]:
            return True

        return False
        #if not hasattr(v2, transform_name):
        #    return False
        #
        #return True


    def check_if_dataset_exists(self, dataset_name: str) -> bool:
        """
        Checks if a builtin torchvision dataset exists with the 
        name specified.

        Args:
            dataset_name (str): the name of the dataset.

        Returns:
            (bool): whether the dataset exists or not.
        """
        processed_dataset_name = ''.join(
            [i[:1].upper() + i[1:] for i in dataset_name.split('_')]
        )

        return processed_dataset_name in dir(torchvision_datasets)


    def extract_schema_params(
            self, config_info: dict) -> typing.Optional[dict]:
        """
        Extracts the required schema parameters from the config info dict
        in order to build the schema to validate against.

        Args:
            config_info: a dict containing the config info from the 
                user.

        Returns:
            a dict (might be None) containing all the required parameters 
                to build the schema.
        """
        schema_params = dict()

        if config_info.get('preprocessed', False):
            schema_params[
                'preprocessing_stack_schema'
            ] = schema_factory.get_schema_by_name(
                preprocessing_stack, 'preprocessing_stack',
                'default'
            )
        else:
            schema_params[
                'preprocessing_stack_schema'
            ] = Schema(object)

        return schema_params


    def transform_schema(self, config_info: dict) -> dict:
        """
        Transforms the config info passed in by the user to 
        construct the config information required by the model builder.

        Args:
            config_info: dict containing the config information

        Returns:
            (dict): the transformed config info
        """
        return config_info


    def build_schema(self, schema_params: dict) -> Schema:
        """
        Builds a schema that can be used to validate the passed in
        config info.

        Args:
            schema_params: a dict containing all the required
                parameters to build the schema.

        Returns:
            a Schema that can be used to validate the config info.
        """
        config_schema = Schema(
            {
                'dataset_type': Schema(
                    'built_in', error="dataset_type must be 'built_in'"
                ),
                Optional('description', default=None): str,
                'params': Schema(
                    {
                        'name': Schema(
                            And(
                                str, self.check_if_dataset_exists
                            ), error="The dataset name is invalid or does not exist."
                        ),
                        Optional('root', default='./'): Schema(
                            And(
                                str,
                                os.path.exists
                            ), error="Dataset save path does not exist."
                        ),
                        Optional('download', default=True): bool,
                        Optional('transform'): Schema(
                            And(
                                Use(self.convert_transform_name),
                                str, self.check_if_transform_exists,
                            )
                        )
                    },
                    error="Invalid params"
                ),
                Optional('preprocessed', default=False): Schema(
                    bool, error="preprocessed must be a boolean value"
                ),
                Optional(
                    'preprocessed_temp_dir',
                    default='datasets/preprocessed_dataset'
                ): Schema(str, error="preprocessed_temp_dir must be a valid path"),
                Optional(
                    'preprocessed_stack',
                    default={'transform_list': []}
                ): schema_params['preprocessing_stack_schema'],
                Optional('save_to_disk', default=False): Schema(
                    bool, error='save_to_disk must be a boolean value'
                ),
                Optional('in_memory', default=False): Schema(
                    bool, "in_memory must be a boolean value"
                ),
                Optional('load_lazily', default=True): Schema(
                    bool, "load_lazily must be a boolean value"
                )
            }, error="Error in built-in dataset configuration"
        )

        return config_schema
