# -*- coding: utf-8 -*-

"""
Validation Constants: constants used during the process
    of config file validation.
"""

from collections import defaultdict


allowed_schema_types = set(
    [
        'model',
        'trainer',
        'hpo_run',
        'plot'
    ]
)

allowed_schema_names = defaultdict(set)

allowed_schema_names['model'] = set(
    [
        'sparsey'
    ]
)