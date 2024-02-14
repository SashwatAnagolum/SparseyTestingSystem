import numpy as np

class HPOObjective:
    def __init__(self, hpo_config: dict):
        self.hpo_config = hpo_config

    # using nan_to_num() carries significant performance penalties so we should redo this
    def average_nested_data(self, data):
        if isinstance(data, list):
            return np.mean(np.nan_to_num([self.average_nested_data(item) for item in data]))
        elif hasattr(data, 'tolist'):  # numpy array
            return np.mean(np.nan_to_num(data))
        else:
            # Scalar value
            return data


    def _convert_name(self, metric_name):
        return ''.join(
                [l.capitalize() for l in metric_name.split('_')] + ['Metric']
                )


    def combine_metrics(self, results: list) -> float:
        """
        Combines multiple metric results into a single scalar value using a specified operation and weights,
        averaging values at different levels within each metric. Only metrics specified in the HPO configuration are used.

        :param results: A list of dictionaries containing metric results.
        :return: A single scalar value representing the combined result.
        """
        operation = self.hpo_config['optimization_objective']['combination_method']
        objective_terms = self.hpo_config['optimization_objective']['objective_terms']

        # set up the dictionary
        obj_vals = {
            'total': 0.0,
            'combination_method': self.hpo_config['optimization_objective']['combination_method'],
            'terms': {}
        }

        # for each metric in the objective
        for term in objective_terms:
            # get the correct format of the name
            metric_name = self._convert_name(term["metric"]["name"])
            # for each result in the results get the averaged value of that metric into a list
            term_values = [self.average_nested_data(result[metric_name]) for result in results]
            
            # average the values across all the results to get the subtotal; also record the weight
            obj_vals["terms"][metric_name] = {'value': np.mean(term_values), 
                                                  'weight': term["weight"]}

        # weight all the values
        weighted_objectives = [term["value"] * term["weight"] for k, term in obj_vals["terms"].items()]
        # then perform the selected operation to combine the weighted values
        if operation == "sum":
            obj_vals["total"] = sum(weighted_objectives)
        elif operation == "mean":
            obj_vals["total"] = np.mean(weighted_objectives)
        elif operation == "product":
            obj_vals["total"] = np.prod(weighted_objectives)

        # and return the results
        return obj_vals

        # # Calculate weighted averages for each metric specified in HPO config
        # metric_averages = []
        # for term in objective_terms:
        #     term_name = term['metric']['name']
        #     weight = term['weight']

        #     term_averages = []
        #     for metric in metric_data:
        #         term_name = ''.join(
        #         [l.capitalize() for l in term_name.split('_')] + ['Metric']
        #         )
        #         if term_name in metric:
        #             # Calculate average for this metric
        #             term_avg = self.average_nested_data(metric[term_name])
        #             term_averages.append(term_avg)
        #     # Apply weight and add to the list of metric averages
        #     if term_averages:
        #         weighted_average = np.mean(term_averages) * weight
        #         metric_averages.append(weighted_average)

        # # Perform the specified operation on the metric averages
        # if operation == 'sum':
        #     return sum(metric_averages)
        # elif operation == 'mean':
        #     return sum(metric_averages) / len(metric_averages) if metric_averages else 0
        # elif operation == 'product':
        #     result = 1
        #     for average in metric_averages:
        #         result *= average
        #     return result
        # else:
        #     raise ValueError("Invalid operation. Choose 'sum', 'mean', or 'product'.")
# Test data in lowercase with underscores
test_metric_data = [
    {
        'BasisSetSizeMetric': [[1,2,3,4], [5,6,7,8]],
        'ExactMatchAccuracyMetric': [[1,1,1,1], [1,1,1,1]],
        'FeatureCoverageMetric': [[1.0, 0.1]],
        'ApproximateMatchAccuracyMetric': 1,
        'BasisSetSizeIncreaseMetric': [np.array([1,1,1,1]), np.array([1,1,1,1])]
    },
    {
        'BasisSetSizeMetric': [[1,1,1,1], [1,1,1,1]],
        'ExactMatchAccuracyMetric': [[1,1,1,1], [1,1,1,1]],
        'FeatureCoverageMetric': [1.0, 1.0],
        'ApproximateMatchAccuracyMetric': 1,
        'BasisSetSizeIncreaseMetric': [np.array([1,1,1,1]), np.array([1,1,1,1])]
    },
    # Add more entries as needed
]

# HPO configuration with lowercase metric names
hpo_config = {
    "optimization_objective": {
        "combination_method": "mean",
        "objective_terms": [
            {'metric': {"name": "basis_set_size"}, "weight": 1.0},
            {'metric': {"name": "exact_match_accuracy"}, "weight": 1},
            {'metric': {"name": "feature_coverage"}, "weight": 1.0},
            {'metric': {"name": "approximate_match_accuracy"}, "weight": 1.0},
            {'metric': {"name": "basis_set_size_increase"}, "weight": 1}
        ]
    }
}

# Initialize HPOObjective with the configuration
hpo_objective = HPOObjective(hpo_config)

# Calculate combined metric
combined_metric = hpo_objective.combine_metrics(test_metric_data)
print("Combined Metric:", combined_metric)