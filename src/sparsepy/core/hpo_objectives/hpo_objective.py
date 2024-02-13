import numpy as np

class HPOObjective:
    def __init__(self, hpo_config: dict):
        self.hpo_config = hpo_config

    def combine_metrics(self, metric_data: list) -> float:
        """
        Combines multiple metric results into a single scalar value using a specified operation and weights,
        averaging values at different levels within each metric. Only metrics specified in the HPO configuration are used.

        :param metric_data: A list of dictionaries containing metric results.
        :return: A single scalar value representing the combined result.
        """
        operation = self.hpo_config['optimization_objective']['combination_method']
        objective_terms = self.hpo_config['optimization_objective']['objective_terms']

        def average_nested_data(data):
            if isinstance(data, list):
                return np.mean([average_nested_data(item) for item in data])
            elif hasattr(data, 'tolist'):  # numpy array
                return np.mean(data)
            else:
                # Scalar value
                return data

        # Calculate weighted averages for each metric specified in HPO config
        metric_averages = []
        for term in objective_terms:
            term_name = term['metric']['name']
            weight = term['weight']

            term_averages = []
            for metric in metric_data:
                term_name = ''.join(
                [l.capitalize() for l in term_name.split('_')] + ['Metric']
                )
                if term_name in metric:
                    # Calculate average for this metric
                    term_avg = average_nested_data(metric[term_name])
                    term_averages.append(term_avg)
            # Apply weight and add to the list of metric averages
            if term_averages:
                weighted_average = np.mean(term_averages) * weight
                metric_averages.append(weighted_average)

        # Perform the specified operation on the metric averages
        if operation == 'sum':
            return sum(metric_averages)
        elif operation == 'mean':
            return sum(metric_averages) / len(metric_averages) if metric_averages else 0
        elif operation == 'product':
            result = 1
            for average in metric_averages:
                result *= average
            return result
        else:
            raise ValueError("Invalid operation. Choose 'sum', 'mean', or 'product'.")
# Test data in lowercase with underscores
test_metric_data = [
    {
        'BasisSetSizeMetric': [[1,1,1,1], [1,1,1,1]],
        'ExactMatchAccuracyMetric': [[1,1,1,1], [1,1,1,1]],
        'FeatureCoverageMetric': [[1.0, 1.0]],
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