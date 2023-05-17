Python implementation of algorithms from the paper **Optimization of Inter-group Criteria for Clustering with
Minimum Size Constraints**:

* `FeasibleSpacing`: finds clusterings with maximum minimum spacing between clusters, provided all clusters have at least `min_size_factor * min_size` elements.
* `ConstrainedMaxMST`: finds clusterings with maximum minimum spanning tree cost, provided all clusters have at least `min_size_factor * min_size` elements.

The results of the experiments in the paper can be found in `full_results.csv`, and can be reproduced by running the following command:

```
python generate_models.py --results_path /path/to/results --sl_filepath /path/to/results/results_sl.joblib --km_filepath /path/to/results/results_km.joblib -sm
````

Check `python generate_models.py --help` for details on how to run the experiments.