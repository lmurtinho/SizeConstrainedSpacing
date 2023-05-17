Python implementation of algorithms from the paper **Optimization of Inter-group Criteria for Clustering with
Minimum Size Constraints**:

* `FeasibleSpacing`: finds clusterings with maximum minimum spacing between clusters, provided all clusters have at least `min_size_factor * min_size` elements.
* `ConstrainedMaxMST`: finds clusterings with maximum minimum spanning tree cost, provided all clusters have at least `min_size_factor * min_size` elements.

The results of the experiments in the paper can be found in `full_results.csv`, and can be reproduced by running `generate_models.py`. Please use `python generate_models.py --help` for details on how to run the experiments. It is necessary to fit the k-means models for the datasets to find the minimum sizes of groups for each clustering, and it is recommended to fit a single-link model to pass as argument to the algorithms to reduce execution time.