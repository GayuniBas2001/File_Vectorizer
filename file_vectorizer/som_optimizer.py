# import numpy as np
# from sklearn.datasets import fetch_california_housing
# from sompy.sompy import SOMFactory

# def load_data():
#     """Loads California housing data and combines features with target."""
#     data_bunch = fetch_california_housing()
#     feature_names = data_bunch.feature_names + ["HouseValue"]
#     data = np.column_stack((data_bunch.data, data_bunch.target))
#     return data, feature_names

# def reduce_dimensionality(data, feature_names, map_size=(10, 10)):
#     """Applies SOM for dimensionality reduction using sompy."""
#     som_factory = SOMFactory()
#     som = som_factory.build(data, normalization='var',
#                             initialization='random',
#                             component_names=feature_names,
#                             mapsize=map_size)

#     som.train(n_job=1, verbose='info')  # Training is important to get BMUs

#     topographic_error = som.calculate_topographic_error()
#     quantization_error = np.mean(som._bmu[1])

#     print(f"Topographic error = {topographic_error}; Quantization error = {quantization_error}")

#     # Get reduced dimensions: BMU indices (2D coordinates)
#     bmu_locations = som._bmu[0]
#     print(f"Original shape: {data.shape}")
#     print(f"Reduced dimensional data shape: {bmu_locations.shape}")  # (n_samples, 2)

#     return bmu_locations

# def main():
#     data, feature_names = load_data()
#     reduced_data = reduce_dimensionality(data, feature_names)

# if __name__ == "__main__":
#     main()
