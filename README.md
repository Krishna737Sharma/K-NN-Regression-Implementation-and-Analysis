# K-NN Regression Implementation and Analysis

## Description
This repository contains the implementation of K-Nearest Neighbors (K-NN) regression from scratch and using Scikit-learn. The project includes hyperparameter tuning, comparison of methods, and visualizations for performance analysis.

## Tasks
1. **K-NN Implementation (from scratch):**
   - Average of k-nearest neighbors.
   - Weighted average with weights as the inverse of distances.
2. **Hyperparameter Tuning:**
   - Comparison of MSE for k values {3, 7, 11} and distance metrics {p=1, p=2, p=5}.
3. **Scikit-learn Implementation:**
   - K-NN regression using the `brute` algorithm.
4. **Verification:**
   - Comparing the best hyperparameters and performance from scratch with Scikit-learn.

## Files
- `regression_dataset.csv`: Dataset used for the regression tasks.
- `knn_regression.py`: Python script containing the implementation.
- `README.md`: Project documentation in Markdown format.

## How to Run
1. Clone the repository.
2. Install required packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
3. Execute the `knn_regression.py` file.

## Results
The best k and distance metric from the scratch implementation match with Scikit-learn's implementation. Detailed performance metrics and plots are generated as part of the analysis.

## License
This project is licensed under the MIT License.
