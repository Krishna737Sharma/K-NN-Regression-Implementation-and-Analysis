<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>K-NN Regression Implementation and Analysis</title>
</head>
<body>
    <h1>K-NN Regression Implementation and Analysis</h1>
    
    <h2>Description</h2>
    <p>This repository contains the implementation of K-Nearest Neighbors (K-NN) regression from scratch and using Scikit-learn. The project includes hyperparameter tuning, comparison of methods, and visualizations for performance analysis.</p>
    
    <h2>Tasks</h2>
    <ol>
        <li>
            <strong>K-NN Implementation (from scratch):</strong>
            <ul>
                <li>Average of k-nearest neighbors.</li>
                <li>Weighted average with weights as the inverse of distances.</li>
            </ul>
        </li>
        <li>
            <strong>Hyperparameter Tuning:</strong>
            <ul>
                <li>Comparison of MSE for k values {3, 7, 11} and distance metrics {p=1, p=2, p=5}.</li>
            </ul>
        </li>
        <li>
            <strong>Scikit-learn Implementation:</strong>
            <ul>
                <li>K-NN regression using the <code>brute</code> algorithm.</li>
            </ul>
        </li>
        <li>
            <strong>Verification:</strong>
            <ul>
                <li>Comparing the best hyperparameters and performance from scratch with Scikit-learn.</li>
            </ul>
        </li>
    </ol>

    <h2>Files</h2>
    <ul>
        <li><code>regression_dataset.csv</code>: Dataset used for the regression tasks.</li>
        <li><code>knn_regression.py</code>: Python script containing the implementation.</li>
        <li><code>README.html</code>: Project documentation in HTML format.</li>
    </ul>

    <h2>How to Run</h2>
    <ol>
        <li>Clone the repository.</li>
        <li>Install required packages: <code>numpy</code>, <code>pandas</code>, <code>matplotlib</code>, <code>scikit-learn</code>.</li>
        <li>Execute the <code>knn_regression.py</code> file.</li>
    </ol>

    <h2>Results</h2>
    <p>The best k and distance metric from the scratch implementation match with Scikit-learn's implementation. Detailed performance metrics and plots are generated as part of the analysis.</p>

    <h2>License</h2>
    <p>This project is licensed under the MIT License.</p>
</body>
</html>
