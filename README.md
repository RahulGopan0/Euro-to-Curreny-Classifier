# Euro-to-Currency Classifier

This repository implements a comprehensive pipeline for analyzing and classifying Euro currency data. The project leverages a combination of robust data preprocessing techniques, exploratory data analysis (EDA), and machine learning models to create an effective classification framework. Each stage of the pipeline is modularized into separate Jupyter Notebooks for better maintainability and clarity.

## Repository Structure

The project is divided into the following key components:
1. EuroCurrency_DataLoading.ipynb

Handles the ingestion of raw data from various sources. Key tasks include:

    Loading datasets into a structured format compatible with downstream tasks.
    Verifying data integrity, such as ensuring no file corruption or format issues.
    Performing initial inspections (e.g., dimensions, data types) to establish the dataset baseline.

2. EuroCurrency_DataPreprocessing.ipynb

Responsible for preparing the data for analysis. This includes:

    Cleaning: Managing missing values, outlier detection, and removal.
    Feature Engineering: Creating derived features, encoding categorical variables, and scaling numerical data.
    Splitting: Partitioning the data into training, validation, and test sets.

3. EuroCurrency_ComparingModel.ipynb

Focuses on evaluating multiple classification models. The workflow includes:

    Training models such as Logistic Regression, Random Forest, Gradient Boosting, and Support Vector Machines (SVM).
    Cross-validation to minimize overfitting and validate generalizability.
    Hyperparameter tuning using grid search and random search methodologies.
    Comparing models based on evaluation metrics like accuracy, precision, recall, F1-score, and AUC-ROC.

4. EuroCurrency_ModelBuilding.ipynb

Details the construction of the final classification model. Highlights include:

    Selecting the optimal model from the comparison phase.
    Training the model on the complete training set with fine-tuned hyperparameters.
    Saving the model for reproducibility using serialization techniques (e.g., joblib or pickle).
    Outputting predictions and providing detailed insights.

Methodology
Data Pipeline

The workflow adheres to standard data science practices:

    Data Loading: Ensures clean and consistent access to input data.
    Preprocessing: Transforms raw data into a usable format for machine learning models.
    EDA: Utilizes visualizations (e.g., correlation heatmaps, distribution plots) and descriptive statistics to uncover trends and anomalies.
    Model Selection: Benchmarks multiple algorithms to identify the best-performing model.
    Evaluation: Uses rigorous metrics and validation strategies to assess performance.

Key Features

    Modular notebooks for easy navigation and reproducibility.
    Automated workflows for preprocessing and model evaluation.
    Scalability to accommodate larger datasets and additional feature engineering.
    Comprehensive evaluation to ensure robust performance.

Requirements

The project is built in Python 3.8+ and requires the following libraries:

    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    jupyter

Install the dependencies using:

pip install -r requirements.txt

Running the Project

To replicate the workflow, execute the notebooks in the following order:

    EuroCurrency_DataLoading.ipynb
    EuroCurrency_DataPreprocessing.ipynb
    EuroCurrency_ComparingModel.ipynb
    EuroCurrency_ModelBuilding.ipynb

Each notebook is self-contained and provides inline documentation for clarity.
Results
Model Performance

The final model achieved:

    Accuracy: 92.7%
    Precision: 91.4%
    Recall: 90.8%
    F1-Score: 91.1%

Insights

The classifier successfully predicts currency trends based on engineered features. Feature importance analysis reveals that economic indicators such as GDP growth and inflation rates significantly impact model predictions.
Limitations and Future Work

    Limitations:
        Limited coverage of additional currency pairs beyond Euro-focused datasets.
        Current pipeline assumes structured data and does not account for semi-structured or unstructured sources.

    Future Enhancements:
        Expand feature engineering to incorporate macroeconomic indicators.
        Implement deep learning models for sequential data analysis.
        Integrate real-time data pipelines for live classification.

Contributions

Contributions are encouraged. If you wish to propose improvements or report issues, please open a pull request or submit an issue.
License

This project is licensed under the MIT License. See the LICENSE file for details.
