# LendingClub Loan Data Analysis and Modeling

## Goal

This project aims to analyze and model LendingClub loan data to predict loan default risks. We employ various data processing, exploration, and machine learning techniques to derive insights from the data and build predictive models.

## Overview

The project follows a structured approach, starting with data processing and exploration, followed by feature engineering, model training, and evaluation. It leverages a range of Python libraries and tools to analyze the LendingClub dataset, visualize patterns, and train machine learning models.

## Features

- **Data Processing**: Initial processing of raw data to prepare it for further analysis and modeling.
- **Data Exploration**: Comprehensive exploration and visualization of the dataset to identify patterns, relationships, and insights.
- **Data Preprocessing**: Advanced preprocessing techniques including handling of missing values and feature scaling.
- **Model Training**: Implementation and training of machine learning models, including Neural Networks and XGBoost, to predict loan default risks.
- **Model Evaluation**: Evaluation of model performance using various metrics and visualization of results.
- **Custom Plot Settings**: Custom configurations for consistent and visually appealing plots.

## Methods and Techniques

### 1. XGBoost Classifier
   - The XGBoost model was trained using the best parameters found through RandomizedSearchCV.
   - The model was evaluated using accuracy, classification report, confusion matrix, and ROC-AUC score on both the training and test datasets.
   - The ROC curve for the XGBoost Classifier was also plotted.

### 2. Random Forest Classifier
   - A Random Forest model was trained using the best parameters found through RandomizedSearchCV.
   - The model was evaluated using the same metrics as the XGBoost model.
   - The ROC curve for the Random Forest Classifier was also plotted.

### 3. Neural Network Classifier
   - A custom Neural Network was defined and trained using Keras.
   - The learning evolution of the model was visualized, showing the changes in loss and AUC score during training.
   - The Neural Network model was evaluated using the same metrics as the previous models.

A comparison of the three models was conducted based on their ROC-AUC scores on the train and test datasets, and the results were visualized in a bar plot.

## Project Organization
```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
├── notebooks  
│   ├── 003_data_exploration.ipynb    <- Jupyter notebook for interactive data exploration and analysis     
├── requirements.txt   <- The requirements file for reproducing the analysis environment.
├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported.
├── src                <- Source code for use in this project.
│   ├── utility   
│   │   ├── plot_settings.py              <- Script containing custom plot settings and configurations.
│   ├── data         
│   │   ├── 001_process_data.py           <- Script for initial data processing.
│   ├── features       
│   │   ├── 004_data_preprocessing.py     <- Script for advanced data preprocessing.
│   ├── models         
│   │   ├── 005_train_model.py            <- Script for training and evaluating machine learning models.
│   └── visualization 
│       ├── 002_data_exploration.py       <- Script for data exploration and visualization.
├── environment.yml    <- Conda environment file.
└── .gitignore         <- Specifies intentionally untracked files to ignore when using Git.
```

## Prerequisites

- Python 3.x
- NumPy
- pandas
- Matplotlib
- seaborn
- scikit-learn
- TensorFlow
- XGBoost

## Setup and Installation

### Using `requirements.txt`

1. Clone the repository:
```sh
git clone https://github.com/omid-sar/LendingClub.com-Omid
cd LendingClub.com-Omid

```

2. Create a new conda environment from the `environment.yml` file:
```sh
conda env create -f environment.yml
```

3. Activate the newly created conda environment:
```sh
conda activate LendingClub.com
```


## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt) file for details.

