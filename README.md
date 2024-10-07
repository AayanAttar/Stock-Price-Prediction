# Stock-Price-Prediction
# Stock-Price-Prediction
# Stock Price Prediction Using Random Forest Regression

## Introduction
This project aims to predict the stock price of Amazon using historical price and volume data derived from various financial instruments. The methodology employed includes exploring the relationships between features, applying data preprocessing techniques, and training a Random Forest Regression model to forecast the target stock price. Additionally, the project encompasses Exploratory Data Analysis (EDA) and performance evaluation using multiple metrics, including Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

### Project Objectives
- To analyze the impact of various financial instruments on Amazon's stock price.
- To develop a predictive model using machine learning techniques.
- To evaluate the model's performance and accuracy.

### Dataset Overview
The dataset comprises historical stock prices and volumes from multiple financial instruments, including:
- **Amazon**
- **Apple**
- **Tesla**
- **Bitcoin**
- **Ethereum**
- **Gold**, etc.

The dataset contains **1243 rows** and **39 columns**, with each row representing a single day’s trading data. The columns include:
- **Date**: The date for which the stock data is recorded.
- **Amazon_Price**: The closing price of Amazon's stock for the day (Target Variable).
- **Volume**: The total number of shares or units traded on that specific day.
- **Other Features**: The prices and volumes of the other included financial instruments, which serve as input features for the model.

### Feature Representation
The dataset consists of several features that may influence the price prediction. Some key features include:
- `Amazon_price`, `Amazon_volume`
- `Apple_price`, `Apple_volume`
- `Tesla_price`, `Tesla_volume`
- `Bitcoin_price`, `Bitcoin_volume`
- `Ethereum_price`, `Ethereum_volume`
- `Gold_price`, `Gold_volume`

The objective is to leverage historical price and volume data to forecast future prices of Amazon based on patterns observed in the dataset.

## Table of Contents
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Model Development](#model-development)
- [Performance Metrics](#performance-metrics)
- [How to Run the Project](#how-to-run-the-project)


## Technologies Used
- **Programming Language**: Python
- **Development Environment**: Jupyter Notebook for interactive coding and data analysis

### Libraries
The project leverages the following libraries for various tasks:
- **Pandas**: Used for data manipulation, cleaning, and preprocessing.
- **NumPy**: Provides support for numerical computations and operations on arrays.
- **Matplotlib & Seaborn**: Utilized for data visualization to understand trends and relationships.
- **Scikit-learn**: Employed for model training, evaluation, and cross-validation.
- **Random Forest**: The primary machine learning algorithm used for regression tasks.
- **StandardScaler**: For feature scaling to normalize the dataset.

## Project Structure
The project directory structure is organized as follows:

## Exploratory Data Analysis
The EDA process involves a series of steps aimed at understanding the dataset better:
- **Distribution Visualization**: Plotting the distribution of the target variable (`Amazon_Price`) to understand its spread.
- **Feature Relationships**: Analyzing relationships between `Amazon_Price` and other features using scatter plots and correlation coefficients.
- **Correlation Matrix**: Computing and visualizing the correlation matrix to identify potential predictors for the target variable.
- **Data Quality Check**: Identifying missing values and checking data types for appropriate conversions.

## Data Preprocessing
Key data preprocessing steps include:
1. **Handling Missing Values**: Filling missing values using the mean of respective columns to maintain data integrity.
2. **Type Conversion**: Converting object data types (e.g., price and volume columns) to float for numerical operations.
3. **Feature Engineering**: Extracting useful components from the date column (e.g., year, month, day) to capture temporal trends.
4. **Feature Scaling**: Using `StandardScaler` to scale the features, ensuring that all input variables contribute equally to the model training.

## Model Development
The Random Forest Regression model is employed for price prediction, involving the following key steps:
1. **Data Splitting**: The dataset is split into training (80%) and testing (20%) sets to evaluate the model's performance.
2. **Model Training**: The training dataset is used to fit the Random Forest model, enabling it to learn from historical data.
3. **Cross-Validation**: Implementing cross-validation techniques to ensure robustness and reduce overfitting by evaluating the model on multiple subsets of the training data.

## Performance Metrics
The model's performance is assessed using the following metrics:
- **Mean Absolute Error (MAE)**: Indicates the average magnitude of errors in a set of predictions, without considering their direction.
- **Mean Squared Error (MSE)**: Measures the average of the squares of the errors—more sensitive to large errors.
- **Root Mean Squared Error (RMSE)**: Provides the error in the original scale, making it easier to interpret.

The results of cross-validation include the mean and standard deviation of the MSE to provide a comprehensive evaluation of model performance.

## How to Run the Project

### Prerequisites
To run this project, ensure that the following are installed:
- **Python**: Version 3.x
- **Jupyter Notebook** or any preferred Integrated Development Environment (IDE) such as PyCharm.
- Required libraries can be installed using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn

or pip install -r requirements.txt

