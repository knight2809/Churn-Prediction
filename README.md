# Customer Churn Prediction

This repository contains a **Logistic Regression** implementation from scratch using Python and NumPy to predict customer churn for a service-based company. The model identifies whether a client will stop using the service based on several behavioral and demographic features.

## Dataset Overview

The dataset includes client information with the following fields and definitions:

| Feature | Description |
| :--- | :--- |
| **Names** | Name of the customer |
| **Age** | Customer Age |
| **Total_Purchase** | Total Ads Purchased |
| **Account_Manager** | Binary (0=No manager, 1=Account manager assigned) |
| **Years** | Total years as a customer |
| **Num_Sites** | Number of websites that use the service |
| **Onboard_date** | Date that the name of the latest contact was onboarded |
| **Location** | Client HQ Address |
| **Company** | Name of Client Company |
| **Churn** | Target Variable (1=Churned, 0=Stayed) |

---

## Project Workflow

### 1. Data Preprocessing
* **Feature Selection**: Dropped non-numeric and high-cardinality features including `Names`, `Location`, `Company`, and `Onboard_date`.
* **Feature Scaling**: Utilized `StandardScaler` to normalize the feature space for efficient gradient descent.
* **Data Splitting**: Performed an 80/20 train-test split using `train_test_split`.

### 2. Implementation from Scratch
The project avoids high-level machine learning libraries for the model logic, implementing the following core functions manually:
* **Sigmoid Function**: $\frac{1}{1+e^{-z}}$ to map predictions to probabilities.
* **Cost Function**: Implements log-loss and calculates the gradient.
* **Gradient Descent**: Optimizes weights ($\theta$) by iterating over the cost function.
* **Accuracy Function**: Calculates the ratio of correct predictions to total observations.

### 3. Training & Performance
The model was trained with a learning rate ($\alpha$) of **0.01** over **1000** iterations.
* **Accuracy Score**: ~**88.89%** on the test set.
* **Convergence**: The cost history reveals a steady decrease in error over the iterations.

---

## Usage

### Prerequisites
* Python 3.x
* Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn

### Running the Project
1. Ensure `customer_churn.csv` and `customers_stop.csv` are in the project directory.
2. Run the Jupyter Notebook `Churn.ipynb`.

### Prediction on New Data
The model includes a pipeline to process unseen data (`customers_stop.csv`). It applies the same scaling and intercept addition used during training:

```python
# Apply prediction to new data
X_new = scaler.transform(df2)
X_new = np.c_[np.ones((X_new.shape[0], 1)), X_new]
y_pred_new = predict(X_new, theta)
y_pred_new = [1 if i > 0.5 else 0 for i in y_pred_new]

# Insert prediction into dataframe
df2['Churn'] = y_pred_new
```
---
## Visualizing Results
The notebook includes a Cost History plot, which visualizes the optimization process. A sharp decline in the cost function indicates that the model successfully learned patterns from the training data.
