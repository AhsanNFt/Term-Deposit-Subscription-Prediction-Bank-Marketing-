
# 🏦 Term Deposit Subscription Prediction

## 📌 Project Overview & Task Objective

The `Term Deposit Subscription Prediction_Main.ipynb` notebook focuses on predicting whether a customer will subscribe to a term deposit using the UCI Bank Marketing Dataset. The objective is to build reliable classification models to help banks identify potential subscribers.

## 📂 Dataset Information

This project uses the **Bank Marketing Dataset** from the UCI repository. It contains client profile details, contact history, and outcomes of previous marketing campaigns.

**Target Variable**:  
- `y`: Indicates whether the client subscribed to a term deposit (`yes` or `no`)

**Key Challenges Addressed**:
- Class imbalance in the target variable
- High-cardinality categorical features
- Identifying strong predictors for subscription

## ✨ Features

- Data loading, inspection, and cleaning
- Encoding of categorical features (Label and One-Hot)
- Addressing target class imbalance using:
  - `class_weight='balanced'`
  - SMOTE (Synthetic Minority Over-sampling Technique)
- EDA with visual insights
- Training and evaluation of:
  - Logistic Regression (balanced weight)
  - Random Forest (balanced weight)
  - Random Forest (with SMOTE)
- Evaluation through classification reports, confusion matrices, and ROC curves

## 🛠️ Installation

To run this notebook locally, install Python and the following dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## 🚀 Approach

**Steps followed in the notebook:**

- **Library Import**: Loaded Python libraries for data processing, visualization, and machine learning.

- **Data Preprocessing**:
  - Removed duplicates and irrelevant features
  - Encoded binary and multiclass categorical variables using `LabelEncoder` and `OneHotEncoder`
  - **Handled class imbalance** using two techniques:
    - `class_weight='balanced'` in Logistic Regression and Random Forest
    - **SMOTE** oversampling for Random Forest to synthetically balance minority class

- **Exploratory Data Analysis (EDA)**:
  - Visualized class imbalance and feature relationships with the target
  - Used correlation heatmaps to identify redundant or irrelevant features

- **Model Training**:
  - **Logistic Regression**: Trained using `class_weight='balanced'`
  - **Random Forest (Balanced)**: Trained using `class_weight='balanced'`
  - **Random Forest (SMOTE)**: Trained on oversampled data using SMOTE

- **Model Evaluation**:
  - Evaluated each model using **confusion matrix**, **classification report**, and **ROC curve**
  - Compared model performance using **accuracy**, **precision**, **recall**, **F1-score**, and **AUC-ROC**

## 🧰 Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Imbalanced-learn**

## 📉 Visualizations

### 📊 Exploratory Data Analysis (EDA)

- **Target Class Distribution**: Revealed strong class imbalance
- **Barplots & Countplots**: Showed trends in subscription across job types, marital status, contact method
- **Correlation Heatmap**: Helped in understanding multicollinearity and feature selection

---

## 📈 Model Performance Visualizations

### 🔹 Logistic Regression (Balanced)

#### 📉 Confusion Matrix
Shows how well the model classified subscribed vs. not subscribed clients.

#### 📈 ROC Curve
Visualizes the trade-off between true positive and false positive rates. AUC indicates model’s ability to distinguish between classes.

---

### 🌲 Random Forest (Balanced)

#### 📉 Confusion Matrix
Shows classification results with weight-balanced learning on imbalanced data.

#### 📈 ROC Curve
Displays overall model discriminative ability under `class_weight='balanced'`.

---

### 🌱 Random Forest (SMOTE)

#### 📉 Confusion Matrix
Trained on oversampled (SMOTE) data, this confusion matrix reveals improved sensitivity to the minority class.

#### 📈 ROC Curve
Shows how well the SMOTE-enhanced model performs compared to non-SMOTE versions.

---

## 📊 Results and Insights

### Key Insights:
- **Imbalance Handling Works**: Both `class_weight='balanced'` and SMOTE improved model sensitivity toward the minority class (`yes`)
- **Top Predictors**: `duration`, `poutcome`, `contact`, and `previous` were among the most influential features
- **Model Comparisons**:
  - **Logistic Regression** offered interpretability with decent AUC
  - **Random Forest (Balanced)** handled non-linearity well while staying fair to both classes
  - **Random Forest (SMOTE)** improved recall and overall sensitivity but may require tuning to avoid overfitting

---

## 🧾 Overall Conclusion

- The notebook successfully demonstrates a complete pipeline for predicting term deposit subscriptions using real-world banking data.
- **Class imbalance**, a common issue in financial datasets, was effectively addressed using both **class weighting** and **SMOTE**, resulting in improved model fairness and recall.
- Among the tested models, **Random Forest with SMOTE** achieved the best sensitivity toward minority class (subscribers), while **Logistic Regression** provided the best interpretability.
- These models and insights can assist financial institutions in designing more effective and targeted marketing strategies, reducing campaign costs and improving customer conversion.
- Future work can involve:
  - Hyperparameter tuning (e.g., grid/random search)
  - Cross-validation
  - Trying advanced ensemble models like **XGBoost**, **LightGBM**, or **CatBoost**

---

## 🧪 Usage

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Term-Deposit-Prediction.git

# 2. Navigate to the project directory
cd Term-Deposit-Prediction

# 3. Launch the notebook
jupyter notebook Term_Deposit_Subscription_Prediction_Main.ipynb
```

## 🤝 Contributing

Contributions are welcome! Submit issues or pull requests to improve the code or documentation.

## 📬 Contact

For feedback or collaboration:  
- GitHub: `AhsanNFt`  
- Email: `syedahsan0991@gmail.com`
