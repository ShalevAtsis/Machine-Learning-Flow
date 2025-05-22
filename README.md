# Titanic Supervised Learning Analysis

## Overview
This repository contains a Jupyter notebook, `Assignment2_supervised_learning_flow.ipynb`, which implements a complete supervised learning workflow to predict passenger survival in the Titanic disaster using the Titanic dataset. The project includes data loading, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, and performance evaluation using multiple machine learning models.

## Dataset
The Titanic dataset is used, containing passenger information such as:
- **Pclass**: Passenger class (1st, 2nd, or 3rd)
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings or spouses aboard
- **Parch**: Number of parents or children aboard
- **Fare**: Ticket fare
- **Embarked**: Port of embarkation
- **Survived**: Survival status (0 = did not survive, 1 = survived)

The dataset is split into training (`titanic_train.csv`) and test (`titanic_test.csv`) sets.

## Project Structure
The notebook is organized into the following parts:
1. **Student Details**: Information about the contributors.
2. **AI Assistance**: Documentation of AI tools (ChatGPT and Grok) used for guidance, including prompts for dataset explanation, code generation, and visualization.
3. **Learning Problem**: Description of the binary classification task to predict survival based on passenger features.
4. **Initial Preparations**:
   - Loading the dataset using pandas.
   - Exploratory Data Analysis (EDA) with summary statistics and visualizations of continuous variables (Age, Fare).
5. **Model Training**:
   - Training K-Nearest Neighbors (KNN), Decision Tree, and Naive Bayes models.
   - Hyperparameter tuning using GridSearchCV for KNN and Decision Tree.
6. **Model Evaluation**:
   - Evaluation of models using accuracy, precision, recall, and F1 score.
   - Application of the best Decision Tree model (with preprocessing pipeline) on the test set.
7. **Performance Estimation**: Comparison of predicted vs. actual survival outcomes on the test set.

## Dependencies
To run the notebook, install the following Python libraries:
- pandas
- matplotlib
- seaborn
- scikit-learn

## How to Run
1. Clone this repository.
2. Ensure the dataset files (`titanic_train.csv` and `titanic_test.csv`) are in the same directory as the notebook.
3. Open the Jupyter notebook:
4. Run all cells to execute the full machine learning pipeline.

## Results
- **Best Model**: Decision Tree with hyperparameters `criterion='entropy', max_depth=10, min_samples_split=2, min_samples_leaf=1`.
- **Test Set Performance**:
  - Accuracy: 0.782
  - Precision: 0.739
  - Recall: 0.557
  - F1 Score: 0.636

The notebook includes a detailed comparison of model performance and a table of predicted vs. actual survival outcomes for the test set.

## Contributors

### Shalev Atsis
- 📞 Phone: [+972 58-5060699](tel:+972585060699)
- 📧 Email: [shalevatsis@gmail.com](mailto:shalevatsis@gmail.com)
- 🔗 LinkedIn: [Shalev Atsis](https://www.linkedin.com/in/shalev-atsis-software-developer)

### Tomer Golan
- 📞 Phone: [+972 53-3454053](tel:+972533454053)
- 📧 Email: [tomergolan2016@gmail.com](mailto:tomergolan2016@gmail.com)
- 🔗 LinkedIn: [Tomer Golan](https://www.linkedin.com/in/tomer-golan24/)

### Shahar Rushetzky
- 📞 Phone: [+972 52-7729726](tel:+972527729726)
- 📧 Email: [sroshetzky@gmail.com](mailto:sroshetzky@gmail.com)
- 🔗 LinkedIn: [Shahar Rushetzky](https://www.linkedin.com/in/shahar-rushetzky)
  
**Computer Science Students, HIT College**

## AI Assistance
The project utilized OpenAI's ChatGPT and xAI's Grok for guidance on dataset explanation, code generation for data preprocessing, visualization, and hyperparameter tuning. Full details of the prompts used are included in the notebook.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
