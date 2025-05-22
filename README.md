Below is a description for the GitHub repository and a README file for the provided Jupyter notebook, `Assignment2_supervised_learning_flow.ipynb`, which details a supervised learning workflow using the Titanic dataset.

---

### GitHub Repository Description

**Titanic Supervised Learning Analysis**

This repository contains a comprehensive machine learning project analyzing the Titanic dataset to predict passenger survival using supervised learning techniques. The Jupyter notebook implements a full machine learning pipeline, including data loading, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning with GridSearchCV, and performance evaluation. Three classification models—K-Nearest Neighbors (KNN), Decision Tree, and Naive Bayes—are trained and compared, with the best-performing Decision Tree model applied to the test set. The project showcases data preprocessing, visualization, and model evaluation using metrics like accuracy, precision, recall, and F1 score. Ideal for learning supervised learning workflows and applying machine learning to real-world datasets.

---

### README File


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

You can install them using pip:
```bash
pip install pandas matplotlib seaborn scikit-learn
```

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/titanic-supervised-learning-analysis.git
   ```
2. Ensure the dataset files (`titanic_train.csv` and `titanic_test.csv`) are in the same directory as the notebook.
3. Open the Jupyter notebook:
   ```bash
   jupyter notebook Assignment2_supervised_learning_flow.ipynb
   ```
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
- Tomer Golan (ID: 3540)
- Shahar Rushetzky (ID: 0469)
- Shalev Atsis (ID: 6961)

## AI Assistance
The project utilized OpenAI's ChatGPT and xAI's Grok for guidance on dataset explanation, code generation for data preprocessing, visualization, and hyperparameter tuning. Full details of the prompts used are included in the notebook.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



---

This README provides a clear overview of the project, instructions for running the code, and details about the dataset, models, and results, making it suitable for a GitHub repository. Let me know if you need further modifications or additional sections!
