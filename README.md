# 📊 Analyzing Income Distribution and Household Demographics Using Python

## 📌 Project Overview

This project focuses on analyzing household income and demographic data using **Python**.
It applies statistical techniques and data visualization to understand income distribution, variability, and relationships with demographic factors.

---

## 🎯 Objectives

* Analyze household income distribution
* Apply descriptive statistical methods
* Understand data types (categorical & numerical)
* Measure central tendency and dispersion
* Study skewness and kurtosis
* Visualize patterns using graphs

---

## 🗂 Dataset Description

The dataset contains **200 records** with the following columns:

| Column Name           | Description                                |
| --------------------- | ------------------------------------------ |
| Household ID          | Unique ID for each household               |
| Age of Household Head | Age of head of household                   |
| Household Income      | Monthly income                             |
| Education Level       | Primary / Secondary / Graduate / Post-Grad |
| Family Size           | Number of family members                   |
| Owns House            | Yes / No                                   |
| Urban Rural           | Urban / Rural                              |

---

## 🧰 Tools & Technologies

* Python 🐍
* Pandas
* NumPy
* Matplotlib
* Seaborn
* VS Code

---

## 🔍 Analysis Performed

### 1. Data Types

* Identified categorical and numerical variables

### 2. Central Tendency

* Mean
* Median
* Mode

### 3. Measures of Dispersion

* Range
* Variance
* Standard Deviation
* Interquartile Range (IQR)

### 4. Distribution Analysis

* Histogram
* KDE Plot
* Gaussian Distribution Curve

### 5. Shape of Distribution

* Skewness
* Kurtosis

### 6. Data Visualization

* Income Distribution Histogram
* Boxplot (Income vs Education Level)
* Boxplot (Urban vs Rural)
* Scatter Plot (Age vs Income)

---

## 📊 Key Insights

* Income distribution shows variability across households
* Education level impacts income distribution
* Urban households tend to have higher income ranges
* Data may show skewness depending on generated values

---

## ▶️ How to Run the Project

1. Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn
```

2. Run the Python script or vs code

```bash
python project.py
```

---

## 📁 Project Structure

```
├── data/
│   └── household_data.csv
├── notebook/
│   └── analysis.ipynb
├── src/
│   └── project.py
├── README.md
```

---

## 📌 Conclusion

This project demonstrates how statistical analysis and visualization techniques can be used to understand income patterns and demographic influences effectively.

---

## 👨‍💻 Author

Dhairya patel

---

## 📜 License

This project is for academic purposes.



📘 Expectation Decider Probability Project
📌 Project Overview

This project simulates and analyzes the probability that a student will pass a competitive mathematics exam using Python. A synthetic dataset of 200 students is generated and analyzed using core concepts from Probability and Statistics.

The project was developed as part of an educational research case study called Expectation Decider, where the goal is to identify the factors that most influence exam success.

🎯 Project Objective

You are hired as a Junior Data Analyst in an educational research institute. The institute wants to build a model that predicts whether a student will pass a competitive mathematics exam based on:

Study Hours per Week
Attendance Percentage
Participation in Group Discussions
Previous Test Score

Using a generated dataset of 200 students, this project performs a complete probability analysis and derives meaningful insights.

📂 Dataset Fields
Column Name	Description
study_hours	Number of hours studied per week
attendance	Percentage attendance in lectures
group_discussion	Participation in group discussions (Yes / No)
previous_test_score	Previous internal test marks (0–100)
final_exam_pass	Final result (Pass / Fail)
📊 Topics Covered
1. Understanding the Basics
Definition of Probability
Key Probability Terminology
Real-world probability events from the dataset
2. Types of Events
Empirical Probability
Theoretical Probability
3. Random Variable & Probability Distribution
Binomial Distribution
Probability Distribution Table
Mean and Variance
4. Venn Diagram
Students studying more than 10 hours/week
Students attending more than 80%
Overlap between both groups
5. Contingency Table & Conditional Probability
Group Discussion vs Exam Result
Probability of passing given discussion participation
6. Understanding Relationships
Independent vs Dependent Events
Mutually Exclusive Events
7. Bayes Theorem Application
Probability of passing given high attendance
🛠️ Technologies Used
Python
NumPy
Pandas
Matplotlib
matplotlib-venn
SciPy
Jupyter Notebook
 or Visual Studio Code
📦 Installation
pip install numpy pandas matplotlib matplotlib-venn scipy
▶️ How to Run
python expectation_decider.py

Or open the notebook:

jupyter notebook Expectation_Decider_Probability_Project.ipynb
📁 Project Structure
Expectation-Decider-Probability-Project/
│── expectation_decider.py
│── Expectation_Decider_Probability_Project.ipynb
│── student_probability_dataset.csv
│── README.md
📈 Sample Outputs
Synthetic dataset of 200 students
Binomial probability distribution table
Mean and variance calculations
Venn diagram visualization
Contingency table
Conditional probability results
Bayes theorem solution
Final analytical summary
📌 Key Insights
Students with higher attendance have a significantly greater chance of passing.
Studying more than 10 hours per week improves exam success.
Participation in group discussions positively affects performance.
Previous test scores are strong indicators of final results.
Group discussion and passing are generally dependent events.

Using Bayes Theorem:

P(Pass∣High Attendance)=0.7778

This means a student with high attendance has approximately a 77.78% chance of passing.

🚀 Future Enhancements
Build a machine learning prediction model
Add hypothesis testing
Create an interactive dashboard using Power BI
Deploy as a web application using Streamlit
👨‍💻 Author

Your Name Here
Dhiarya patel


📜 License

This project is open source and available under the MIT License.
