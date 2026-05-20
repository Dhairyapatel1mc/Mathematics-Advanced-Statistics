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

# 🏥 PR.3 Derivable Judgement  
## Statistical Decision-Making Model using Python

A complete healthcare analytics project using **Python**, **Statistics**, and **Data Visualization**.

This project generates a synthetic healthcare dataset and applies statistical hypothesis testing techniques such as:

- T-Test
- Chi-Square Test
- ANOVA
- Correlation Analysis
- Confidence Intervals

It also creates visualizations and exports datasets automatically.

---

# 📌 Project Overview

This project simulates real-world healthcare data and performs statistical decision-making analysis to identify relationships between:

- Smoking and Diabetes
- Age and BMI
- BMI across Age Groups
- Health Risk Factors

The project demonstrates how statistics are used in healthcare analytics and data science.

---

# 📂 Project Structure

```bash
project-folder/
│
├── statistical_model.py
├── health_dataset.csv
├── diabetes_distribution.png
├── smoking_vs_diabetes.png
├── bmi_by_age_group.png
├── age_vs_bmi.png
├── README.md
```

---

# 🚀 Features

✅ Synthetic Healthcare Dataset Generation  
✅ Confidence Interval Calculation  
✅ Hypothesis Testing  
✅ T-Test Implementation  
✅ Chi-Square Test  
✅ ANOVA Analysis  
✅ Pearson Correlation  
✅ Data Visualization using Seaborn & Matplotlib  
✅ CSV Export Support  

---

# 🛠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main Programming Language |
| Pandas | Data Handling |
| NumPy | Numerical Computation |
| SciPy | Statistical Analysis |
| Statsmodels | Statistical Testing |
| Matplotlib | Visualization |
| Seaborn | Graphical Analysis |

---

# 📦 Installation

Install required libraries:

```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn openpyxl
```

---

# ▶️ Run the Project

```bash
python statistical_model.py
```

---

# 📊 Statistical Tests Included

## 1. T-Test
Compares BMI between smokers and non-smokers.

## 2. Chi-Square Test
Checks association between smoking and diabetes.

## 3. ANOVA
Compares BMI across age groups.

## 4. Pearson Correlation
Measures relationship between Age and BMI.

## 5. Confidence Intervals
Calculates 95% confidence intervals for health variables.

---

# 📈 Visualizations

## Diabetes Distribution

![Diabetes Distribution](diabetes_distribution.png)

---

## Smoking Status vs Diabetes

![Smoking vs Diabetes](smoking_vs_diabetes.png)

---

## BMI Across Age Groups

![BMI by Age Group](bmi_by_age_group.png)

---

## Age vs BMI Correlation

![Age vs BMI](age_vs_bmi.png)

---

# 📄 Sample Dataset Columns

| Column Name | Description |
|---|---|
| age | Patient age |
| bmi | Body Mass Index |
| smoking_status | Smoking category |
| diabetes | Diabetes condition |
| hypertension | Hypertension condition |
| glucose_level | Glucose reading |
| blood_pressure | Blood pressure reading |

---

# 🧠 Statistical Concepts Used

- Hypothesis Testing
- P-Values
- Critical Values
- Confidence Intervals
- Correlation
- Covariance
- Probability Distribution

---

# 📌 Example Hypotheses

### Hypothesis 1
- H0: Smoking has no effect on diabetes prevalence.
- H1: Smoking affects diabetes prevalence.

### Hypothesis 2
- H0: Mean BMI is equal across age groups.
- H1: Mean BMI differs across age groups.

---

# 📷 Add Your Own Screenshots
<img width="1025" height="648" alt="image" src="https://github.com/user-attachments/assets/13ef51cd-7f40-4850-8457-df0bb15e80ab" />
<img width="1020" height="638" alt="image" src="https://github.com/user-attachments/assets/1950b78a-ca7a-4d64-9c42-57c99faa7d51" />
<img width="768" height="513" alt="image" src="https://github.com/user-attachments/assets/ffd1e99c-535e-4263-9158-3919de8d72ae" />





# 📚 Learning Outcomes

This project helps understand:

- Real-world statistical analysis
- Data science workflows
- Healthcare analytics
- Python visualization techniques
- Statistical decision-making models

---

# 🤝 Contributing

Pull requests are welcome.

For major changes, please open an issue first to discuss what you would like to change.

---

# 📜 License

This project is open-source and available under the MIT License.

---

# ⭐ GitHub Tips

If you like this project:

⭐ Star the repository  
🍴 Fork the project  
📢 Share with others  

---

# 👨‍💻 Author
Dhairya patel
Developed using Python and Statistical Analysis techniques.
