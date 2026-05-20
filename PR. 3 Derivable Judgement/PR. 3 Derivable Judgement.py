import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.weightstats import ztest
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import uuid
import warnings
warnings.filterwarnings("ignore")



np.random.seed(42)
n = 500  


def random_date(start_date, end_date):
    delta = end_date - start_date
    return start_date + timedelta(days=np.random.randint(0, delta.days))

def generate_disease_probability(age, bmi, smoker, exercise, glucose, bp):
    prob = 0.05
    
    if age > 45:
        prob += 0.10
    if bmi > 30:
        prob += 0.12
    if smoker == "Smoker":
        prob += 0.15
    if exercise in ["Rarely", "Never"]:
        prob += 0.08
    if glucose > 126:
        prob += 0.25
    if bp > 140:
        prob += 0.12
    
    return min(prob, 0.95)

age_groups = ["18-25", "26-35", "36-45", "46-60", "60+"]
genders = ["Male", "Female", "Other"]
regions = ["North", "South", "East", "West"]
smoking_statuses = ["Smoker", "Non-Smoker", "Former Smoker"]
exercise_levels = ["Daily", "Weekly", "Rarely", "Never"]


data = []

for _ in range(n):
    age = np.random.randint(18, 81)
    weight = np.random.randint(45, 121)
    gender = np.random.choice(genders, p=[0.48, 0.50, 0.02])
    region = np.random.choice(regions)
    smoking = np.random.choice(smoking_statuses, p=[0.25, 0.55, 0.20])
    exercise = np.random.choice(exercise_levels, p=[0.25, 0.35, 0.25, 0.15])

    height = np.random.uniform(1.5, 1.9) 
    bmi = weight / (height ** 2)

    blood_pressure = np.random.normal(120, 15)
    cholesterol = np.random.normal(200, 35)
    glucose = np.random.normal(100, 25)


    diabetes_prob = generate_disease_probability(
        age, bmi, smoking, exercise, glucose, blood_pressure
    )
    diabetes = np.random.rand() < diabetes_prob

    hypertension_prob = min(
        0.05 + (age > 50) * 0.15 + (bmi > 30) * 0.10 + (blood_pressure > 140) * 0.40,
        0.95
    )
    hypertension = np.random.rand() < hypertension_prob

  
    if age <= 25:
        age_group = "18-25"
    elif age <= 35:
        age_group = "26-35"
    elif age <= 45:
        age_group = "36-45"
    elif age <= 60:
        age_group = "46-60"
    else:
        age_group = "60+"

    visit_date = random_date(datetime(2023, 1, 1), datetime(2025, 12, 31))

    data.append({
        "record_id": str(uuid.uuid4()),
        "age_group": age_group,
        "age": age,
        "weight": weight,
        "gender": gender,
        "region": region,
        "smoking_status": smoking,
        "exercise_frequency": exercise,
        "bmi": round(bmi, 2),
        "blood_pressure": round(blood_pressure, 2),
        "diabetes": diabetes,
        "hypertension": hypertension,
        "cholesterol_level": round(cholesterol, 2),
        "glucose_level": round(glucose, 2),
        "visit_date": visit_date.date()
    })


df = pd.DataFrame(data)


df.to_csv("health_dataset.csv", index=False)

print("Dataset generated successfully!")
print("Shape:", df.shape)
print(df.head())



print("\n" + "="*50)
print("HYPOTHESES")
print("="*50)

print("""
Hypothesis 1:
H0: Smoking has no effect on diabetes prevalence.
H1: Smoking affects diabetes prevalence.

Hypothesis 2:
H0: Mean BMI is the same across all age groups.
H1: Mean BMI differs across age groups.

Hypothesis 3:
H0: Age and BMI are not correlated.
H1: Age and BMI are correlated.
""")



print("\n" + "="*50)
print("95% CONFIDENCE INTERVALS")
print("="*50)

for col in ["age", "weight", "bmi", "blood_pressure"]:
    mean = df[col].mean()
    sem = stats.sem(df[col])
    ci = stats.t.interval(0.95, len(df[col])-1, loc=mean, scale=sem)

    print(f"{col}:")
    print(f"  Mean = {mean:.2f}")
    print(f"  95% CI = ({ci[0]:.2f}, {ci[1]:.2f})")



print("\n" + "="*50)
print("CRITICAL VALUE")
print("="*50)

alpha = 0.05
critical_z = stats.norm.ppf(1 - alpha/2)
critical_t = stats.t.ppf(1 - alpha/2, df=len(df)-1)

print(f"Alpha = {alpha}")
print(f"Critical Z-value = ±{critical_z:.4f}")
print(f"Critical T-value = ±{critical_t:.4f}")



print("\n" + "="*50)
print("T-TEST: BMI of Smokers vs Non-Smokers")
print("="*50)

smokers_bmi = df[df["smoking_status"] == "Smoker"]["bmi"]
nonsmokers_bmi = df[df["smoking_status"] == "Non-Smoker"]["bmi"]

t_stat, p_value = stats.ttest_ind(smokers_bmi, nonsmokers_bmi)

print(f"T-statistic = {t_stat:.4f}")
print(f"P-value = {p_value:.6f}")

if p_value < alpha:
    print("Result: Reject H0")
    print("Interpretation: Mean BMI differs significantly.")
else:
    print("Result: Fail to Reject H0")
    print("Interpretation: No significant difference in mean BMI.")



print("\n" + "="*50)
print("CHI-SQUARE TEST: Smoking Status vs Diabetes")
print("="*50)

contingency_table = pd.crosstab(df["smoking_status"], df["diabetes"])

chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print("Contingency Table:")
print(contingency_table)
print(f"\nChi-square = {chi2:.4f}")
print(f"Degrees of Freedom = {dof}")
print(f"P-value = {p_value:.6f}")

if p_value < alpha:
    print("Result: Reject H0")
    print("Interpretation: Smoking and diabetes are significantly associated.")
else:
    print("Result: Fail to Reject H0")
    print("Interpretation: No significant association.")



print("\n" + "="*50)
print("ANOVA TEST: BMI across Age Groups")
print("="*50)

groups = [group["bmi"].values for _, group in df.groupby("age_group")]
f_stat, p_value = stats.f_oneway(*groups)

print(f"F-statistic = {f_stat:.4f}")
print(f"P-value = {p_value:.6f}")

if p_value < alpha:
    print("Result: Reject H0")
    print("Interpretation: BMI differs significantly across age groups.")
else:
    print("Result: Fail to Reject H0")
    print("Interpretation: No significant difference in BMI across age groups.")



print("\n" + "="*50)
print("COVARIANCE AND CORRELATION: Age vs BMI")
print("="*50)

covariance = df["age"].cov(df["bmi"])
correlation, corr_p = stats.pearsonr(df["age"], df["bmi"])

print(f"Covariance = {covariance:.4f}")
print(f"Pearson Correlation = {correlation:.4f}")
print(f"P-value = {corr_p:.6f}")

if corr_p < alpha:
    print("Result: Reject H0")
    print("Interpretation: Significant correlation exists.")
else:
    print("Result: Fail to Reject H0")
    print("Interpretation: No significant correlation.")



plt.figure(figsize=(6, 4))
sns.countplot(x="diabetes", data=df)
plt.title("Diabetes Distribution")
plt.savefig("diabetes_distribution.png")
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(x="smoking_status", hue="diabetes", data=df)
plt.title("Smoking Status vs Diabetes")
plt.savefig("smoking_vs_diabetes.png")
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(x="age_group", y="bmi", data=df)
plt.title("BMI Across Age Groups")
plt.savefig("bmi_by_age_group.png")
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(x="age", y="bmi", data=df)
plt.title("Age vs BMI")
plt.savefig("age_vs_bmi.png")
plt.show()



print("\n" + "="*50)
print("FINAL SUMMARY")
print("="*50)

print("""
1. Confidence intervals were calculated for Age, Weight, BMI, and Blood Pressure.
2. T-test compared BMI between smokers and non-smokers.
3. Chi-square test checked association between smoking and diabetes.
4. ANOVA tested whether BMI differs across age groups.
5. Covariance and Pearson correlation measured relationship between Age and BMI.
6. All hypotheses were interpreted using alpha = 0.05.
7. Dataset and charts were saved successfully.
""")

print("Files Generated:")
print("- health_dataset.csv")
print("- diabetes_distribution.png")
print("- smoking_vs_diabetes.png")
print("- bmi_by_age_group.png")
print("- age_vs_bmi.png")