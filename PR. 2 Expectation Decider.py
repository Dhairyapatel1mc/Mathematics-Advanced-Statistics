

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from scipy.stats import binom

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


np.random.seed(42)
n = 200


study_hours = np.random.randint(2, 21, n)                 
attendance = np.random.randint(50, 101, n)                
group_discussion = np.random.choice(['Yes', 'No'], n, p=[0.6, 0.4])
previous_test_score = np.random.randint(35, 101, n)      


pass_score = (
    0.25 * study_hours +
    0.03 * attendance +
    0.04 * previous_test_score +
    np.where(group_discussion == 'Yes', 2, 0)
)


final_exam_pass = np.where(pass_score > 10, 'Pass', 'Fail')


df = pd.DataFrame({
    'study_hours': study_hours,
    'attendance': attendance,
    'group_discussion': group_discussion,
    'previous_test_score': previous_test_score,
    'final_exam_pass': final_exam_pass
})

print("=" * 60)
print("DATASET (FIRST 10 ROWS)")
print("=" * 60)
print(df.head(10))

print("\nDATASET SHAPE:", df.shape)


df.to_csv('student_probability_dataset.csv', index=False)
print("\nDataset saved as student_probability_dataset.csv")


# ==============================================================
# 1. UNDERSTANDING THE BASICS
# ==============================================================

print("\n" + "=" * 60)
print("1. UNDERSTANDING THE BASICS")
print("=" * 60)

print("""
Probability:
Probability is the measure of how likely an event is to happen.
Its value ranges from 0 to 1.

0   = Impossible event
1   = Certain event
0.5 = Equal chance

Basic Formula:
P(A) = Number of favorable outcomes / Total outcomes
""")

print("""
Key Probability Terminology:
1. Experiment - Process that produces outcomes.
2. Outcome - Single result.
3. Sample Space - All possible outcomes.
4. Event - Set of outcomes.
5. Independent Events - One event does not affect another.
6. Conditional Probability - Probability of A given B.
""")

print("""
Examples from Dataset:
1. Student passes the exam.
2. Student attends more than 80%.
3. Student studies more than 10 hours/week.
""")


# ==============================================================
# 2. TYPES OF EVENTS
# ==============================================================

print("\n" + "=" * 60)
print("2. TYPES OF EVENTS")
print("=" * 60)


empirical_pass = (df['final_exam_pass'] == 'Pass').mean()

print(f"Empirical Probability of Passing = {empirical_pass:.4f}")


theoretical_head = 1 / 2

print(f"Theoretical Probability of Head in Coin Toss = {theoretical_head:.4f}")


# ==============================================================
# 3. RANDOM VARIABLE & PROBABILITY DISTRIBUTION
# ==============================================================

print("\n" + "=" * 60)
print("3. RANDOM VARIABLE & PROBABILITY DISTRIBUTION")
print("=" * 60)


n_trials = 3
p = empirical_pass



x_values = np.arange(0, 4)
probabilities = binom.pmf(x_values, n_trials, p)

distribution = pd.DataFrame({
    'X (Number of Passes)': x_values,
    'Probability': probabilities
})

print(distribution)



mean = n_trials * p
variance = n_trials * p * (1 - p)

print(f"\nMean = {mean:.4f}")
print(f"Variance = {variance:.4f}")


# ==============================================================
# 4. VENN DIAGRAM
# ==============================================================

print("\n" + "=" * 60)
print("4. VENN DIAGRAM")
print("=" * 60)


A = set(df[df['study_hours'] > 10].index)


B = set(df[df['attendance'] > 80].index)

print(f"Students with study_hours > 10: {len(A)}")
print(f"Students with attendance > 80: {len(B)}")
print(f"Students satisfying both: {len(A & B)}")


plt.figure(figsize=(6, 6))
venn2([A, B],
      set_labels=('Study > 10 hrs', 'Attendance > 80%'))
plt.title('Venn Diagram of Student Groups')
plt.show()


# ==============================================================
# 5. CONTINGENCY TABLE & PROBABILITY CALCULATIONS
# ==============================================================

print("\n" + "=" * 60)
print("5. CONTINGENCY TABLE")
print("=" * 60)

contingency = pd.crosstab(
    df['group_discussion'],
    df['final_exam_pass'],
    margins=True
)

print(contingency)


pass_given_yes = (
    contingency.loc['Yes', 'Pass'] /
    contingency.loc['Yes', 'All']
)

print(f"\nP(Pass | Group Discussion = Yes) = {pass_given_yes:.4f}")


# ==============================================================
# 6. UNDERSTANDING RELATIONSHIPS
# ==============================================================

print("\n" + "=" * 60)
print("6. UNDERSTANDING RELATIONSHIPS")
print("=" * 60)


overall_pass = contingency.loc['All', 'Pass'] / contingency.loc['All', 'All']

print(f"Overall P(Pass) = {overall_pass:.4f}")
print(f"P(Pass | Group Discussion = Yes) = {pass_given_yes:.4f}")


if abs(pass_given_yes - overall_pass) < 0.05:
    relation = "Approximately Independent"
else:
    relation = "Dependent"

print(f"Relationship: {relation}")

print("""
Mutually Exclusive?
No.
A student can participate in group discussions AND pass the exam simultaneously.
""" )


# ==============================================================
# 7. BAYES THEOREM APPLICATION
# ==============================================================

print("\n" + "=" * 60)
print("7. BAYES THEOREM APPLICATION")
print("=" * 60)



P_H_given_P = 0.70
P_H_given_F = 0.40
P_H = 0.60


P_Pass = 2 / 3
P_Fail = 1 - P_Pass



P_Pass_given_H = (P_H_given_P * P_Pass) / P_H

print(f"P(Pass) = {P_Pass:.4f}")
print(f"P(Fail) = {P_Fail:.4f}")
print(f"P(Pass | High Attendance) = {P_Pass_given_H:.4f}")


# ==============================================================
# FINAL SUMMARY
# ==============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("""
1. Probability measures the likelihood of an event.
2. Empirical probability is based on observed data.
3. The number of passing students among 3 follows a Binomial Distribution.
4. Mean = np and Variance = np(1-p).
5. Students who study more and attend regularly have higher success rates.
6. Group discussion participation generally increases probability of passing.
7. Group discussion and passing are dependent events.
8. Bayes Theorem showed that:
   P(Pass | High Attendance) = 0.7778

Conclusion:
Students with higher attendance, better previous scores,
more study hours, and active group discussion participation
have a much greater probability of passing the final exam.
""")