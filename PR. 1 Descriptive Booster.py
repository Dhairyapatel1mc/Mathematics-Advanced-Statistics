
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

n = 200

data = pd.DataFrame({
    "Household ID": ["H" + str(i) for i in range(1, n+1)],
    "Age of Household Head": np.random.randint(25, 70, n),
    "Household Income": np.random.randint(10000, 100000, n),
    "Education Level": np.random.choice(["Primary", "Secondary", "Graduate", "Post-Grad"], n),
    "Family Size": np.random.randint(2, 8, n),
    "Owns House": np.random.choice(["Yes", "No"], n),
    "Urban Rural": np.random.choice(["Urban", "Rural"], n)
})

print(data.head())

print("\nData Types:\n")
print(data.dtypes)

print("\nCentral Tendency (Income):")
print("Mean:", data["Household Income"].mean())
print("Median:", data["Household Income"].median())
print("Mode:", data["Household Income"].mode()[0])

print("\nCentral Tendency (Age):")
print("Mean:", data["Age of Household Head"].mean())
print("Median:", data["Age of Household Head"].median())
print("Mode:", data["Age of Household Head"].mode()[0])

income = data["Household Income"]

print("\nDispersion Measures (Income):")
print("Range:", income.max() - income.min())
print("Variance:", income.var())
print("Standard Deviation:", income.std())

Q1 = income.quantile(0.25)
Q3 = income.quantile(0.75)
IQR = Q3 - Q1

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)


plt.figure()
sns.histplot(income, kde=True)
plt.title("Income Distribution (Histogram + KDE)")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()


mean = income.mean()
std = income.std()

x = np.linspace(income.min(), income.max(), 100)
y = (1/(std * np.sqrt(2*np.pi))) * np.exp(-(x-mean)**2 / (2*std**2))

plt.figure()
plt.hist(income, density=True)
plt.plot(x, y)
plt.title("Gaussian Distribution Fit")
plt.show()


print("\nShape of Distribution:")
print("Skewness:", income.skew())
print("Kurtosis:", income.kurt())


plt.figure()
sns.boxplot(x="Education Level", y="Household Income", data=data)
plt.title("Income vs Education Level")
plt.show()


plt.figure()
sns.boxplot(x="Urban Rural", y="Household Income", data=data)
plt.title("Urban vs Rural Income")
plt.show()


plt.figure()
sns.scatterplot(x="Age of Household Head", y="Household Income", data=data)
plt.title("Age vs Income")
plt.show()

data.to_csv("household_data.csv", index=False)