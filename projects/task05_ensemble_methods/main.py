import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cars_data = pd.read_csv('CarsData.csv')
print(cars_data.head())

missing_values = cars_data.isnull().sum()
summary_statistics = cars_data.describe()

print(missing_values)
print(summary_statistics)

transmission_counts = cars_data['transmission'].value_counts()
fuel_type_counts = cars_data['fuelType'].value_counts()
manufacturer_counts = cars_data['Manufacturer'].value_counts()

print(transmission_counts)
print(fuel_type_counts)
print(manufacturer_counts.head(10))

numeric_cols = cars_data.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_cols.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()
