# Import libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

print("Feature names:", diabetes.feature_names)  
print("Feature data shape:", diabetes.data.shape)
print("Target data shape:", diabetes.target.shape)
data = fetch_openml(name='diabetes', version=1, as_frame=True)

import pandas as pd

df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target
print(diabetes.DESCR)
print(df.sample(5))
print(df.describe())
df.dtypes

#בדתי מה העמודות שלי על מנת לבחור נתונים מעניינים לאנליטיקות 
features = list(df.columns)
print("Available features:", features)
selected_features = [features[0], features[2]]
print("Selected features: ", selected_features)

selected_features = ['age','bmi', 's6']
fig, axs  = plt.subplots(1, len(selected_features), figsize = (20,3))

for ax, feature in zip(axs, selected_features):
    ax.hist(df[feature], bins=20, color='skyblue', edgecolor='black')
    ax.set_xlabel(feature)

reference_feature = selected_features[0]
y = df[reference_feature]

fig, axs  = plt.subplots(1, len(selected_features), figsize = (20,3))

for ax, feature in zip(axs, selected_features):
  ax.scatter(df[feature], y)
  ax.set_xlabel(feature)
  ax.set_ylabel(reference_feature)
  ax.set_title(f'Scatter plot: {feature} vs {reference_feature}')
  plt.grid(True)

  plt.tight_layout()

plt.show()

#glucose levels vs bmi scatter 
reference_feature = selected_features[1]  # The reference feature
comparison_feature = selected_features[2]  # A feature to compare to

# Create a scatter plot for the selected pair
plt.figure(figsize=(8, 6))
plt.scatter(df[reference_feature], df[comparison_feature], alpha=0.6)
plt.xlabel('BMI', fontsize = 16)
plt.ylabel('Blood Glucose Level', fontsize = 16)
plt.title('Scatter Plot: BMI vs Blood Glucose Levels In Diabetic Patients', fontsize = 20)
plt.grid(True)

# Save the plot as an image file
plt.savefig('correlation_plot.png')

plt.show()
