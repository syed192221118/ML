import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# Prepare the dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Encode categorical variables
df_encoded = pd.get_dummies(df.drop('PlayTennis', axis=1))
target = df['PlayTennis'].apply(lambda x: 1 if x == 'Yes' else 0)

# Train the Decision Tree model
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(df_encoded, target)

# Print the decision tree
tree_rules = export_text(clf, feature_names=list(df_encoded.columns))
print(tree_rules)

# Classify a new sample
new_sample = pd.DataFrame({
    'Outlook': ['Sunny'],
    'Temperature': ['Cool'],
    'Humidity': ['High'],
    'Windy': [True]
})

new_sample_encoded = pd.get_dummies(new_sample)
new_sample_encoded = new_sample_encoded.reindex(columns=df_encoded.columns, fill_value=0)
prediction = clf.predict(new_sample_encoded)

print("Prediction for the new sample:", 'Yes' if prediction[0] == 1 else 'No')
