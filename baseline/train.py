import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("baseline/final_dataset.csv")

X = df.drop(["UniqueID","Label"], axis=1)
y = df["Label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

j48_model = DecisionTreeClassifier(random_state=42)
j48_model.fit(X_train, y_train)

y_pred_j48 = j48_model.predict(X_test)

print("J48 Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_j48, target_names=le.classes_))
