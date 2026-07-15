import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Dataset Generation (హిమోగ్లోబిన్, RBC కౌంట్ డేటా)
np.random.seed(42)
n_samples = 1000

gender = np.random.randint(0, 2, n_samples) # 0=Female, 1=Male
hemoglobin = np.random.uniform(8.0, 17.5, n_samples)
rbc_count = np.random.uniform(3.5, 5.8, n_samples)

# Anemia కండిషన్ రూల్స్ వర్తింపజేయడం
anemic = []
for g, hb in zip(gender, hemoglobin):
    if (g == 0 and hb < 12.0) or (g == 1 and hb < 13.5):
        anemic.append(1) # Anemia ఉంది
    else:
        anemic.append(0) # Normal గా ఉంది

df = pd.DataFrame({'Gender': gender, 'Hemoglobin': hemoglobin, 'RBC_Count': rbc_count, 'Result': anemic})

# 2. Data Splitting
X = df[['Gender', 'Hemoglobin', 'RBC_Count']]
y = df['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy Check
predictions = model.predict(X_test)
print(f"Model Training Done! Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 4. Save trained model object
with open('anemia_detector.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Saved model file successfully!")