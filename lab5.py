import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


wine_data = load_wine()
X = wine_data.data
y = wine_data.target


df = pd.DataFrame(X, columns=wine_data.feature_names)
df['target'] = y


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)


new_wine = np.array([[4.25, 1.68, 2.51, 15.3, 125.0, 2.70, 3.02, 0.25, 2.27, 5.61, 1.08, 3.82, 1065.0]])


new_wine_scaled = scaler.transform(new_wine)


predicted_class = model.predict(new_wine_scaled)

print(f'Предсказанный класс для нового вина: {predicted_class[0]}')
print(f'Название класса: {wine_data.target_names[predicted_class[0]]}')