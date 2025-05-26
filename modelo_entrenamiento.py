import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Cargar dataset
df = pd.read_csv('datos/dataset.csv')

X = df[['left_ear', 'right_ear', 'avg_ear', 'movimiento']].values
y = df['label'].values

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Red neuronal simple
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')  # 3 clases
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Guardar modelo y scaler
model.save('modelo_entrenado.h5')
joblib.dump(scaler, 'scaler.pkl')

print("Modelo entrenado y guardado.")
