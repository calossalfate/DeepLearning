import sys
import io
# Configurar encoding para Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Cargar dataset
df = pd.read_csv("data/comentarios.csv", encoding='latin-1')

X = df["comentario"].astype(str).values
y = df["sentimiento"].values

# 2. División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Vectorización (reducida para dataset pequeño)
vectorizer = TextVectorization(max_tokens=3000, output_sequence_length=50)
vectorizer.adapt(X_train)

# 4. Modelo Deep Learning (simplificado pero sigue siendo deep)
model = Sequential([
    vectorizer,
    Embedding(input_dim=3000, output_dim=32),
    GlobalAveragePooling1D(),
    Dense(64, activation="relu"),
    Dropout(0.4),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Optimizer con learning rate ajustado
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 5. Callbacks
early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
check = ModelCheckpoint("mejor_modelo_sentimientos.keras", save_best_only=True)

# 6. Entrenar (ajustado para mejor aprendizaje)
print("Iniciando entrenamiento...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=16,
    callbacks=[early, check],
    verbose=0  # Sin output para evitar problemas de encoding
)
print("Entrenamiento completado!")

# 7. Evaluar
pred = (model.predict(X_test) > 0.5).astype(int)

print("\n=== REPORTE DE CLASIFICACION ===")
print(classification_report(y_test, pred, target_names=['Negativo', 'Positivo']))

cm = confusion_matrix(y_test, pred)
print("\n=== MATRIZ DE CONFUSION ===")
print(cm)
print("\nInterpretacion:")
print(f"Verdaderos Negativos: {cm[0][0]}")
print(f"Falsos Positivos: {cm[0][1]}")
print(f"Falsos Negativos: {cm[1][0]}")
print(f"Verdaderos Positivos: {cm[1][1]}")

# 8. Gráficos de entrenamiento y matriz de confusión
fig = plt.figure(figsize=(15, 5))

# Gráfico 1: Loss
plt.subplot(1, 3, 1)
plt.plot(history.history["loss"], label="loss", marker='o')
plt.plot(history.history["val_loss"], label="val_loss", marker='s')
plt.legend()
plt.title("Evolución de Pérdida")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)

# Gráfico 2: Accuracy
plt.subplot(1, 3, 2)
plt.plot(history.history["accuracy"], label="accuracy", marker='o')
plt.plot(history.history["val_accuracy"], label="val_accuracy", marker='s')
plt.legend()
plt.title("Evolución de Precisión")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)

# Gráfico 3: Matriz de Confusión
plt.subplot(1, 3, 3)
im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.colorbar(im)
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Negativo', 'Positivo'])
plt.yticks(tick_marks, ['Negativo', 'Positivo'])

# Agregar números en cada celda
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=20, fontweight='bold')

plt.ylabel('Valor Real')
plt.xlabel('Predicción')

plt.tight_layout()
plt.savefig("img/entrenamiento.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ Graficos guardados en 'img/entrenamiento.png'")

model.save("modelo_sentimientos_final.keras")
