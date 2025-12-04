# 🧠 Modelo de Análisis de Sentimientos con Deep Learning

Proyecto de Deep Learning para clasificación de sentimientos en comentarios en español utilizando TensorFlow/Keras.

## 📊 Características

- **Dataset**: 600 comentarios balanceados (300 positivos, 300 negativos)
- **Arquitectura**: Red neuronal profunda con Embedding + GlobalAveragePooling + capas Dense
- **Precisión**: ~78% en el conjunto de prueba
- **Framework**: TensorFlow/Keras

## 🏗️ Arquitectura del Modelo

```python
Sequential([
    TextVectorization (max_tokens=3000, sequence_length=50),
    Embedding (output_dim=32),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

## 📁 Estructura del Proyecto

```
DeepLearning/
├── data/
│   ├── comentarios.csv          # Dataset de entrenamiento
│   └── dataset.py               # Script para manejo de datos
├── src/
│   └── modelo_sentimientos_deeplearning.py  # Modelo principal
├── img/
│   └── entrenamiento.png        # Gráficos de entrenamiento
└── README.md
```

## 🚀 Instalación

### Prerrequisitos

- Python 3.12
- pip

### Configuración del Entorno

```bash
# Clonar el repositorio
git clone https://github.com/calossalfate/DeepLearning.git
cd DeepLearning

# Crear entorno virtual
py -3.12 -m venv .venv-tf

# Activar entorno virtual (Windows)
.\.venv-tf\Scripts\activate

# Instalar dependencias
pip install tensorflow scikit-learn pandas numpy matplotlib
```

## 💻 Uso

```bash
# Entrenar el modelo
python src/modelo_sentimientos_deeplearning.py
```

El script generará:
- `mejor_modelo_sentimientos.keras` - Mejor modelo durante entrenamiento
- `modelo_sentimientos_final.keras` - Modelo final
- `img/entrenamiento.png` - Gráficos de loss y accuracy

## 📈 Resultados

### Métricas de Rendimiento

```
              precision    recall  f1-score   support

           0       0.81      0.65      0.72        40
           1       0.77      0.89      0.82        53

    accuracy                           0.78        93
```

### Matriz de Confusión

```
[[26 14]    ← Negativos: 26 correctos, 14 falsos positivos
 [ 6 47]]   ← Positivos: 47 correctos, 6 falsos negativos
```

## 🔧 Tecnologías Utilizadas

- **TensorFlow/Keras**: Framework de Deep Learning
- **scikit-learn**: Métricas y división de datos
- **pandas**: Manipulación de datos
- **numpy**: Operaciones numéricas
- **matplotlib**: Visualización de resultados

## 📝 Dataset

El dataset contiene 600 comentarios en español sobre productos y servicios:
- 300 comentarios positivos (sentimiento = 1)
- 300 comentarios negativos (sentimiento = 0)

Cada comentario incluye expresiones naturales sobre:
- Calidad de productos
- Experiencias de compra
- Servicio al cliente
- Características técnicas

## 🎯 Características del Modelo

- **Vectorización de texto**: Max 3000 tokens, secuencias de 50 palabras
- **Embedding**: Representación densa de 32 dimensiones
- **Regularización**: Dropout (0.4 y 0.3) para prevenir overfitting
- **Early Stopping**: Patience de 5 épocas
- **Optimizador**: Adam con learning rate de 0.001

## 📊 Visualizaciones

El script genera gráficos que muestran:
- Evolución de la pérdida (loss) en entrenamiento y validación
- Evolución de la precisión (accuracy) en entrenamiento y validación

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es de código abierto y está disponible para uso educativo.

## 👤 Autor

**Carlos** - [calossalfate](https://github.com/calossalfate)

## 🙏 Agradecimientos

- Proyecto desarrollado con fines educativos
- Dataset creado para entrenamiento de modelos de NLP
- Basado en arquitecturas modernas de Deep Learning

---

⭐️ Si este proyecto te fue útil, considera darle una estrella en GitHub!

