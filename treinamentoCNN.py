import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Carregar o conjunto de dados
df = pd.read_csv('fer2013.csv')

# Pré-processamento do conjunto de dados
img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
img_array = np.stack(img_array, axis=0) / 255.0

# Divisão em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(img_array, df.emotion.values, test_size=0.2, random_state=42)

# Definindo o modelo CNN
model_cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),  # Camada de convolução com 64 filtros
    tf.keras.layers.MaxPooling2D((2, 2)),  # Camada de pooling para redução de dimensionalidade
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),  # Outra camada de convolução com 128 filtros
    tf.keras.layers.MaxPooling2D((2, 2)),  # Outra camada de pooling
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),  # Outra camada de convolução com 256 filtros
    tf.keras.layers.MaxPooling2D((2, 2)),  # Outra camada de pooling
    tf.keras.layers.Flatten(),  # Camada de achatamento para conectar à camada densa
    tf.keras.layers.Dense(256, activation='relu'),  # Camada densa com 256 unidades
    tf.keras.layers.Dropout(0.3),  # Dropout para regularização
    tf.keras.layers.Dense(128, activation='relu'),  # Outra camada densa com 128 unidades
    tf.keras.layers.Dropout(0.3),  # Dropout adicional
    tf.keras.layers.Dense(7, activation='softmax')  # Camada de saída com 7 unidades para classificação multiclasse
])

# Compilando o modelo
model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Configurando o callback para salvar o melhor modelo durante o treinamento
checkpoint_path = 'checkpointCNN/best_model_mlp.h5'
call_back = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                               monitor='val_accuracy',
                                               verbose=1,
                                               save_freq='epoch',
                                               save_best_only=True,
                                               save_weights_only=False,
                                               mode='max')

# Treinamento do modelo
model_cnn.fit(X_train, y_train, epochs=20, validation_split=0.1, callbacks=[call_back])

# Avaliação do modelo no conjunto de teste
_, test_accuracy = model_cnn.evaluate(X_test, y_test)
print(f'Acurácia no conjunto de teste: {test_accuracy * 100:.2f}%')

# Carregar o modelo treinado
final_model_mlp = tf.keras.models.load_model(checkpoint_path)

# Salvar o modelo treinado
final_model_mlp.save('checkpointCNN/best_model_mlp.h5')