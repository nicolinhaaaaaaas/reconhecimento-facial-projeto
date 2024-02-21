import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import streamlit as st
import cv2

# Carregar o conjunto de dados
df = pd.read_csv('fer2013.csv')  # Substitua pelo caminho do seu conjunto de dados

# Pré-processamento do conjunto de dados
img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
img_array = np.stack(img_array, axis=0) / 255.0

label_to_text = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4:'sadness', 5:'surprise', 6:'neutral'}

# Divisão em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(img_array, df.emotion.values, test_size=0.2, random_state=42)

# Definindo o modelo CNN
model_cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])

# Compilando o modelo
model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Configurando o callback para salvar o melhor modelo durante o treinamento
checkpoint_path = 'checkpoint/best_model_mlp.h5'
call_back = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                               # Parâmetro monitor: qual métrica será monitorada
                                                monitor='val_accuracy',
                                                # Verbose: controla a quantidade de informações impressas durante o treinamento
                                                verbose=1,
                                                # Salva o modelo a cada época
                                                save_freq='epoch',
                                                # Parâmetro save_best_only: se True, o modelo é salvo quando a métrica monitorada é maximizada
                                                save_best_only=True,
                                                # determina se apenas os pesos do modelo devem ser salvos ou o modelo completo, FALSE significa completo
                                                save_weights_only=False,
                                                # Parâmetro mode: indica se a métrica monitorada deve ser maximizada ou minimizada
                                                mode='max')

# Treinamento do modelo
model_cnn.fit(X_train, y_train, epochs=20, validation_split=0.1, callbacks=[call_back])

# Avaliação do modelo no conjunto de teste
_, test_accuracy = model_cnn.evaluate(X_test, y_test)
print(f'Acurácia no conjunto de teste: {test_accuracy * 100:.2f}%')

# Carregar o modelo treinado
final_model_mlp = load_model(checkpoint_path)

# Salvar o modelo treinado
final_model_mlp.save('checkpoint/best_model_mlp.h5')

# Função para realizar o reconhecimento facial
def recognize_facial_expression(image):
    
    # Pré-processar a imagem para o modelo de emoção
    preprocessed_image = preprocess_image(image)
    
    # Aplicar o modelo para obter as previsões de emoção
    predicted_class = final_model_mlp.predict(preprocessed_image).argmax()
    predicted_emotion = label_to_text[predicted_class]
    
    return predicted_emotion

def preprocess_image(image):
    # Converter a imagem para escala de cinza e redimensionar para o tamanho esperado pelo modelo
    img = image.convert('L').resize((48, 48))
    img_array = np.array(img).reshape(1, 48, 48, 1).astype('float32') / 255.0
    return img_array

# Pré-processamento da imagem para o modelo
def preprocess_image(image):
    img = Image.open(image).convert('L').resize((48, 48))
    img_array = np.array(img).reshape(1, 48, 48, 1).astype('float32') / 255.0
    return img_array

