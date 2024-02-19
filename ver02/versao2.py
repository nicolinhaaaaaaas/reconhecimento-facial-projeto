import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import tensorflow as tf
from matplotlib import pyplot
import streamlit as st

# Carregar o conjunto de dados
df = pd.read_csv('fer2013.csv')  # Substitua pelo caminho do seu conjunto de dados
df.head()

# Pré-processamento do conjunto de dados
img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
img_array = np.stack(img_array, axis=0) / 255.0

labels = df.emotion.values

pyplot.imshow(np.array(df.pixels.loc[0].split(' ')).reshape(48, 48).astype('float32'))

label_to_text = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4:'sadness', 5:'surprise', 6:'neutral'}

fig = pyplot.figure(1, (14, 14))
k = 0
for label in sorted(df.emotion.unique()):
  for j in range(3):
    px = df[df.emotion==label].pixels.iloc[k]
    px = np.array(px.split(' ')).reshape(48, 48).astype('float32')
    k += 1
    ax = pyplot.subplot(7, 7, k)
    ax.imshow(px)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label_to_text[label])
    pyplot.tight_layout()

# Divisão em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(img_array, labels, test_size=0.2, random_state=42)

print("Conjunto de treinamento (X_train):")
print(X_train)

print("\nConjunto de teste (X_test):")
print(X_test)

print("\nLabels do conjunto de treinamento (y_train):")
print(y_train)

print("\nLabels do conjunto de teste (y_test):")
print(y_test)

# Definindo o modelo CNN
model_cnn = tf.keras.models.Sequential([
    # Camada convolucional com 32 filtros de tamanho 3x3, função de ativação relu
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    # Camada de pooling para redução de dimensionalidade
    tf.keras.layers.MaxPooling2D((2, 2)),
    # Camada convolucional com 64 filtros de tamanho 3x3, função de ativação relu
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # Camada de pooling para redução de dimensionalidade
    tf.keras.layers.MaxPooling2D((2, 2)),
    # Camada convolucional com 128 filtros de tamanho 3x3, função de ativação relu
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    # Camada de pooling para redução de dimensionalidade
    tf.keras.layers.MaxPooling2D((2, 2)),
    # Achatamento para conectar com a camada densa
    tf.keras.layers.Flatten(),
    # Camada densa com 128 neurônios e função de ativação relu
    tf.keras.layers.Dense(128, activation='relu'),
    # Camada de dropout para evitar overfitting
    tf.keras.layers.Dropout(0.5),
    # Camada densa com 64 neurônios e função de ativação relu
    tf.keras.layers.Dense(64, activation='relu'),
    # Camada de dropout para evitar overfitting
    tf.keras.layers.Dropout(0.5),
    # Camada densa com 32 neurônios e função de ativação relu
    tf.keras.layers.Dense(32, activation='relu'),
    # Camada de dropout para evitar overfitting
    tf.keras.layers.Dropout(0.5),
    # Camada densa com 7 neurônios e função de ativação softmax, que serve para converter a saída em probabilidades
    tf.keras.layers.Dense(7, activation='softmax')
])

# Compilando o modelo
model_cnn.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

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

