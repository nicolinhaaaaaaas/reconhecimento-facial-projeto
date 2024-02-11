import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import time
import tensorflow as tf
from matplotlib import pyplot

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

# Definindo o modelo MLP
#Isso cria um modelo sequencial, que é uma pilha linear de camadas. Os dados fluem sequencialmente através das camadas
model_mlp = tf.keras.models.Sequential([

    #Converte os dados de entrada em um vetor 1D, nesse caso de 48x48 pixels para em escala de cinza (1)
    tf.keras.layers.Flatten(input_shape=(48, 48, 1)),
    #Camada densa com 128 neurônios e função de ativação relu
    tf.keras.layers.Dense(128, activation='relu'),
    #Camada de dropout para evitar overfitting
    tf.keras.layers.Dropout(0.5),
    #Camada densa com 64 neurônios e função de ativação relu
    tf.keras.layers.Dense(64, activation='relu'),
    #Camada de dropout para evitar overfitting
    tf.keras.layers.Dropout(0.5),
    #Camada densa com 32 neurônios e função de ativação relu
    tf.keras.layers.Dense(32, activation='relu'),
    #Camada de dropout para evitar overfitting
    tf.keras.layers.Dropout(0.5),
    #Camada densa com 7 neurônios e função de ativação softmax, que serve para converter a saidas em probabilidades
    tf.keras.layers.Dense(7, activation='softmax')
])

# Compilando o modelo
model_mlp.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
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
model_mlp.fit(X_train, y_train, epochs=20, validation_split=0.1, callbacks=[call_back])

# Avaliação do modelo no conjunto de teste
_, test_accuracy = model_mlp.evaluate(X_test, y_test)
print(f'Acurácia no conjunto de teste: {test_accuracy * 100:.2f}%')

# Carregar o modelo treinado
final_model_mlp = load_model(checkpoint_path)

# Mapeamento das labels para texto
label_to_text = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

# Pré-processamento da imagem para o modelo
def preprocess_image(image):
    img = Image.open(image).convert('L').resize((48, 48))
    img_array = np.array(img).reshape(1, 48, 48, 1).astype('float32') / 255.0
    return img_array

# Carregando o modelo treinado
final_model_mlp = load_model(checkpoint_path)

import cv2

# Função para capturar o vídeo da câmera
def capture_video():
    # Carregar o classificador de detecção de rosto
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)  # 0 para a câmera padrão
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Conversão para escala de cinza para detecção de rosto
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostos na imagem
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Iterar sobre os rostos detectados
        for (x, y, w, h) in faces:
            # Recortar a região do rosto
            face_roi = gray_frame[y:y+h, x:x+w]
            
            # Redimensionar a imagem facial para o tamanho esperado pelo modelo
            resized_face = cv2.resize(face_roi, (48, 48))
            
            # Converter a imagem facial para escala de cinza
            gray_face = cv2.cvtColor(resized_face, cv2.COLOR_GRAY2BGR)
            
            # Normalizar os pixels da imagem facial
            normalized_face = gray_face.astype('float32') / 255.0
            
            # Pré-processamento da imagem para o modelo de emoção
            preprocessed_frame = normalized_face.reshape(48, 48)
            
            # Aplicação do modelo para obter as previsões de emoção
            predicted_class = final_model_mlp.predict(preprocessed_frame).argmax()
            predicted_emotion = label_to_text[predicted_class]
            
            # Desenhar um retângulo ao redor do rosto detectado
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Exibir a emoção detectada sobreposta ao retângulo
            cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        # Exibir o vídeo com as detecções de rosto e emoções
        cv2.imshow('Emotion Recognition', frame)
        
        # Pressione 'q' para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Libere o objeto VideoCapture e feche a janela
    cap.release()
    cv2.destroyAllWindows()

# Chamada da função para capturar o vídeo da câmera
capture_video()