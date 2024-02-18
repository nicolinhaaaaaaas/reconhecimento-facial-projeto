import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

# Treinamento do modelo
model_cnn.fit(X_train, y_train, epochs=30, validation_split=0.1)

# Função para processar a imagem e fazer a previsão
def predict_emotion(image):
    resized_image = cv2.resize(image, (48, 48))
    normalized_image = resized_image.astype('float32') / 255.0
    preprocessed_frame = normalized_image.reshape(1, 48, 48, 1)
    predicted_class = model_cnn.predict(preprocessed_frame).argmax()
    predicted_emotion = label_to_text[predicted_class]
    return predicted_emotion

# Função para capturar o vídeo da câmera
def capture_video():
    # Carregar o classificador de detecção de rosto
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Iniciar a captura de vídeo
    cap = cv2.VideoCapture(0)

    # Loop para capturar o vídeo
    while True:

        # Ler um frame do vídeo
        ret, frame = cap.read()

        # Verificar se o frame foi lido corretamente
        if not ret:
            break

        # Converter o frame para escala de cinza
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostos na imagem
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterar sobre os rostos detectados
        for (x, y, w, h) in faces:
            # Recortar a região do rosto
            face_roi = gray_frame[y:y+h, x:x+w]

            # Fazer a previsão da emoção
            predicted_emotion = predict_emotion(face_roi)

            # Desenhar um retângulo ao redor do rosto detectado
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Exibir a emoção detectada sobreposta ao retângulo
            cv2.putText(frame, str(predicted_emotion), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Exibir o vídeo com as detecções de rosto e emoções
        cv2.imshow('Emotion Recognition', frame)

        # Pressione 'q' para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Definir o título da página
st.title('Reconhecimento de Expressões Faciais')

# Adicionar um botão para ativar a câmera
if st.button('Ativar Câmera'):
    capture_video()

# Caixa de upload de arquivo
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

# Se o usuário fizer o upload de um arquivo
if uploaded_file is not None:
    # Carregar a imagem
    image = Image.open(uploaded_file)
    
    # Exibir a imagem
    st.image(image, caption='Imagem Carregada', use_column_width=True)
    
    # Botão para iniciar o reconhecimento facial
    if st.button('Reconhecer Expressão Facial'):
        # Realizar o reconhecimento facial
        predicted_emotion = predict_emotion(np.array(image))
        
        # Exibir o resultado do reconhecimento facial
        st.success(f'A expressão facial identificada é: {predicted_emotion}')
