from tensorflow.keras.models import load_model
import streamlit as st

label_to_text = {0:'raiva', 1:'nojo', 2:'medo', 3:'feliz', 4:'triste', 5:'surpreso', 6:'neutro'}

# Carregando o modelo treinado
checkpoint_path = 'checkpoint/best_model_mlp.h5'
final_model_mlp = load_model(checkpoint_path)

# Função para processar a imagem e fazer a previsão
def predict_emotion(image):
    # Pré-processamento da imagem
    resized_image = cv2.resize(image, (48, 48))
    normalized_image = resized_image.astype('float32') / 255.0
    preprocessed_frame = normalized_image.reshape(1, 48, 48, 1)
    
    # Fazer a previsão
    predicted_class = final_model_mlp.predict(preprocessed_frame).argmax()
    predicted_emotion = label_to_text[predicted_class]
    
    return predicted_emotion

import cv2

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
            cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        # Exibir o vídeo com as detecções de rosto e emoções
        cv2.imshow('Emotion Recognition', frame)
        
        # Pressione 'q' para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Libere o objeto VideoCapture e feche a janela
    cap.release()
    cv2.destroyAllWindows()

# Definir o título da página
st.title('Reconhecimento de Expressões Faciais')

# Adicionar um botão para ativar a câmera
if st.button('Ativar Câmera'):
    capture_video()

