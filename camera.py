import cv2
import mediapipe as mp
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

label_to_text = {0:'raiva', 1:'nojo', 2:'medo', 3:'feliz', 4:'triste', 5:'surpreso', 6:'neutro'}

# Carregando o modelo treinado
checkpoint_path = 'checkpoint/best_model_mlp.h5'
final_model_mlp = load_model(checkpoint_path)

# Função para detectar emoção em uma região facial
def predict_emotion(face_roi):
    # Redimensionar a região facial para o tamanho esperado pelo modelo
    resized_face = cv2.resize(face_roi, (48, 48))
    
    # Converter a imagem para escala de cinza
    gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
    
    # Normalizar os pixels da imagem
    normalized_face = gray_face.astype('float32') / 255.0
    
    # Pré-processar a imagem para o modelo
    preprocessed_face = normalized_face.reshape(1, 48, 48, 1)
    
    # Fazer a previsão
    predicted_class = final_model_mlp.predict(preprocessed_face).argmax()
    predicted_emotion = label_to_text[predicted_class]
    
    return predicted_emotion

# Iniciar o mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Função para capturar o vídeo da câmera e realizar a detecção de rostos e emoções
def capture_video():
    # Iniciar a captura de vídeo
    cap = cv2.VideoCapture(0)
    
    # Loop para capturar o vídeo
    while cap.isOpened():
        # Ler um frame do vídeo
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converter o frame de BGR para RGB (mediapipe usa RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar rostos no frame usando mediapipe
        results = face_detection.process(rgb_frame)
        
        # Verificar se algum rosto foi detectado
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Recortar a região do rosto
                face_roi = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
                
                # Fazer a previsão da emoção na região do rosto
                predicted_emotion = predict_emotion(face_roi)
                
                # Desenhar um retângulo ao redor do rosto detectado
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
                
                # Exibir a emoção detectada sobreposta ao retângulo
                cv2.putText(frame, predicted_emotion, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        # Exibir o vídeo com as detecções de rosto e emoções
        cv2.imshow('Emotion Recognition', frame)
        
        # Pressione 'q' para sair do loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    # Libere o objeto VideoCapture e feche a janela
    cap.release()
    cv2.destroyAllWindows()

# Definir o título da página
st.title('Reconhecimento de Expressões Faciais')

# Adicionar um botão para ativar a câmera
if st.button('Ativar Câmera'):
    capture_video()