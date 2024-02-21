import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time

# Carregar o modelo treinado
checkpoint_path = 'checkpoint/best_model_mlp.h5'
final_model_mlp = load_model(checkpoint_path)

label_to_text = {0: 'raiva', 1: 'nojo', 2: 'medo', 3: 'feliz', 4: 'triste', 5: 'surpreso', 6: 'neutro'}

# Função para processar o frame de vídeo e fazer a previsão da emoção
def predict_emotion(frame):
    # Redimensionar o frame para o tamanho esperado pelo modelo
    resized_frame = cv2.resize(frame, (48, 48))
    
    # Converter o frame para escala de cinza
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    
    # Normalizar os pixels da imagem
    normalized_frame = gray_frame.astype('float32') / 255.0
    
    # Pré-processar o frame para o modelo
    preprocessed_frame = normalized_frame.reshape(1, 48, 48, 1)
    
    # Fazer a previsão
    predicted_class = final_model_mlp.predict(preprocessed_frame).argmax()
    predicted_emotion = label_to_text[predicted_class]
    
    return predicted_emotion

# Definir a classe do transformador de vídeo
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        # Inicializar o modelo de detecção de rostos
        self.face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
        
    def transform(self, frame):
        # Converter o frame para um array numpy
        frame_bgr = np.array(frame.to_image())

        # Converter de BGR para RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Detectar rostos no frame
        results = self.face_detection.process(frame_rgb)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame_rgb.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)

                # Recortar a região do rosto
                face_roi = frame_rgb[y:y+h, x:x+w]
                
                # Fazer a previsão da emoção
                predicted_emotion = predict_emotion(face_roi)
                
                # Desenhar a emoção detectada no frame
                frame_rgb = cv2.putText(frame_rgb, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                print(predicted_emotion)
        
        return frame_rgb

# Definir o título da página
st.title('Reconhecimento de Expressões Faciais')

webrtc_streamer(key="emotion-recognition-1", video_processor_factory=VideoTransformer)