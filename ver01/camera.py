import cv2
from tensorflow.keras.models import load_model

# Carregando o modelo treinado
checkpoint_path = 'checkpoint/best_model_mlp.h5'
final_model_mlp = load_model(checkpoint_path)

label_to_text = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4:'sadness', 5:'surprise', 6:'neutral'}

# Função para processar a imagem e fazer a previsão
def predict_emotion(image):
    # Redimensionar a imagem para o tamanho esperado pelo modelo
    resized_image = cv2.resize(image, (48, 48))
    # Normalizar os pixels da imagem
    normalized_image = resized_image.astype('float32') / 255.0
    # Adicionar uma dimensão extra para a imagem (formato esperado pelo modelo)
    preprocessed_frame = normalized_image.reshape(1, 48, 48, 1)
    # Fazer a previsão usando o modelo final_model_mlp
    predicted_class = final_model_mlp.predict(preprocessed_frame).argmax()
    # Obter a emoção correspondente ao índice previsto
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

# Chamada da função para capturar o vídeo da câmera
capture_video()