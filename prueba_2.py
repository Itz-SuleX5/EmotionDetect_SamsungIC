import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import os
import threading
from collections import deque

# Definir las clases de emociones 
CLASES_EMOCIONES = ['amusement', 'excitement', 'anger', 'fear', 'sadness', 'contentment', 'awe', 'disgust'] # Estas son para el modelo
EMOCIONES_ESPAÑOL = ['diversion', 'emocion', 'enojo', 'miedo', 'tristeza', 'satisfaccion', 'asombro', 'disgusto'] # Y estas son para mostrarla

# Definir colores para cada emoción (formato BGR)
COLORES_EMOCIONES = {
    'diversión': (0, 255, 0),      # Verde
    'emoción': (0, 215, 255),      # Amarillo
    'enojo': (0, 0, 255),          # Rojo
    'miedo': (255, 0, 0),          # Azul
    'tristeza': (130, 0, 75),      # Púrpura
    'satisfacción': (0, 255, 128), # Verde claro
    'asombro': (255, 255, 0),      # Cian
    'disgusto': (0, 0, 128)        # Rojo oscuro
}

# Cargar el clasificador de rostros de OpenCV
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise Exception("No se pudo cargar el clasificador de rostros")
except Exception as e:
    print(f"Error al cargar el clasificador: {e}")
    face_cascade = None

# Cargar el modelo entrenado
def cargar_modelo(ruta_modelo='modelo_emociones_001.h5'):
    """
    1.Carga el modelo de detección de emociones desde la ruta especificada
    2.Verifica que el archivo exista antes de intentar cargarlo
    """
    try:
        # Verificar si el archivo existe
        if not os.path.exists(ruta_modelo):
            print(f"El archivo del modelo no existe en la ruta: {ruta_modelo}")
            return None
            
        # Configurar TensorFlow para usar solo la memoria necesaria
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Error al configurar GPU: {e}")
        
        # Cargar el modelo
        modelo = load_model(ruta_modelo)
        print(f"Modelo cargado correctamente desde {ruta_modelo}")
        
        # Realizar una predicción de prueba para verificar que funciona
        dummy_data = np.zeros((1, 48, 48, 1))  # Tamaño de prueba genérico
        try:
            _ = modelo.predict(dummy_data, verbose=0)
            print("Prueba de predicción exitosa")
        except Exception as e:
            print(f"Advertencia: La prueba de predicción falló: {e}")
            
        return modelo
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

# Función para obtener el tamaño de entrada del modelo
def obtener_tamaño_modelo(modelo):
    """
    Obtiene el tamaño de entrada que espera el modelo
    """
    try:
        # Obtener la forma de entrada del modelo directamente del modelo
        input_shape = modelo.input_shape
        
        # Extraer las dimensiones de altura y anchura (ignorando el batch y canales)
        if input_shape is not None and len(input_shape) >= 3:
            altura = input_shape[1]
            anchura = input_shape[2]
            print(f"Tamaño de entrada detectado: {altura}x{anchura}")
            return (altura, anchura)
        else:
            # Si no se puede determinar, usar un valor predeterminado
            print("No se pudo determinar el tamaño de entrada del modelo, usando 48x48 como predeterminado")
            return (48, 48)
    except Exception as e:
        print(f"Error al obtener tamaño del modelo: {e}")
        return (48, 48)  # Valor predeterminado

# Función para preprocesar la imagen del rostro
def preprocesar_imagen(imagen, tamaño):
    """
    1.Preprocesa la imagen para que sea compatible con el modelo
    2.Convierte a escala de grises, redimensiona y normaliza
    """
    # Verificar que la imagen no esté vacía
    if imagen is None or imagen.size == 0:
        print("Error: Imagen vacía en preprocesar_imagen")
        return None
    
    try:
        # Convertir la imagen de RGB a escala de grises
        if len(imagen.shape) == 3:  # Si la imagen tiene 3 canales (RGB)
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
        else:
            imagen_gris = imagen
            
        # Redimensionar la imagen al tamaño esperado por el modelo
        imagen_redimensionada = cv2.resize(imagen_gris, (tamaño[1], tamaño[0]))
        
        # Normalizar los valores de píxeles (de 0 a 1)
        imagen_normalizada = imagen_redimensionada / 255.0
        
        # Asegurarse de que la imagen tenga la forma correcta para el modelo
        imagen_con_canal = np.expand_dims(imagen_normalizada, axis=-1)  # Añadir la dimensión del canal
        
        # Añadir la dimensión del batch para la predicción
        imagen_con_batch = np.expand_dims(imagen_con_canal, axis=0)
        
        return imagen_con_batch
    except Exception as e:
        print(f"Error en preprocesar_imagen: {e}")
        return None

# Clase para suavizar las predicciones de emociones
class SmoothEmotion:
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        self.emotion_buffer = {}
        
    def update(self, face_id, emotion_idx, probability):
        """
        Actualiza el buffer de emociones para una cara específica y devuelve
        la emoción suavizada
        """
        if face_id not in self.emotion_buffer:
            self.emotion_buffer[face_id] = deque(maxlen=self.buffer_size)
            
        self.emotion_buffer[face_id].append((emotion_idx, probability))
        
        # Calcular la emoción más frecuente en el buffer
        emotion_counts = {}
        total_prob = {}
        
        for e_idx, prob in self.emotion_buffer[face_id]:
            if e_idx not in emotion_counts:
                emotion_counts[e_idx] = 0
                total_prob[e_idx] = 0
            emotion_counts[e_idx] += 1
            total_prob[e_idx] += prob
            
        # Encontrar la emoción más frecuente
        max_count = 0
        smoothed_emotion = 0
        smoothed_prob = 0
        
        for e_idx, count in emotion_counts.items():
            if count > max_count:
                max_count = count
                smoothed_emotion = e_idx
                smoothed_prob = total_prob[e_idx] / count
                
        return smoothed_emotion, smoothed_prob
        
    def clear_old_faces(self, current_faces):
        """
        Elimina caras que ya no están presentes en el frame
        """
        faces_to_remove = []
        for face_id in self.emotion_buffer:
            if face_id not in current_faces:
                faces_to_remove.append(face_id)
                
        for face_id in faces_to_remove:
            del self.emotion_buffer[face_id]

# Función para convertir probabilidad a puntos (1-10)
def probabilidad_a_puntos(probabilidad):
    """
    Convierte una probabilidad (0-1) a una escala de puntos (1-10)
    """
    # Asegurar que la probabilidad esté en el rango [0, 1]
    prob_clipped = max(0, min(1, probabilidad))
    # Convertir a puntos del 1 al 10
    return round(prob_clipped * 9) + 1  # Multiplicar por 9 y sumar 1 para obtener rango de 1-10

# Función para mostrar la emoción y sus puntos
def mostrar_emocion(frame, emocion_idx, probabilidad, x, y, w, h):
    """
    Muestra la emoción en español y sus puntos encima de la cabeza de la persona
    """
    # Obtener la emoción en español
    emocion_esp = EMOCIONES_ESPAÑOL[emocion_idx]
    
    # Convertir probabilidad a puntos (1-10)
    puntos = probabilidad_a_puntos(probabilidad)
    
    # Texto de la emoción
    texto = f"{emocion_esp}: {puntos} pts"

    # Obtener el color específico para esta emoción
    color = COLORES_EMOCIONES.get(emocion_esp, (255, 255, 255))  # Blanco como color predeterminado

    # Calcular el ancho del texto para centrar
    (text_width, text_height), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_x = x + (w - text_width) // 2  # Centrar el texto sobre la cara
    
    # Colocar el texto encima de la cabeza 
    text_y = max(30, y - 15)
    
    # Dibujar un fondo semitransparente para el texto
    overlay = frame.copy()
    cv2.rectangle(overlay, (text_x - 5, text_y - text_height - 5), 
                 (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Dibujar el texto
    cv2.putText(frame, texto, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Dibujar el rectángulo alrededor del rostro con el color específico de la emoción
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

# Clase para ejecutar la predicción en un hilo separado
class EmotionPredictor:
    def __init__(self, modelo, tamaño_modelo):
        self.modelo = modelo
        self.tamaño_modelo = tamaño_modelo
        self.queue = []
        self.results = {}
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._predict_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def add_face(self, face_id, roi):
        with self.lock:
            self.queue.append((face_id, roi))
    
    def get_result(self, face_id):
        with self.lock:
            return self.results.get(face_id, (None, 0))
    
    def _predict_loop(self):
        while self.running:
            face_to_process = None
            with self.lock:
                if self.queue:
                    face_to_process = self.queue.pop(0)
            
            if face_to_process:
                face_id, roi = face_to_process
                
                # Preprocesar la imagen
                roi_procesada = preprocesar_imagen(roi, tamaño=self.tamaño_modelo)
                
                if roi_procesada is not None:
                    try:
                        # Realizar predicción
                        prediccion = self.modelo.predict(roi_procesada, verbose=0)
                        clase_predicha = np.argmax(prediccion)
                        probabilidad = prediccion[0][clase_predicha]
                        
                        with self.lock:
                            self.results[face_id] = (clase_predicha, probabilidad)
                    except Exception as e:
                        print(f"Error al realizar predicción: {e}")
            
            time.sleep(0.01)  # Pequeña pausa para no saturar la CPU
            
    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

# Función principal para ejecutar el reconocimiento de emociones
def reconocer_emociones_camara(modelo, umbral_confianza=0.5):
    """
    Ejecuta el reconocimiento de emociones utilizando la cámara
    """
    if face_cascade is None:
        print("Error: No se pudo cargar el clasificador de rostros. Saliendo...")
        return
    
    # Obtener el tamaño de entrada que espera el modelo
    tamaño_modelo = obtener_tamaño_modelo(modelo)
    
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)
    
    # Ajustar la resolución para mejorar rendimiento si es necesario
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Verificar si la cámara se abrió correctamente
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    print("Presiona 'q' para salir, 'r' para reiniciar detección, 'p' para pausar/continuar.")

    # Inicializar suavizador de emociones
    smoother = SmoothEmotion(buffer_size=8)
    
    # Inicializar predictor en hilo separado
    predictor = EmotionPredictor(modelo, tamaño_modelo)

    # Variables para cálculo de FPS
    fps_contador = 0
    fps_inicio = time.time()
    fps = 0
    
    # Variable para pausar la detección
    paused = False
    
    # Crear ventana con tamaño ajustable
    cv2.namedWindow('Detector de Emociones', cv2.WINDOW_NORMAL)

    while True:
        # Capturar frame de la cámara
        ret, frame = cap.read()

        if not ret:
            print("Error: No se pudo capturar el frame.")
            break
            
        # Si se pausa la detección, mostrar mensaje
        if paused:
            cv2.putText(frame, "PAUSA - Presiona 'p' para continuar", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Mostrar el frame resultante
            cv2.imshow('Detector de Emociones', frame)
            
            # Procesar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = False
                
            continue

        # Voltear el frame horizontalmente para efecto espejo natural
        frame = cv2.flip(frame, 1)

        # Convertir a escala de grises para detección de rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Crear mapa de caras actuales
        current_faces = {}

        # Procesamiento de cada rostro detectado
        for i, (x, y, w, h) in enumerate(faces):
            face_id = f"face_{i}"
            current_faces[face_id] = (x, y, w, h)
            
            # Extraer la región de interés (ROI)
            roi = frame[y:y+h, x:x+w]

            # Verificar si el ROI es válido
            if roi.size == 0:
                continue
                
            # Añadir cara para predicción en hilo separado
            predictor.add_face(face_id, roi)
            
            # Obtener resultado de la predicción
            clase_predicha, probabilidad = predictor.get_result(face_id)
            
            if clase_predicha is not None:
                # Suavizar la emoción para evitar parpadeos
                clase_suavizada, prob_suavizada = smoother.update(face_id, clase_predicha, probabilidad)
                
                # Mostrar la emoción solo si la confianza supera el umbral
                if prob_suavizada > umbral_confianza:
                    mostrar_emocion(frame, clase_suavizada, prob_suavizada, x, y, w, h)
                else:
                    # Si la confianza es baja, mostrar como "Desconocido"
                    texto = "Desconocido"
                    cv2.putText(frame, texto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Limpiar caras que ya no están presentes
        smoother.clear_old_faces(current_faces)

        # Calcular y mostrar FPS
        fps_contador += 1
        if time.time() - fps_inicio > 1:
            fps = fps_contador
            fps_contador = 0
            fps_inicio = time.time()

        # Mostrar FPS en la pantalla
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Añadir instrucciones en la parte inferior
        cv2.putText(frame, "q: Salir | r: Reiniciar | p: Pausar", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Mostrar el frame resultante
        cv2.imshow('Detector de Emociones', frame)

        # Procesar teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reiniciar el suavizador de emociones
            smoother = SmoothEmotion(buffer_size=8)
            print("Detección reiniciada")
        elif key == ord('p'):
            paused = True
            print("Detección pausada")

    # Detener el predictor
    predictor.stop()
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

# Función principal
def main():
    print("=== Detector de Emociones ===")
    print("Iniciando el sistema...")
    
    # Cargar el modelo
    modelo = cargar_modelo()

    if modelo is None:
        print("ERROR: No se pudo cargar el modelo. Asegúrate de que el archivo modelo_emociones_001.h5 existe.")
        print("Ingresa la ruta al archivo del modelo o presiona Enter para salir: ")
        ruta_alternativa = input().strip()
        
        if ruta_alternativa:
            modelo = cargar_modelo(ruta_alternativa)
            
    if modelo is None:
        print("No se pudo cargar el modelo. Saliendo...")
        return
        
    print("Iniciando la cámara...")

    reconocer_emociones_camara(modelo)
    
    print("Programa finalizado.")

# Ejecución principal
if __name__ == "__main__":
    main()