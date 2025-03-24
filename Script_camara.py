import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import time

# Definir las clases de emociones (inglés y español)
CLASES_EMOCIONES = ['amusement', 'excitement', 'anger', 'fear', 'sadness', 'contentment', 'awe', 'disgust'] # Esto es para que funcione el modelo
EMOCIONES_ESPAÑOL = ['diversión', 'emocion', 'enojo', 'miedo', 'tristeza', 'satisfaccion', 'asombro', 'disgusto'] # Y estas solo para mostrarlas 

# Cargar el clasificador de rostros de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar el modelo entrenado
def cargar_modelo(ruta_modelo='mejor_modelo_emociones.h5'):
    try:
        modelo = load_model(ruta_modelo)
        print(f"Modelo cargado correctamente desde {ruta_modelo}")
        return modelo
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

# Función para obtener el tamaño de entrada del modelo
def obtener_tamaño_modelo(modelo):
    """
    Obtiene el tamaño de entrada que espera el modelo.
    """
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
        print("No se pudo determinar el tamaño de entrada del modelo, usando 224x224 como predeterminado")
        return (224, 224)

# Función para preprocesar la imagen del rostro
def preprocesar_imagen(imagen, tamaño):
    """
    Preprocesa la imagen para que sea compatible con el modelo.
    Convierte a escala de grises, redimensiona y normaliza.
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
        imagen_redimensionada = cv2.resize(imagen_gris, tamaño)
        
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

# Función para convertir probabilidad a puntos (1-10)
def probabilidad_a_puntos(probabilidad):
    """
    Convierte una probabilidad (0-1) a una escala de puntos (1-10)
    """
    return int(probabilidad * 10) + 1 if probabilidad > 0 else 1

# Función para mostrar la emoción y sus puntos
def mostrar_emocion(frame, emocion_idx, probabilidad, x, y, w, h):
    """
    Muestra la emoción en español y sus puntos encima de la cabeza de la persona.
    """
    # Obtener la emoción en español
    emocion_esp = EMOCIONES_ESPAÑOL[emocion_idx]
    
    # Convertir probabilidad a puntos (1-10)
    puntos = probabilidad_a_puntos(probabilidad)
    
    # Texto de la emoción
    texto = f"{emocion_esp}: {puntos} pts"

    # Determinar el color según la emoción
    if CLASES_EMOCIONES[emocion_idx] in ['amusement', 'contentment', 'awe', 'excitement']:
        color = (0, 255, 0)  # Verde para emociones positivas
    elif CLASES_EMOCIONES[emocion_idx] in ['anger', 'fear', 'sadness', 'disgust']:
        color = (0, 0, 255)  # Rojo para emociones negativas
    else:
        color = (255, 255, 0)  # Azul para neutral

    # Dibujar el rectángulo alrededor del rostro
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Colocar el texto encima de la cabeza (10 píxeles arriba del rectángulo del rostro)
    posicion_texto_y = y - 20 if y - 20 > 20 else 20
    cv2.putText(frame, texto, (x, posicion_texto_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Función principal para ejecutar el reconocimiento de emociones
def reconocer_emociones_camara(modelo, umbral_confianza=0.5):
    """
    Ejecuta el reconocimiento de emociones utilizando la cámara web.
    """
    # Obtener el tamaño de entrada que espera el modelo
    tamaño_modelo = obtener_tamaño_modelo(modelo)
    
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    # Verificar si la cámara se abrió correctamente
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    print("Presiona 'q' para salir.")

    # Contadores para calcular FPS
    fps_contador = 0
    fps_inicio = time.time()
    fps = 0

    while True:
        # Capturar frame de la cámara
        ret, frame = cap.read()

        if not ret:
            print("Error: No se pudo capturar el frame.")
            break

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

        # Procesamiento de cada rostro detectado
        for (x, y, w, h) in faces:
            # Extraer la región de interés (ROI)
            roi = frame[y:y+h, x:x+w]

            # Verificar si el ROI es válido
            if roi.size == 0:
                continue

            # Preprocesar la imagen
            roi_procesada = preprocesar_imagen(roi, tamaño=tamaño_modelo)
            
            if roi_procesada is None:
                continue

            # Realizar predicción
            try:
                prediccion = modelo.predict(roi_procesada, verbose=0)

                # Obtener la clase con la probabilidad más alta
                clase_predicha = np.argmax(prediccion)
                probabilidad = prediccion[0][clase_predicha]

                # Mostrar la emoción solo si la confianza supera el umbral
                if probabilidad > umbral_confianza:
                    mostrar_emocion(frame, clase_predicha, probabilidad, x, y, w, h)
                else:
                    # Si la confianza es baja, mostrar como "Desconocido"
                    cv2.putText(frame, "Desconocido", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            except Exception as e:
                print(f"Error al realizar predicción: {e}")

        # Calcular y mostrar FPS
        fps_contador += 1
        if time.time() - fps_inicio > 1:
            fps = fps_contador
            fps_contador = 0
            fps_inicio = time.time()

        # Mostrar FPS en la pantalla
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar el frame resultante
        cv2.imshow('Reconocimiento de Emociones', frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

# Función para visualizar los resultados 
def visualizar_emociones_simple():
    """
    Versión que muestra solo la cámara con la detección de emociones.
    """
    # Cargar el modelo
    modelo = cargar_modelo()
    if modelo is None:
        print("No se pudo cargar el modelo. Saliendo...")
        return
    
    reconocer_emociones_camara(modelo)

if __name__ == "__main__":
    # Cargar el modelo
    modelo = cargar_modelo()

    if modelo is None:
        print("No se pudo cargar el modelo. Saliendo...")
    else:
        visualizar_emociones_simple()
