import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import time

# Definir las clases de emociones (inglés y español)
CLASES_EMOCIONES = ['amusement', 'excitement', 'anger', 'fear', 'sadness', 'contentment', 'awe', 'disgust'] # Esto es para que funcione el modelo
EMOCIONES_ESPAÑOL = ['diversión', 'emoción', 'enojo', 'miedo', 'tristeza', 'satisfacción', 'asombro', 'disgusto'] # Y estas solo para mostrarlas 

# Cargar el clasificador de rostros de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar el modelo entrenado
def cargar_modelo(ruta_modelo='modelo_emociones_001.h5'):
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

# Función para analizar las emociones en una imagen
def analizar_emociones_imagen(ruta_imagen, modelo, tamaño_modelo, umbral_confianza=0.5):
    """
    Analiza las emociones de los rostros en una imagen.
    Retorna la imagen con las anotaciones y los resultados de las emociones.
    """
    # Cargar la imagen
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
        return None, []
    
    # Convertir a escala de grises para detección de rostros
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros
    rostros = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    resultados = []
    
    # Procesamiento de cada rostro detectado
    for i, (x, y, w, h) in enumerate(rostros):
        # Extraer la región de interés (ROI)
        roi = imagen[y:y+h, x:x+w]
        
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
                # Obtener la emoción en español
                emocion_esp = EMOCIONES_ESPAÑOL[clase_predicha]
                
                # Convertir probabilidad a puntos (1-10)
                puntos = probabilidad_a_puntos(probabilidad)
                
                # Texto de la emoción
                texto = f"{emocion_esp}: {puntos} pts"
                
                # Determinar el color según la emoción
                if CLASES_EMOCIONES[clase_predicha] in ['amusement', 'contentment', 'awe', 'excitement']:
                    color = (0, 255, 0)  # Verde para emociones positivas
                elif CLASES_EMOCIONES[clase_predicha] in ['anger', 'fear', 'sadness', 'disgust']:
                    color = (0, 0, 255)  # Rojo para emociones negativas
                else:
                    color = (255, 255, 0)  # Azul para neutral
                    
                # Dibujar el rectángulo alrededor del rostro
                cv2.rectangle(imagen, (x, y), (x+w, y+h), color, 2)
                
                # Colocar el texto encima de la cabeza
                posicion_texto_y = y - 20 if y - 20 > 20 else 20
                cv2.putText(imagen, texto, (x, posicion_texto_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Guardar los resultados para el gráfico
                resultados.append({
                    'rostro': i + 1,
                    'emocion': emocion_esp,
                    'probabilidad': probabilidad,
                    'puntos': puntos
                })
            else:
                # Si la confianza es baja, mostrar como "Desconocido"
                cv2.putText(imagen, "Desconocido", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        except Exception as e:
            print(f"Error al realizar predicción: {e}")
    
    return imagen, resultados

# Función para crear un gráfico de barras de las emociones detectadas
def crear_grafico_emociones(resultados, nombre_imagen):
    """
    Crea un gráfico de barras con las emociones detectadas en la imagen.
    """
    if not resultados:
        print(f"No se detectaron emociones en la imagen {nombre_imagen}")
        return None
    
    # Configurar el gráfico
    plt.figure(figsize=(12, 6))
    
    # Crear datos para el gráfico
    rostros = [f"Rostro {r['rostro']}" for r in resultados]
    puntos = [r['puntos'] for r in resultados]
    emociones = [r['emocion'] for r in resultados]
    
    # Crear barras con colores según la emoción
    colores = []
    for emocion in emociones:
        if emocion in ['diversión', 'emoción', 'satisfacción', 'asombro']:
            colores.append('green')
        elif emocion in ['enojo', 'miedo', 'tristeza', 'disgusto']:
            colores.append('red')
        else:
            colores.append('blue')
    
    # Crear el gráfico de barras
    barras = plt.bar(rostros, puntos, color=colores)
    
    # Añadir etiquetas a las barras
    for i, barra in enumerate(barras):
        plt.text(barra.get_x() + barra.get_width()/2., barra.get_height() + 0.3,
                emociones[i], ha='center', va='bottom', rotation=0)
    
    # Configurar el gráfico
    plt.title(f"Emociones detectadas en {nombre_imagen}")
    plt.xlabel("Rostros detectados")
    plt.ylabel("Puntuación (1-10)")
    plt.ylim(0, 11)  # Establecer límite del eje Y
    plt.tight_layout()
    
    return plt.gcf()

# Función principal para procesar un directorio de imágenes
def procesar_directorio_imagenes(directorio_entrada, directorio_salida, modelo):
    """
    Procesa todas las imágenes en un directorio y guarda los resultados.
    """
    # Verificar que los directorios existan
    if not os.path.exists(directorio_entrada):
        print(f"Error: El directorio de entrada {directorio_entrada} no existe.")
        return
    
    # Crear el directorio de salida si no existe
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)
    
    # Crear subdirectorios para imágenes y gráficos
    dir_imagenes = os.path.join(directorio_salida, "imagenes")
    dir_graficos = os.path.join(directorio_salida, "graficos")
    
    if not os.path.exists(dir_imagenes):
        os.makedirs(dir_imagenes)
    if not os.path.exists(dir_graficos):
        os.makedirs(dir_graficos)
    
    # Obtener el tamaño de entrada que espera el modelo
    tamaño_modelo = obtener_tamaño_modelo(modelo)
    
    # Obtener la lista de archivos de imagen
    extensiones_validas = ['.jpg', '.jpeg', '.png', '.bmp']
    archivos_imagen = [f for f in os.listdir(directorio_entrada) 
                      if os.path.splitext(f.lower())[1] in extensiones_validas]
    
    if not archivos_imagen:
        print(f"No se encontraron imágenes en el directorio {directorio_entrada}")
        return
    
    print(f"Procesando {len(archivos_imagen)} imágenes...")
    
    # Procesar cada imagen
    resultados_totales = []
    
    for i, archivo in enumerate(archivos_imagen):
        print(f"Procesando imagen {i+1}/{len(archivos_imagen)}: {archivo}")
        
        # Ruta completa de la imagen
        ruta_imagen = os.path.join(directorio_entrada, archivo)
        
        # Analizar la imagen
        imagen_anotada, resultados = analizar_emociones_imagen(ruta_imagen, modelo, tamaño_modelo)
        
        if imagen_anotada is None:
            continue
        
        # Guardar la imagen anotada
        nombre_base = os.path.splitext(archivo)[0]
        ruta_salida_imagen = os.path.join(dir_imagenes, f"{nombre_base}_anotado.jpg")
        cv2.imwrite(ruta_salida_imagen, imagen_anotada)
        print(f"  - Imagen anotada guardada en: {ruta_salida_imagen}")
        
        # Crear y guardar el gráfico
        if resultados:
            figura = crear_grafico_emociones(resultados, archivo)
            if figura:
                ruta_salida_grafico = os.path.join(dir_graficos, f"{nombre_base}_grafico.png")
                figura.savefig(ruta_salida_grafico)
                plt.close(figura)
                print(f"  - Gráfico guardado en: {ruta_salida_grafico}")
            
            # Guardar los resultados para el informe final
            resultados_totales.append({
                'imagen': archivo,
                'rostros_detectados': len(resultados),
                'resultados': resultados
            })
    
    # Crear un gráfico de resumen
    if resultados_totales:
        ruta_resumen = os.path.join(directorio_salida, "resumen_emociones.png")
        crear_grafico_resumen(resultados_totales, ruta_resumen)
        print(f"Resumen guardado en: {ruta_resumen}")
    
    print(f"Procesamiento completado. Resultados guardados en {directorio_salida}")

# Función para crear un gráfico de resumen de todas las imágenes
def crear_grafico_resumen(resultados_totales, ruta_salida):
    """
    Crea un gráfico de resumen con la distribución de emociones en todas las imágenes.
    """
    # Contar las emociones
    contador_emociones = {emocion: 0 for emocion in EMOCIONES_ESPAÑOL}
    total_rostros = 0
    
    for resultado_imagen in resultados_totales:
        for deteccion in resultado_imagen['resultados']:
            contador_emociones[deteccion['emocion']] += 1
            total_rostros += 1
    
    # Filtrar emociones con conteo > 0
    emociones = [emocion for emocion, conteo in contador_emociones.items() if conteo > 0]
    conteos = [contador_emociones[emocion] for emocion in emociones]
    
    if not emociones:
        print("No se detectaron emociones en ninguna imagen")
        return
    
    # Crear gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Gráfico de barras
    ax1.bar(emociones, conteos)
    ax1.set_title("Distribución de emociones")
    ax1.set_xlabel("Emoción")
    ax1.set_ylabel("Número de rostros")
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    
    # Gráfico circular
    ax2.pie(conteos, labels=emociones, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    ax2.set_title("Proporción de emociones")
    
    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close()
    
    print(f"Resumen: Se analizaron {total_rostros} rostros en {len(resultados_totales)} imágenes")

# Función principal que se ejecuta al iniciar el script
def main():
    # directorios de entrada y salida
    directorio_entrada = "imagenes"
    directorio_salida = "resultados"
    
    # Cargar el modelo
    modelo = cargar_modelo()
    if modelo is None:
        print("No se pudo cargar el modelo. Saliendo...")
        return
    
    # Procesar las imágenes
    procesar_directorio_imagenes(directorio_entrada, directorio_salida, modelo)

if __name__ == "__main__":
    main()