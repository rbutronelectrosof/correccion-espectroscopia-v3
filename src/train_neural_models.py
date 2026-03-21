"""
================================================================================
ENTRENAMIENTO DE MODELOS DE REDES NEURONALES PARA CLASIFICACION ESPECTRAL
================================================================================

Este script entrena modelos de Machine Learning (KNN, CNN) para clasificar
automaticamente estrellas segun su tipo espectral (O, B, A, F, G, K, M).

TIPOS DE MODELOS DISPONIBLES:
-----------------------------
1. KNN (K-Nearest Neighbors):
   - Algoritmo simple pero efectivo
   - Usa "features" extraidas del espectro (anchos equivalentes de lineas)
   - Precision tipica: 85-90%
   - Ventaja: Rapido de entrenar, facil de interpretar

2. CNN 1D (Red Neuronal Convolucional 1D):
   - Usa el espectro completo como entrada
   - Aprende patrones automaticamente
   - Precision tipica: 70-85% (depende mucho del entrenamiento)
   - Ventaja: Puede detectar patrones sutiles

3. CNN 2D (Red Neuronal Convolucional 2D):
   - Usa imagenes PNG de los espectros
   - Similar a clasificacion de imagenes
   - Requiere generar imagenes previamente

COMO LOGRAR UN BUEN ENTRENAMIENTO:
----------------------------------
1. DATOS SUFICIENTES:
   - Minimo 50-100 ejemplos por clase
   - Datos balanceados (similar cantidad por tipo espectral)
   - Datos de buena calidad (bien normalizados)

2. HIPERPARAMETROS IMPORTANTES:
   - learning_rate: 0.001 (default) - Si el modelo no aprende, probar 0.0001
   - epochs: 20-50 - Mas epocas si el modelo sigue mejorando
   - batch_size: 16-64 - Mas pequeno = mas ruido pero mejor generalizacion
   - dropout: 0.2-0.5 - Evita sobreajuste (overfitting)

3. SENALES DE BUEN ENTRENAMIENTO:
   - La perdida (loss) disminuye gradualmente
   - La precision (accuracy) aumenta gradualmente
   - La precision de validacion es similar a la de entrenamiento

4. SENALES DE PROBLEMAS:
   - Overfitting: Accuracy de entrenamiento alta, validacion baja
     Solucion: Aumentar dropout, reducir capas, mas datos
   - Underfitting: Ambas accuracy bajas
     Solucion: Mas epochs, reducir dropout, mas neuronas
   - Modelo no aprende: Loss no baja
     Solucion: Reducir learning_rate, verificar datos

USO:
----
    # Entrenar KNN (recomendado para empezar)
    python train_neural_models.py --model knn --catalog data/elodie/ --output models/

    # Entrenar CNN 1D
    python train_neural_models.py --model cnn_1d --catalog data/elodie/ --epochs 30

    # Con parametros personalizados
    python train_neural_models.py --model knn --catalog data/elodie/ --n-neighbors 7
"""

import numpy as np
import os
import sys
import json
import argparse
import time
import pandas as pd
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Agregar directorio actual al path para importaciones
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spectral_classification_corrected import (
    normalize_to_continuum,
    measure_diagnostic_lines
)


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def convert_to_native(obj):
    """
    Convierte tipos numpy a tipos nativos de Python para guardar en JSON.

    Esto es necesario porque JSON no puede serializar tipos numpy directamente.
    """
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.str_, np.bytes_)):
        return str(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def get_tensorflow():
    """
    Importa TensorFlow solo cuando se necesita.

    TensorFlow es una libreria pesada (~500MB), por eso solo se carga
    cuando realmente se va a usar (para CNN).
    """
    try:
        import tensorflow as tf
        # Suprimir mensajes de TensorFlow (son muy verbosos)
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        return tf
    except ImportError:
        print("ERROR: TensorFlow no esta instalado.")
        print("Instala con: pip install tensorflow")
        print("Nota: La instalacion puede tardar varios minutos.")
        sys.exit(1)


# ============================================================================
# CARGA DE DATOS
# ============================================================================

def load_spectrum_file(filepath):
    """
    Carga un archivo de espectro (.txt).

    Los archivos de espectro tienen dos columnas:
    - Columna 1: Longitud de onda (Angstroms)
    - Columna 2: Flujo (intensidad de luz)

    Esta funcion intenta varios formatos (CSV, tabulado, espacios)
    para ser compatible con diferentes fuentes de datos.
    """
    try:
        # Intentar con pandas primero (mas robusto con diferentes formatos)
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                if len(df.columns) >= 2:
                    wavelengths = df.iloc[:, 0].values.astype(float)
                    flux = df.iloc[:, 1].values.astype(float)
                    return wavelengths, flux
            except:
                pass

        # Intentar diferentes delimitadores con numpy
        for delimiter in [',', '\t', ' ', None]:
            for skiprows in [1, 0]:  # Con o sin encabezado
                try:
                    data = np.loadtxt(filepath, delimiter=delimiter, skiprows=skiprows, encoding='latin-1')
                    if len(data.shape) == 2 and data.shape[1] >= 2:
                        wavelengths = data[:, 0]
                        flux = data[:, 1]
                        return wavelengths, flux
                except:
                    continue

    except Exception as e:
        print(f"Error cargando {filepath}: {e}")
    return None, None


def extract_type_from_filename(filename):
    """
    Extrae el tipo espectral del nombre del archivo.

    Los archivos del catalogo ELODIE tienen nombres como:
    - BD+023375_tipoA5.txt  -> Tipo A
    - HD161677_tipo_B6V.txt -> Tipo B

    Solo extraemos la letra principal (O, B, A, F, G, K, M)
    porque los subtipos (A0, A5, etc.) requieren mas datos para distinguir.
    """
    import re

    # Patron 1: _tipoX o _tipo_X
    match = re.search(r'_tipo_?([OBAFGKM]\d*[IV]*)', filename, re.IGNORECASE)
    if match:
        tipo_completo = match.group(1).upper()
        return tipo_completo[0]  # Solo la letra principal

    # Patron 2: tipo espectral en cualquier parte del nombre
    match = re.search(r'([OBAFGKM])\d', filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def load_catalog_data(catalog_path, max_files=None):
    """
    Carga todos los espectros de un catalogo.

    PROCESO:
    1. Lee cada archivo .txt del directorio
    2. Extrae el tipo espectral del nombre del archivo
    3. Normaliza el espectro (ajusta el continuo a 1.0)
    4. Mide las lineas espectrales (calcula anchos equivalentes)
    5. Extrae "features" para el modelo de ML

    FEATURES EXTRAIDAS:
    - Anchos equivalentes de lineas importantes (He, H, Ca, Fe, etc.)
    - Ratios entre lineas (He I/He II, Ca/H, Fe/H)

    Estas features son las que el modelo KNN usa para clasificar.
    La CNN usa el espectro completo en lugar de features.

    Returns:
        spectra: Array de espectros normalizados (para CNN)
        labels: Array de tipos espectrales
        features: Array de features (para KNN)
        filenames: Lista de nombres de archivo
    """
    print(f"\nCargando catalogo desde: {catalog_path}")

    spectra = []
    labels = []
    features_list = []
    filenames = []

    # Buscar archivos .txt
    files = [f for f in os.listdir(catalog_path) if f.endswith('.txt')]

    if max_files:
        files = files[:max_files]

    print(f"Archivos encontrados: {len(files)}")

    for i, filename in enumerate(files):
        filepath = os.path.join(catalog_path, filename)

        # Extraer tipo espectral del nombre
        tipo = extract_type_from_filename(filename)
        if tipo is None:
            continue

        # Cargar espectro
        wavelengths, flux = load_spectrum_file(filepath)
        if wavelengths is None:
            continue

        try:
            # PASO 1: Normalizar al continuo
            # Esto ajusta el espectro para que el continuo este en 1.0
            # Las lineas de absorcion apareceran como valores < 1.0
            flux_normalized, continuum = normalize_to_continuum(wavelengths, flux)

            # PASO 2: Medir lineas diagnostico
            # Calcula el "ancho equivalente" de cada linea espectral
            # El ancho equivalente indica la intensidad de la linea
            measurements = measure_diagnostic_lines(wavelengths, flux_normalized)

            # PASO 3: Extraer features para KNN
            # Estas son las lineas mas importantes para clasificacion
            feature_names = [
                # Helio (estrellas calientes O, B)
                'He_II_4686',   # He II - solo en tipo O (muy caliente)
                'He_I_4471',    # He I - tipos O y B
                'He_I_4026',    # He I - tipos O y B

                # Hidrogeno (serie de Balmer - maxima en tipo A)
                'H_beta',       # Hidrogeno beta (4861 A)
                'H_gamma',      # Hidrogeno gamma (4341 A)
                'H_delta',      # Hidrogeno delta (4102 A)
                'H_epsilon',    # Hidrogeno epsilon (3970 A)

                # Silicio (subtipos de B)
                'Si_IV_4089',   # Si IV - B0-B1
                'Si_III_4553',  # Si III - B1-B3
                'Si_II_4128',   # Si II - B5-B9

                # Metales (estrellas frias F, G, K)
                'Mg_II_4481',   # Magnesio
                'Ca_II_K',      # Calcio K (3934 A) - muy importante para F-K
                'Ca_I_4227',    # Calcio I

                # Hierro (aumenta en tipos G, K, M)
                'Fe_I_4046',
                'Fe_I_4144',
                'Fe_I_4383',
                'Fe_I_4957'
            ]

            features = []
            for name in feature_names:
                ew = measurements.get(name, {}).get('ew', 0.0)
                features.append(ew)

            # PASO 4: Agregar ratios diagnosticos
            # Estos ratios ayudan a distinguir entre tipos similares
            He_I = measurements.get('He_I_4471', {}).get('ew', 0.0)
            He_II = measurements.get('He_II_4686', {}).get('ew', 0.0)
            H_avg = (measurements.get('H_beta', {}).get('ew', 0.0) +
                     measurements.get('H_gamma', {}).get('ew', 0.0) +
                     measurements.get('H_delta', {}).get('ew', 0.0)) / 3.0
            Ca_II_K = measurements.get('Ca_II_K', {}).get('ew', 0.0)
            Fe_avg = (measurements.get('Fe_I_4046', {}).get('ew', 0.0) +
                      measurements.get('Fe_I_4383', {}).get('ew', 0.0)) / 2.0

            features.extend([
                He_I / (He_II + 0.01),      # ratio He I/He II (distingue O de B)
                Ca_II_K / (H_avg + 0.01),   # ratio Ca/H (distingue A de F/G)
                Fe_avg / (H_avg + 0.01),    # ratio Fe/H (distingue F de G/K)
            ])

            spectra.append(flux_normalized)
            labels.append(tipo)
            features_list.append(features)
            filenames.append(filename)

            if (i + 1) % 100 == 0:
                print(f"  Procesados: {i + 1}/{len(files)}")

        except Exception as e:
            print(f"  Error procesando {filename}: {e}")
            continue

    print(f"\nEspectros cargados exitosamente: {len(spectra)}")

    # Mostrar distribucion de tipos (importante para detectar desbalance)
    unique, counts = np.unique(labels, return_counts=True)
    print("\nDistribucion de tipos espectrales:")
    print("-" * 30)
    for t, c in zip(unique, counts):
        print(f"  Tipo {t}: {c:4d} espectros")
    print("-" * 30)

    # Advertencia si hay desbalance
    if len(counts) > 0 and max(counts) / min(counts) > 5:
        print("\n[!] ADVERTENCIA: Los datos estan desbalanceados.")
        print("    Esto puede afectar la precision en clases minoritarias.")
        print("    Considera usar tecnicas de balanceo o recopilar mas datos.")

    return np.array(spectra, dtype=object), np.array(labels), np.array(features_list), filenames


# ============================================================================
# ENTRENAMIENTO KNN (K-Nearest Neighbors)
# ============================================================================

def train_knn(X_train, y_train, n_neighbors=5, weights='uniform', metric='euclidean'):
    """
    Entrena un modelo KNN (K-Nearest Neighbors).

    COMO FUNCIONA KNN:
    ------------------
    Para clasificar un nuevo espectro:
    1. Calcula la distancia a todos los espectros de entrenamiento
    2. Encuentra los K vecinos mas cercanos
    3. Vota: la clase mas comun entre los K vecinos es la prediccion

    PARAMETROS IMPORTANTES:
    -----------------------
    n_neighbors (K): Cuantos vecinos considerar
        - K pequeno (3-5): Mas sensible, puede ser ruidoso
        - K grande (10-15): Mas estable, pero puede perder detalles
        - Recomendado: Empezar con K=5, ajustar segun resultados

    weights: Como ponderar los vecinos
        - 'uniform': Todos los vecinos valen igual
        - 'distance': Vecinos mas cercanos tienen mas peso
        - Recomendado: 'distance' suele dar mejores resultados

    metric: Como medir la distancia
        - 'euclidean': Distancia en linea recta (default)
        - 'manhattan': Suma de diferencias absolutas
        - 'cosine': Angulo entre vectores (bueno para espectros)

    TIPS PARA MEJORAR KNN:
    ----------------------
    1. Normalizar los datos (StandardScaler) - MUY IMPORTANTE
    2. Probar diferentes valores de K (5, 7, 9, 11)
    3. Usar 'weights=distance' si los datos tienen ruido
    4. Si hay muchas features, considerar reduccion de dimensionalidad (PCA)
    """
    print(f"\n" + "="*50)
    print(f"ENTRENANDO KNN")
    print(f"="*50)
    print(f"  Vecinos (K): {n_neighbors}")
    print(f"  Pesos: {weights}")
    print(f"  Metrica: {metric}")
    print(f"  Muestras de entrenamiento: {len(X_train)}")

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        n_jobs=-1  # Usar todos los CPUs disponibles
    )

    model.fit(X_train, y_train)

    print(f"  Entrenamiento completado!")

    return model


# ============================================================================
# ENTRENAMIENTO CNN 1D (Red Neuronal Convolucional)
# ============================================================================

def train_cnn_1d(X_train, y_train, n_classes, epochs=20, batch_size=32,
                 learning_rate=0.001, dropout_rate=0.3, dense_units=128,
                 X_val=None, y_val=None):
    """
    Entrena una Red Neuronal Convolucional 1D para espectros.

    COMO FUNCIONA UNA CNN:
    ----------------------
    1. CAPAS CONVOLUCIONALES: Detectan patrones locales en el espectro
       - Aprenden a reconocer formas de lineas espectrales
       - Cada filtro detecta un patron diferente

    2. CAPAS DE POOLING: Reducen el tamano
       - Hacen el modelo mas robusto a pequenos desplazamientos

    3. CAPAS DENSAS: Combinan los patrones detectados
       - Aprenden que combinaciones de patrones indican cada tipo

    4. CAPA DE SALIDA (SOFTMAX): Produce probabilidades
       - Una probabilidad para cada tipo espectral

    ARQUITECTURA DE ESTE MODELO:
    ----------------------------
    Input (espectro) -> Conv1D(32) -> Pool -> Conv1D(64) -> Pool ->
    Conv1D(128) -> Pool -> Dense(128) -> Dense(64) -> Output(7 clases)

    PARAMETROS IMPORTANTES:
    -----------------------
    epochs: Cuantas veces ver todos los datos
        - Mas epochs = mas aprendizaje, pero riesgo de overfitting
        - Empezar con 20, aumentar si sigue mejorando
        - EarlyStopping detiene automaticamente si no mejora

    batch_size: Cuantas muestras procesar juntas
        - Mas grande = mas rapido, pero menos preciso
        - Mas pequeno = mas lento, pero mejor generalizacion
        - Tipico: 16, 32, 64

    learning_rate: Que tan rapido aprende
        - Muy alto (>0.01): Aprende rapido pero inestable
        - Muy bajo (<0.0001): Muy lento, puede atascarse
        - Recomendado: 0.001 (default), bajar a 0.0001 si es inestable

    dropout_rate: Porcentaje de neuronas a "apagar" durante entrenamiento
        - Previene overfitting (que memorice en lugar de aprender)
        - Tipico: 0.2 a 0.5
        - Si hay overfitting: aumentar dropout
        - Si hay underfitting: reducir dropout

    SENALES DURANTE EL ENTRENAMIENTO:
    ----------------------------------
    - loss: Error del modelo (debe BAJAR)
    - accuracy: Precision (debe SUBIR)
    - val_loss: Error en validacion (debe BAJAR, similar a loss)
    - val_accuracy: Precision en validacion (debe ser similar a accuracy)

    PROBLEMAS COMUNES:
    ------------------
    1. val_loss sube mientras loss baja = OVERFITTING
       Solucion: Mas dropout, menos epochs, mas datos

    2. loss no baja = NO APRENDE
       Solucion: Aumentar learning_rate, verificar datos

    3. accuracy muy baja = UNDERFITTING
       Solucion: Mas epochs, menos dropout, mas neuronas
    """
    tf = get_tensorflow()
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    print(f"\n" + "="*50)
    print(f"ENTRENANDO CNN 1D")
    print(f"="*50)
    print(f"  Epocas maximas: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Dropout: {dropout_rate}")
    print(f"  Neuronas densas: {dense_units}")
    print(f"  Clases a predecir: {n_classes}")
    print(f"  Muestras de entrenamiento: {len(X_train)}")

    # Reshape para Conv1D: (samples, timesteps, features)
    # timesteps = longitud del espectro, features = 1 (solo flujo)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    if X_val is not None:
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    # ARQUITECTURA DE LA RED
    model = Sequential([
        # ========================================
        # BLOQUE 1: Primera capa convolucional
        # ========================================
        # Conv1D: Detecta patrones locales en el espectro
        # - 32 filtros: Aprende 32 patrones diferentes
        # - kernel_size=7: Cada filtro "ve" 7 puntos a la vez
        # - activation='relu': Solo pasa valores positivos
        Conv1D(32, kernel_size=7, activation='relu', padding='same',
               input_shape=(X_train.shape[1], 1)),
        # BatchNormalization: Normaliza las salidas (estabiliza entrenamiento)
        BatchNormalization(),
        # MaxPooling1D: Reduce tamano a la mitad, mantiene valores maximos
        MaxPooling1D(pool_size=2),
        # Dropout: "Apaga" 15% de neuronas aleatoriamente (evita overfitting)
        Dropout(dropout_rate * 0.5),

        # ========================================
        # BLOQUE 2: Segunda capa convolucional
        # ========================================
        # Mas filtros (64) para patrones mas complejos
        Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate * 0.7),

        # ========================================
        # BLOQUE 3: Tercera capa convolucional
        # ========================================
        # Aun mas filtros (128) para patrones de alto nivel
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),

        # ========================================
        # CAPAS DENSAS (fully connected)
        # ========================================
        # Flatten: Convierte el tensor 3D a vector 1D
        Flatten(),
        # Dense: Capa completamente conectada
        Dense(dense_units, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(dense_units // 2, activation='relu'),
        Dropout(dropout_rate * 0.5),

        # ========================================
        # CAPA DE SALIDA
        # ========================================
        # Softmax: Convierte a probabilidades (suman 1.0)
        # n_classes neuronas, una por cada tipo espectral
        Dense(n_classes, activation='softmax')
    ])

    # COMPILAR EL MODELO
    model.compile(
        # Adam: Optimizador adaptativo, ajusta learning rate automaticamente
        optimizer=Adam(learning_rate=learning_rate),
        # sparse_categorical_crossentropy: Para clasificacion multiclase
        loss='sparse_categorical_crossentropy',
        # Metrica a monitorear
        metrics=['accuracy']
    )

    # CALLBACKS: Funciones que se ejecutan durante el entrenamiento
    callbacks = [
        # EarlyStopping: Detiene si no mejora en 5 epochs
        # Evita perder tiempo y overfitting
        EarlyStopping(
            monitor='val_loss',      # Monitorear error de validacion
            patience=5,              # Esperar 5 epochs antes de parar
            restore_best_weights=True  # Restaurar mejor modelo
        ),
        # ReduceLROnPlateau: Reduce learning rate si se estanca
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,              # Reducir a la mitad
            patience=3,              # Despues de 3 epochs sin mejora
            min_lr=1e-6              # No bajar de este valor
        )
    ]

    print(f"\n  Iniciando entrenamiento...")
    print(f"  (EarlyStopping activado: se detendra si no mejora)")
    print()

    # ENTRENAR
    validation_data = (X_val, y_val) if X_val is not None else None
    validation_split = 0.2 if validation_data is None else 0.0

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1  # Mostrar progreso
    )

    print(f"\n  Entrenamiento completado!")
    print(f"  Epocas ejecutadas: {len(history.history['loss'])}")

    return model, history


# ============================================================================
# ENTRENAMIENTO CNN 2D (para imagenes)
# ============================================================================

def train_cnn_2d(image_dir, labels_dict, epochs=20, batch_size=32,
                 image_size=(128, 128), dropout_rate=0.3, learning_rate=0.001):
    """
    Entrena una CNN 2D para imagenes de espectros.

    Similar a la CNN 1D pero para imagenes PNG.
    Util si tienes espectros como graficos/imagenes.
    """
    tf = get_tensorflow()
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from PIL import Image

    print(f"\nEntrenando CNN 2D:")
    print(f"  Directorio de imagenes: {image_dir}")
    print(f"  Tamano de imagen: {image_size}")
    print(f"  Epocas: {epochs}")

    # Cargar imagenes
    images = []
    labels = []

    for img_name, tipo in labels_dict.items():
        img_path = os.path.join(image_dir, img_name)
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('L')  # Escala de grises
                img = img.resize(image_size)
                images.append(np.array(img) / 255.0)  # Normalizar a 0-1
                labels.append(tipo)
            except Exception as e:
                print(f"  Error cargando {img_name}: {e}")

    if len(images) == 0:
        raise ValueError("No se encontraron imagenes validas")

    print(f"  Imagenes cargadas: {len(images)}")

    X = np.array(images).reshape(-1, image_size[0], image_size[1], 1)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    n_classes = len(encoder.classes_)

    print(f"  Clases: {list(encoder.classes_)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(image_size[0], image_size[1], 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate * 0.5),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate * 0.7),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(128, activation='relu'),
        Dropout(dropout_rate * 0.5),

        Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluar
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nAccuracy en test: {test_acc * 100:.2f}%")

    return model, encoder, history, (X_test, y_test)


# ============================================================================
# PIPELINE PRINCIPAL DE ENTRENAMIENTO
# ============================================================================

def train_and_save_model(model_type, catalog_path, output_dir, **kwargs):
    """
    Pipeline completo: carga datos, entrena modelo, guarda resultados.

    Esta funcion coordina todo el proceso de entrenamiento:
    1. Carga los espectros del catalogo
    2. Prepara los datos (normaliza, extrae features)
    3. Divide en entrenamiento y prueba
    4. Entrena el modelo elegido
    5. Evalua el rendimiento
    6. Guarda el modelo y metadata
    """
    print(f"\n{'='*60}")
    print(f"  INICIANDO ENTRENAMIENTO: {model_type.upper()}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    # ========================================
    # PASO 1: Cargar datos
    # ========================================
    spectra, labels, features, filenames = load_catalog_data(
        catalog_path,
        max_files=kwargs.get('max_files', None)
    )

    if len(spectra) == 0:
        raise ValueError("No se cargaron espectros validos")

    # ========================================
    # PASO 2: Filtrar clases con pocas muestras
    # ========================================
    # Necesitamos minimo 5 muestras por clase para hacer un split estratificado
    MIN_SAMPLES_PER_CLASS = 5
    print(f"\nFiltrando clases con menos de {MIN_SAMPLES_PER_CLASS} muestras...")

    try:
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_classes = unique_labels[counts >= MIN_SAMPLES_PER_CLASS]
    except Exception as e:
        print(f"  ERROR en filtrado: {e}")
        raise

    if len(valid_classes) < len(unique_labels):
        removed = set(unique_labels) - set(valid_classes)
        print(f"\n[!] Eliminando clases con muy pocas muestras: {removed}")

        valid_set = set(valid_classes)
        filtered_indices = [i for i, lbl in enumerate(labels) if lbl in valid_set]

        spectra = np.array([spectra[i] for i in filtered_indices], dtype=object)
        labels = np.array([labels[i] for i in filtered_indices])
        features = np.array([features[i] for i in filtered_indices])
        filenames = [filenames[i] for i in filtered_indices]

        print(f"   Muestras restantes: {len(spectra)}")

    # ========================================
    # PASO 3: Codificar labels
    # ========================================
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    classes = [str(c) for c in encoder.classes_]
    n_classes = len(classes)

    print(f"\nClases finales: {classes}")
    print(f"Total de muestras: {len(spectra)}")

    # ========================================
    # PASO 4: Dividir datos (entrenamiento / prueba)
    # ========================================
    test_size = kwargs.get('test_size', 0.2)
    print(f"\nDividiendo datos: {int((1-test_size)*100)}% entrenamiento, {int(test_size*100)}% prueba")

    results = {
        'model_type': model_type,
        'n_samples': len(spectra),
        'n_classes': n_classes,
        'classes': classes,
        'test_size': test_size,
        'timestamp': datetime.now().isoformat()
    }

    # ========================================
    # PASO 5: Entrenar segun el tipo de modelo
    # ========================================

    if model_type == 'knn':
        # -------------------------------------
        # KNN: Usa features (anchos equivalentes)
        # -------------------------------------
        X = features

        # IMPORTANTE: Normalizar features (StandardScaler)
        # KNN es sensible a la escala de los datos
        # Sin normalizar, features con valores grandes dominarian
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, stratify=y, random_state=42
        )

        # Entrenar
        model = train_knn(
            X_train, y_train,
            n_neighbors=kwargs.get('n_neighbors', 5),
            weights=kwargs.get('weights', 'uniform'),
            metric=kwargs.get('metric', 'euclidean')
        )

        # Evaluar
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Cross-validation (evaluacion mas robusta)
        # Divide los datos en 5 partes y entrena/evalua 5 veces
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)

        results.update({
            'accuracy_test': float(accuracy),
            'accuracy_cv_mean': float(cv_scores.mean()),
            'accuracy_cv_std': float(cv_scores.std()),
            'n_neighbors': kwargs.get('n_neighbors', 5),
            'weights': kwargs.get('weights', 'uniform'),
            'metric': kwargs.get('metric', 'euclidean')
        })

        # Guardar modelo y scaler
        model_path = os.path.join(output_dir, 'knn_model.pkl')
        scaler_path = os.path.join(output_dir, 'knn_scaler.pkl')

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        print(f"\nModelo KNN guardado en: {model_path}")
        print(f"Scaler guardado en: {scaler_path}")

    elif model_type == 'cnn_1d':
        # -------------------------------------
        # CNN 1D: Usa espectro completo
        # -------------------------------------
        print("\nVerificando TensorFlow...")
        tf = get_tensorflow()
        print(f"  TensorFlow version: {tf.__version__}")

        # Interpolar todos los espectros a la misma longitud
        # (necesario porque la CNN espera entrada de tamano fijo)
        target_length = kwargs.get('spectrum_length', 1000)
        print(f"\nInterpolando espectros a longitud {target_length}...")

        X = []
        for i, spec in enumerate(spectra):
            try:
                if len(spec) != target_length:
                    x_old = np.linspace(0, 1, len(spec))
                    x_new = np.linspace(0, 1, target_length)
                    spec_interp = np.interp(x_new, x_old, spec)
                    X.append(spec_interp)
                else:
                    X.append(spec)
            except Exception as e:
                print(f"  Error interpolando espectro {i}: {e}")
                continue

        X = np.array(X, dtype=np.float32)
        print(f"  Shape de datos: {X.shape}")

        if len(X) < 10:
            raise ValueError(f"Muy pocos espectros validos: {len(X)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

        # Entrenar CNN
        model, history = train_cnn_1d(
            X_train, y_train, n_classes,
            epochs=kwargs.get('epochs', 20),
            batch_size=kwargs.get('batch_size', 32),
            learning_rate=kwargs.get('learning_rate', 0.001),
            dropout_rate=kwargs.get('dropout_rate', 0.3),
            dense_units=kwargs.get('dense_units', 128),
            X_val=X_test, y_val=y_test
        )

        # Evaluar
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        test_loss, accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)

        results.update({
            'accuracy_test': float(accuracy),
            'final_loss': float(test_loss),
            'epochs_trained': len(history.history['loss']),
            'spectrum_length': target_length,
            'epochs': kwargs.get('epochs', 20),
            'batch_size': kwargs.get('batch_size', 32),
            'learning_rate': kwargs.get('learning_rate', 0.001),
            'dropout_rate': kwargs.get('dropout_rate', 0.3),
            'dense_units': kwargs.get('dense_units', 128)
        })

        # Guardar modelo
        model_path = os.path.join(output_dir, 'cnn_model.h5')
        model.save(model_path)

        print(f"\nModelo CNN 1D guardado en: {model_path}")

    elif model_type == 'cnn_2d':
        # -------------------------------------
        # CNN 2D: Usa imagenes PNG
        # -------------------------------------
        image_dir = kwargs.get('image_dir', catalog_path)

        png_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]
        if len(png_files) == 0:
            raise ValueError(
                f"CNN_2D requiere imagenes PNG.\n"
                f"No se encontraron archivos .png en: {image_dir}\n"
                f"Usa KNN o CNN_1D para datos espectrales (.txt)"
            )

        labels_dict = {}
        for filename, tipo in zip(filenames, labels):
            img_name = filename.replace('.txt', '.png')
            if img_name in png_files:
                labels_dict[img_name] = str(tipo)

        if len(labels_dict) == 0:
            raise ValueError("No se encontraron imagenes PNG correspondientes")

        model, img_encoder, history, (X_test, y_test) = train_cnn_2d(
            image_dir, labels_dict,
            epochs=kwargs.get('epochs', 20),
            batch_size=kwargs.get('batch_size', 32),
            image_size=kwargs.get('image_size', (128, 128)),
            dropout_rate=kwargs.get('dropout_rate', 0.3),
            learning_rate=kwargs.get('learning_rate', 0.001)
        )

        test_loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        results.update({
            'accuracy_test': float(accuracy),
            'final_loss': float(test_loss),
            'epochs_trained': len(history.history['loss']),
            'image_size': kwargs.get('image_size', (128, 128))
        })

        model_path = os.path.join(output_dir, 'cnn_2d_model.h5')
        model.save(model_path)

        print(f"\nModelo CNN 2D guardado en: {model_path}")

    # ========================================
    # PASO 6: Guardar metadata
    # ========================================
    training_time = time.time() - start_time
    results['training_time_seconds'] = training_time

    metadata_path = os.path.join(output_dir, f'{model_type}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(convert_to_native(results), f, indent=2)

    # ========================================
    # RESUMEN FINAL
    # ========================================
    print(f"\n{'='*60}")
    print(f"  RESULTADOS DE ENTRENAMIENTO")
    print(f"{'='*60}")
    print(f"  Modelo: {model_type.upper()}")
    print(f"  Accuracy en test: {results['accuracy_test'] * 100:.2f}%")
    if 'accuracy_cv_mean' in results:
        print(f"  Accuracy cross-val: {results['accuracy_cv_mean'] * 100:.1f}% (+/- {results['accuracy_cv_std'] * 100:.1f}%)")
    print(f"  Muestras totales: {results['n_samples']}")
    print(f"  Clases: {results['classes']}")
    print(f"  Tiempo de entrenamiento: {training_time:.1f} segundos")
    print(f"{'='*60}")

    # Interpretacion del resultado
    acc = results['accuracy_test'] * 100
    print(f"\n  INTERPRETACION:")
    if acc >= 85:
        print(f"  [OK] Excelente! El modelo tiene muy buen rendimiento.")
    elif acc >= 70:
        print(f"  [OK] Buen rendimiento. Podria mejorar con mas datos o ajustes.")
    elif acc >= 50:
        print(f"  [!] Rendimiento moderado. Considera:")
        print(f"      - Mas datos de entrenamiento")
        print(f"      - Ajustar hiperparametros")
        print(f"      - Verificar calidad de los datos")
    else:
        print(f"  [X] Rendimiento bajo. El modelo necesita mejoras significativas.")

    print()

    return results


# ============================================================================
# INTERFAZ DE LINEA DE COMANDOS
# ============================================================================

def main():
    """
    Punto de entrada principal.

    Ejemplos de uso:
    ----------------
    # Entrenar KNN (rapido, recomendado para empezar)
    python train_neural_models.py --model knn --catalog data/elodie/ --output models/


    # Entrenar KNN con mas vecinos
    python train_neural_models.py --model knn --catalog data/elodie/ --n-neighbors 7

    # Entrenar CNN 1D (requiere TensorFlow)
    python train_neural_models.py --model cnn_1d --catalog data/elodie/ --epochs 30

    # CNN con parametros personalizados
    python train_neural_models.py --model cnn_1d --catalog data/elodie/ \\
        --epochs 50 --batch-size 16 --learning-rate 0.0005 --dropout 0.4
    """
    sys.stdout.reconfigure(line_buffering=True)
    print("=== Modulos cargados. Preparando entrenamiento neural... ===", flush=True)

    parser = argparse.ArgumentParser(
        description='Entrenar modelos KNN/CNN para clasificacion espectral',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EJEMPLOS:
  # KNN (rapido, bueno para empezar)
  python train_neural_models.py --model knn --catalog data/elodie/

  # CNN 1D (mas preciso, requiere TensorFlow)
  python train_neural_models.py --model cnn_1d --catalog data/elodie/ --epochs 30

TIPS PARA MEJORAR:
  - KNN: Probar --n-neighbors 3,5,7,9 y --weights distance
  - CNN: Si hay overfitting, aumentar --dropout 0.4 o 0.5
  - CNN: Si no aprende, reducir --learning-rate 0.0001
        """
    )

    parser.add_argument('--model', type=str, required=True, choices=['knn', 'cnn_1d', 'cnn_2d'],
                        help='Tipo de modelo: knn, cnn_1d, o cnn_2d')
    parser.add_argument('--catalog', type=str, required=True,
                        help='Directorio con espectros (.txt)')
    parser.add_argument('--output', type=str, default='models/',
                        help='Directorio para guardar el modelo')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraccion de datos para prueba (default: 0.2)')

    # Parametros KNN
    parser.add_argument('--n-neighbors', type=int, default=5,
                        help='[KNN] Numero de vecinos K (default: 5)')
    parser.add_argument('--weights', type=str, default='uniform',
                        choices=['uniform', 'distance'],
                        help='[KNN] Pesos: uniform o distance')
    parser.add_argument('--metric', type=str, default='euclidean',
                        choices=['euclidean', 'manhattan', 'cosine'],
                        help='[KNN] Metrica de distancia')

    # Parametros CNN
    parser.add_argument('--epochs', type=int, default=20,
                        help='[CNN] Numero de epocas (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='[CNN] Tamano de batch (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='[CNN] Tasa de aprendizaje (default: 0.001)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='[CNN] Tasa de dropout (default: 0.3)')
    parser.add_argument('--dense-units', type=int, default=128,
                        help='[CNN] Neuronas en capa densa (default: 128)')

    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximo numero de archivos a procesar (para pruebas)')

    args = parser.parse_args()

    # Entrenar
    results = train_and_save_model(
        model_type=args.model,
        catalog_path=args.catalog,
        output_dir=args.output,
        test_size=args.test_size,
        n_neighbors=args.n_neighbors,
        weights=args.weights,
        metric=args.metric,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout,
        dense_units=args.dense_units,
        max_files=args.max_files
    )

    return results


if __name__ == '__main__':
    main()
