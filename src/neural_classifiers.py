"""
================================================================================
CLASIFICADORES NEURONALES PARA ESPECTROS ESTELARES
================================================================================

Este modulo contiene las clases para cargar y usar modelos pre-entrenados
(KNN y CNN) para clasificacion espectral.

MODELOS SOPORTADOS:
-------------------
1. KNN (K-Nearest Neighbors):
   - Archivo: knn_model.pkl
   - Scaler: knn_scaler.pkl (normaliza los datos)
   - Usa features extraidas (anchos equivalentes de lineas)

2. CNN 1D (Red Neuronal Convolucional):
   - Archivo: cnn_model.h5
   - Usa el espectro completo como entrada
   - Requiere TensorFlow instalado

3. CNN 2D (para imagenes):
   - Archivo: cnn_2d_model.h5
   - Usa imagenes PNG de espectros

USO BASICO:
-----------
    from neural_classifiers import NeuralClassifierManager

    # Cargar todos los modelos disponibles
    manager = NeuralClassifierManager(models_dir='models/')

    # Ver que modelos estan disponibles
    print(manager.get_available_models())  # ['knn', 'cnn_1d']

    # Predecir con KNN (usando mediciones de lineas)
    resultado = manager.predict('knn', measurements=measurements_dict)

    # Predecir con CNN 1D (usando espectro normalizado)
    resultado = manager.predict('cnn_1d', spectrum=flux_normalized)

    # El resultado contiene:
    # - tipo: Tipo espectral predicho (ej: 'G')
    # - confianza: Probabilidad del tipo predicho (0-100%)
    # - probabilidades: Probabilidad de cada clase
"""

import numpy as np
import os
import json
import joblib


# ============================================================================
# CLASIFICADOR KNN
# ============================================================================

class KNNClassifier:
    """
    Clasificador KNN (K-Nearest Neighbors) para espectros estelares.

    COMO FUNCIONA:
    --------------
    1. Extrae "features" del espectro (anchos equivalentes de lineas)
    2. Normaliza las features con el scaler guardado
    3. Busca los K espectros mas similares en el conjunto de entrenamiento
    4. La clase mas comun entre los K vecinos es la prediccion

    FEATURES UTILIZADAS:
    --------------------
    - Lineas de Helio (He I, He II) - estrellas calientes
    - Lineas de Hidrogeno (H beta, H gamma, etc.) - tipos A-F
    - Lineas de Calcio (Ca II K) - estrellas frias
    - Lineas de Hierro (Fe I) - estrellas frias
    - Ratios entre lineas (He I/He II, Ca/H, Fe/H)
    """

    def __init__(self, model_path):
        """
        Carga un modelo KNN entrenado.

        Parameters:
            model_path: Ruta al archivo .pkl del modelo
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

        # Cargar modelo KNN
        self.model = joblib.load(model_path)
        self.model_path = model_path
        self.model_dir = os.path.dirname(model_path)

        # Cargar scaler (IMPORTANTE: normaliza los datos igual que en entrenamiento)
        scaler_path = model_path.replace('.pkl', '_scaler.pkl').replace('_model', '_scaler')
        if not os.path.exists(scaler_path):
            scaler_path = os.path.join(self.model_dir, 'knn_scaler.pkl')

        self.scaler = None
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        # Cargar metadata (informacion del entrenamiento)
        metadata_path = model_path.replace('.pkl', '_metadata.json').replace('_model', '_metadata')
        if not os.path.exists(metadata_path):
            metadata_path = os.path.join(self.model_dir, 'knn_metadata.json')

        self.metadata = {}
        self.classes = []

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.classes = self.metadata.get('classes', [])

        # Si no hay metadata, obtener clases del modelo
        if not self.classes and hasattr(self.model, 'classes_'):
            self.classes = list(self.model.classes_)

    def extract_features(self, measurements):
        """
        Extrae features de las mediciones de lineas espectrales.

        Las features son los anchos equivalentes de las lineas diagnosticas
        mas importantes para clasificacion espectral.

        Parameters:
            measurements: dict de mediciones (de measure_diagnostic_lines())

        Returns:
            features: array numpy con las features
        """
        # Lista de lineas a extraer (debe coincidir con el entrenamiento)
        feature_names = [
            # Helio - estrellas O y B
            'He_II_4686', 'He_I_4471', 'He_I_4026',
            # Hidrogeno - maximo en tipo A
            'H_beta', 'H_gamma', 'H_delta', 'H_epsilon',
            # Silicio - subtipos de B
            'Si_IV_4089', 'Si_III_4553', 'Si_II_4128',
            # Metales - estrellas F, G, K
            'Mg_II_4481', 'Ca_II_K', 'Ca_I_4227',
            # Hierro - aumenta hacia tipos frios
            'Fe_I_4046', 'Fe_I_4144', 'Fe_I_4383', 'Fe_I_4957'
        ]

        features = []
        for name in feature_names:
            # Obtener ancho equivalente (EW) de cada linea
            ew = measurements.get(name, {}).get('ew', 0.0)
            features.append(ew)

        # Agregar ratios diagnosticos (ayudan a distinguir tipos similares)
        He_I = measurements.get('He_I_4471', {}).get('ew', 0.0)
        He_II = measurements.get('He_II_4686', {}).get('ew', 0.0)
        H_avg = (measurements.get('H_beta', {}).get('ew', 0.0) +
                 measurements.get('H_gamma', {}).get('ew', 0.0) +
                 measurements.get('H_delta', {}).get('ew', 0.0)) / 3.0
        Ca_II_K = measurements.get('Ca_II_K', {}).get('ew', 0.0)
        Fe_avg = (measurements.get('Fe_I_4046', {}).get('ew', 0.0) +
                  measurements.get('Fe_I_4383', {}).get('ew', 0.0)) / 2.0

        features.extend([
            He_I / (He_II + 0.01),      # Distingue O de B
            Ca_II_K / (H_avg + 0.01),   # Distingue A de F/G
            Fe_avg / (H_avg + 0.01),    # Distingue F de G/K
        ])

        return np.array(features)

    def predict(self, features=None, measurements=None):
        """
        Predice el tipo espectral.

        Parameters:
            features: array de features ya extraidas
            measurements: dict de mediciones (si no se dan features)

        Returns:
            dict con:
            - tipo: Tipo espectral predicho
            - confianza: Probabilidad (0-100%)
            - probabilidades: dict con probabilidad de cada clase
            - metodo: 'knn'
        """
        if features is None and measurements is None:
            raise ValueError("Debe proporcionar features o measurements")

        if features is None:
            features = self.extract_features(measurements)

        # Asegurar formato correcto (2D)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        # IMPORTANTE: Normalizar con el mismo scaler del entrenamiento
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Obtener probabilidades de cada clase
        proba = self.model.predict_proba(features)[0]
        pred_idx = np.argmax(proba)

        # Obtener nombre de la clase predicha
        if self.classes:
            tipo = self.classes[pred_idx]
        elif hasattr(self.model, 'classes_'):
            tipo = str(self.model.classes_[pred_idx])
        else:
            tipo = str(pred_idx)

        # Construir diccionario de probabilidades
        probabilidades = {}
        if self.classes:
            for i, clase in enumerate(self.classes):
                probabilidades[clase] = float(proba[i] * 100)
        else:
            for i, p in enumerate(proba):
                probabilidades[str(i)] = float(p * 100)

        return {
            'tipo': tipo,
            'confianza': float(proba[pred_idx] * 100),
            'probabilidades': probabilidades,
            'metodo': 'knn'
        }

    def get_info(self):
        """Retorna informacion del modelo."""
        return {
            'tipo': 'knn',
            'clases': self.classes,
            'n_neighbors': getattr(self.model, 'n_neighbors', None),
            'metric': getattr(self.model, 'metric', None),
            'accuracy': self.metadata.get('accuracy_test', None)
        }


# ============================================================================
# CLASIFICADOR CNN
# ============================================================================

class CNNClassifier:
    """
    Clasificador CNN (Red Neuronal Convolucional) para espectros estelares.

    COMO FUNCIONA:
    --------------
    1. Recibe el espectro normalizado completo
    2. Lo pasa por capas convolucionales que detectan patrones
    3. Las capas densas combinan los patrones para clasificar
    4. La capa softmax produce probabilidades para cada tipo

    TIPOS DE CNN:
    -------------
    - CNN 1D: Para datos espectrales (array de flujo)
    - CNN 2D: Para imagenes PNG de espectros

    NOTA: Requiere TensorFlow instalado (pip install tensorflow)
    """

    def __init__(self, model_path, model_type='1d'):
        """
        Carga un modelo CNN entrenado.

        Parameters:
            model_path: Ruta al archivo .h5 o .keras del modelo
            model_type: '1d' para espectros, '2d' para imagenes
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

        # Importar TensorFlow (solo cuando se necesita)
        try:
            import tensorflow as tf
            # Suprimir mensajes de TensorFlow
            tf.get_logger().setLevel('ERROR')
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            self.tf = tf
        except ImportError:
            raise ImportError(
                "TensorFlow no esta instalado.\n"
                "Instala con: pip install tensorflow\n"
                "Nota: La instalacion puede tardar varios minutos."
            )

        # Cargar modelo
        self.model = tf.keras.models.load_model(model_path)
        self.model_path = model_path
        self.model_dir = os.path.dirname(model_path)
        self.model_type = model_type

        # Cargar metadata
        metadata_path = model_path.replace('.h5', '_metadata.json').replace('.keras', '_metadata.json')
        if not os.path.exists(metadata_path):
            if '1d' in model_path or model_type == '1d':
                metadata_path = os.path.join(self.model_dir, 'cnn_1d_metadata.json')
            else:
                metadata_path = os.path.join(self.model_dir, 'cnn_2d_metadata.json')

        if not os.path.exists(metadata_path):
            metadata_path = os.path.join(self.model_dir, 'cnn_metadata.json')

        self.metadata = {}
        self.classes = []

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.classes = self.metadata.get('classes', [])

        # Parametros de preprocesamiento
        self.spectrum_length = self.metadata.get('spectrum_length', 1000)
        self.image_size = self.metadata.get('image_size', (128, 128))

    def preprocess_spectrum(self, spectrum):
        """
        Preprocesa un espectro para la CNN 1D.

        El espectro debe:
        1. Estar normalizado (continuo en ~1.0)
        2. Tener la misma longitud que los datos de entrenamiento

        Si la longitud es diferente, se interpola automaticamente.

        Parameters:
            spectrum: array de flujo normalizado

        Returns:
            array con shape (1, longitud, 1) para la CNN
        """
        # Interpolar si la longitud es diferente
        if len(spectrum) != self.spectrum_length:
            x_old = np.linspace(0, 1, len(spectrum))
            x_new = np.linspace(0, 1, self.spectrum_length)
            spectrum = np.interp(x_new, x_old, spectrum)

        # Reshape para Conv1D: (batch, timesteps, features)
        return spectrum.reshape(1, -1, 1)

    def preprocess_image(self, image):
        """
        Preprocesa una imagen para la CNN 2D.

        Parameters:
            image: array de imagen o path a archivo PNG

        Returns:
            array con shape (1, height, width, 1) para la CNN
        """
        from PIL import Image as PILImage

        if isinstance(image, str):
            # Es un path a archivo
            img = PILImage.open(image).convert('L')  # Escala de grises
            img = img.resize(self.image_size)
            image = np.array(img) / 255.0  # Normalizar a 0-1

        # Reshape para Conv2D: (batch, height, width, channels)
        return image.reshape(1, self.image_size[0], self.image_size[1], 1)

    def predict(self, spectrum=None, image=None):
        """
        Predice el tipo espectral.

        Parameters:
            spectrum: array de flujo normalizado (para CNN 1D)
            image: array de imagen o path PNG (para CNN 2D)

        Returns:
            dict con:
            - tipo: Tipo espectral predicho
            - confianza: Probabilidad (0-100%)
            - probabilidades: dict con probabilidad de cada clase
            - metodo: 'cnn_1d' o 'cnn_2d'
        """
        if self.model_type == '1d':
            if spectrum is None:
                raise ValueError("Debe proporcionar spectrum para CNN 1D")
            X = self.preprocess_spectrum(spectrum)
        else:
            if image is None:
                raise ValueError("Debe proporcionar image para CNN 2D")
            X = self.preprocess_image(image)

        # Predecir (verbose=0 para no mostrar barra de progreso)
        proba = self.model.predict(X, verbose=0)[0]
        pred_idx = np.argmax(proba)

        # Obtener nombre de la clase
        if self.classes:
            tipo = self.classes[pred_idx]
        else:
            tipo = str(pred_idx)

        # Construir diccionario de probabilidades
        probabilidades = {}
        if self.classes:
            for i, clase in enumerate(self.classes):
                probabilidades[clase] = float(proba[i] * 100)
        else:
            for i, p in enumerate(proba):
                probabilidades[str(i)] = float(p * 100)

        return {
            'tipo': tipo,
            'confianza': float(proba[pred_idx] * 100),
            'probabilidades': probabilidades,
            'metodo': f'cnn_{self.model_type}'
        }

    def get_info(self):
        """Retorna informacion del modelo."""
        return {
            'tipo': f'cnn_{self.model_type}',
            'clases': self.classes,
            'spectrum_length': self.spectrum_length if self.model_type == '1d' else None,
            'image_size': self.image_size if self.model_type == '2d' else None,
            'accuracy': self.metadata.get('accuracy_test', None)
        }


# ============================================================================
# MANAGER DE CLASIFICADORES
# ============================================================================

class NeuralClassifierManager:
    """
    Manager para cargar y gestionar multiples clasificadores neuronales.

    Detecta automaticamente que modelos estan disponibles en el directorio
    y permite usarlos de forma unificada.

    EJEMPLO DE USO:
    ---------------
        manager = NeuralClassifierManager(models_dir='models/')

        # Ver modelos disponibles
        print(manager.get_available_models())  # ['knn', 'cnn_1d']

        # Verificar si existe un modelo
        if manager.has_model('knn'):
            resultado = manager.predict('knn', measurements=mediciones)

        # Obtener el mejor modelo disponible
        model_type, classifier = manager.get_best_available()
    """

    def __init__(self, models_dir='models'):
        """
        Inicializa el manager y carga modelos disponibles.

        Parameters:
            models_dir: Directorio donde buscar archivos de modelos
        """
        self.models_dir = models_dir
        self.classifiers = {}

        self._load_available_models()

    def _load_available_models(self):
        """Busca y carga todos los modelos disponibles en el directorio."""
        if not os.path.exists(self.models_dir):
            return

        # Buscar modelo KNN (.pkl)
        knn_path = os.path.join(self.models_dir, 'knn_model.pkl')
        if os.path.exists(knn_path):
            try:
                self.classifiers['knn'] = KNNClassifier(knn_path)
                print(f"  [OK] KNN cargado desde {knn_path}")
            except Exception as e:
                print(f"  [X] Error cargando KNN: {e}")

        # Buscar modelo CNN 1D (.h5)
        cnn_1d_paths = [
            os.path.join(self.models_dir, 'cnn_model.h5'),
            os.path.join(self.models_dir, 'cnn_1d_model.h5'),
            os.path.join(self.models_dir, 'cnn_model.keras'),
        ]
        for path in cnn_1d_paths:
            if os.path.exists(path):
                try:
                    self.classifiers['cnn_1d'] = CNNClassifier(path, model_type='1d')
                    print(f"  [OK] CNN 1D cargado desde {path}")
                    break
                except Exception as e:
                    print(f"  [X] Error cargando CNN 1D: {e}")

        # Buscar modelo CNN 2D (.h5)
        cnn_2d_paths = [
            os.path.join(self.models_dir, 'cnn_2d_model.h5'),
            os.path.join(self.models_dir, 'cnn_2d_model.keras'),
        ]
        for path in cnn_2d_paths:
            if os.path.exists(path):
                try:
                    self.classifiers['cnn_2d'] = CNNClassifier(path, model_type='2d')
                    print(f"  [OK] CNN 2D cargado desde {path}")
                    break
                except Exception as e:
                    print(f"  [X] Error cargando CNN 2D: {e}")

    def get_available_models(self):
        """Retorna lista de modelos disponibles."""
        return list(self.classifiers.keys())

    def has_model(self, model_type):
        """Verifica si un modelo especifico esta disponible."""
        return model_type in self.classifiers

    def predict(self, model_type, **kwargs):
        """
        Realiza prediccion con un modelo especifico.

        Parameters:
            model_type: 'knn', 'cnn_1d', o 'cnn_2d'
            **kwargs: Argumentos para el predictor:
                - measurements: dict de mediciones (para KNN)
                - spectrum: array de flujo (para CNN 1D)
                - image: imagen o path (para CNN 2D)

        Returns:
            dict con resultado de prediccion
        """
        if model_type not in self.classifiers:
            raise ValueError(
                f"Modelo '{model_type}' no disponible.\n"
                f"Modelos disponibles: {self.get_available_models()}"
            )

        return self.classifiers[model_type].predict(**kwargs)

    def get_best_available(self):
        """
        Retorna el mejor clasificador disponible.

        Orden de prioridad:
        1. cnn_1d - Mas preciso para espectros
        2. knn - Rapido y confiable
        3. cnn_2d - Solo si hay imagenes

        Returns:
            (model_type, classifier) o (None, None) si no hay modelos
        """
        priority = ['cnn_1d', 'knn', 'cnn_2d']
        for model_type in priority:
            if model_type in self.classifiers:
                return model_type, self.classifiers[model_type]
        return None, None

    def get_info(self):
        """Retorna informacion de todos los modelos cargados."""
        info = {}
        for name, classifier in self.classifiers.items():
            info[name] = classifier.get_info()
        return info
