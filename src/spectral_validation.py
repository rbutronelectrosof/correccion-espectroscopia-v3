#!/usr/bin/env python3
"""
SISTEMA DE VALIDACIÓN MULTI-MÉTODO
===================================

Implementa sistema de clasificación espectral con múltiples métodos:
1. Clasificador físico (reglas basadas en física espectral)
2. Árbol de decisión (ML entrenado con catálogo)
3. Template matching (comparación con espectros de referencia)
4. Sistema de votación ponderada con métricas de confianza

Uso:
    from spectral_validation import SpectralValidator

    validator = SpectralValidator(models_dir='models/')
    result = validator.classify(wavelengths, flux)

    # result contiene:
    # - tipo_final: Tipo predicho por votación
    # - confianza: Confianza global (0-100%)
    # - alternativas: Top 3 clasificaciones
    # - detalles: Resultados de cada método
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
from collections import defaultdict
from pathlib import Path

# Importar módulo de clasificación física
from spectral_classification_corrected import (
    normalize_to_continuum,
    measure_diagnostic_lines,
    classify_star_corrected,
    SPECTRAL_LINES
)

# Importar clasificadores neuronales (opcional)
try:
    from neural_classifiers import NeuralClassifierManager
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    print("[!] Clasificadores neuronales no disponibles (neural_classifiers.py)")


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

MAIN_TYPES = ['O', 'B', 'A', 'F', 'G', 'K', 'M']

# Pesos para votación ponderada (AJUSTADOS según validación 2026-01-19)
# El clasificador físico tiene 5% accuracy en tipos F/G/K cuando Ca II K no está disponible
# El árbol de decisión tiene 84% accuracy con el catálogo elodie
# Los clasificadores neuronales (KNN/CNN) pueden mejorar la clasificación si están entrenados
DEFAULT_WEIGHTS = {
    'physical': 0.10,  # Clasificador físico - bajo peso por problemas con tipos tardíos
    'decision_tree': 0.40,  # Árbol de decisión (ML) - mejor accuracy con catálogo elodie
    'template_matching': 0.10,  # Template matching
    'neural': 0.40  # KNN/CNN - si está disponible (ajustable desde UI)
}

# Pesos cuando neural NO está disponible (se redistribuyen)
DEFAULT_WEIGHTS_NO_NEURAL = {
    'physical': 0.15,
    'decision_tree': 0.70,
    'template_matching': 0.15
}


# ============================================================================
# CLASIFICADOR POR TEMPLATE MATCHING
# ============================================================================

class TemplateMatchingClassifier:
    """
    Clasificador basado en comparación con espectros de referencia.

    Compara el espectro medido con templates de anchos equivalentes
    para cada tipo espectral usando chi-cuadrado reducido.
    """

    def __init__(self, templates_path=None):
        """
        Parameters
        ----------
        templates_path : str, optional
            Ruta al archivo JSON con templates por tipo
            Si None, usar templates default
        """
        self.templates = self._load_templates(templates_path)

    def _load_templates(self, templates_path):
        """Carga templates de referencia"""
        if templates_path and os.path.exists(templates_path):
            with open(templates_path, 'r') as f:
                return json.load(f)
        else:
            # Templates default (valores típicos de la literatura)
            return self._create_default_templates()

    def _create_default_templates(self):
        """Crea templates default basados en valores típicos"""
        templates = {
            'O': {
                'He_II_4686': 1.0, 'He_I_4471': 0.5, 'H_beta': 2.0,
                'Si_IV_4089': 0.3, 'Ca_II_K': 0.0, 'Fe_I_4046': 0.0
            },
            'B': {
                'He_II_4686': 0.0, 'He_I_4471': 1.0, 'H_beta': 3.0,
                'Si_III_4553': 0.5, 'Mg_II_4481': 0.5, 'Ca_II_K': 0.1,
                'Fe_I_4046': 0.0
            },
            'A': {
                'He_I_4471': 0.0, 'H_beta': 9.0, 'H_gamma': 8.0,
                'H_delta': 7.0, 'Ca_II_K': 2.0, 'Fe_I_4046': 0.2
            },
            'F': {
                'H_beta': 4.0, 'H_gamma': 3.5, 'Ca_II_K': 4.0,
                'Ca_I_4227': 0.8, 'Fe_I_4046': 0.5, 'Fe_I_4144': 0.4
            },
            'G': {
                'H_beta': 2.0, 'Ca_II_K': 8.0, 'Ca_I_4227': 1.5,
                'Fe_I_4046': 1.0, 'Fe_I_4144': 0.8, 'Fe_I_4383': 0.7
            },
            'K': {
                'H_beta': 1.0, 'Ca_II_K': 10.0, 'Ca_I_4227': 2.5,
                'Fe_I_4046': 1.5, 'Fe_I_4144': 1.2, 'Fe_I_4383': 1.0
            },
            'M': {
                'H_beta': 0.5, 'Ca_II_K': 8.0, 'Ca_I_4227': 3.0,
                'Fe_I_4046': 2.0, 'Fe_I_4144': 1.8, 'TiO_4955': 1.5
            }
        }
        return templates

    def classify(self, measurements):
        """
        Clasifica espectro comparando con templates.

        Parameters
        ----------
        measurements : dict
            Mediciones de líneas espectrales

        Returns
        -------
        best_type : str
            Tipo espectral con menor chi-cuadrado
        chi2_dict : dict
            Chi-cuadrado reducido para cada tipo
        confidence : float
            Confianza (0-100%)
        """
        chi2_dict = {}

        for stype, template in self.templates.items():
            chi2 = 0.0
            n_lines = 0

            for line_name, template_ew in template.items():
                measured_ew = measurements.get(line_name, {}).get('ew', 0.0)

                # Error estimado (10% del valor + 0.05 Å de ruido)
                error = max(template_ew * 0.1, 0.05)

                # Chi-cuadrado
                chi2 += ((measured_ew - template_ew) / error) ** 2
                n_lines += 1

            # Chi-cuadrado reducido
            chi2_reduced = chi2 / n_lines if n_lines > 0 else 1e10
            chi2_dict[stype] = chi2_reduced

        # Mejor ajuste (menor chi2)
        best_type = min(chi2_dict, key=chi2_dict.get)
        best_chi2 = chi2_dict[best_type]

        # Confianza basada en chi2 reducido
        # chi2 ~ 1 → excelente (100%)
        # chi2 ~ 5 → moderado (50%)
        # chi2 > 10 → malo (< 20%)
        confidence = 100.0 / (1.0 + best_chi2 / 2.0)

        return best_type, chi2_dict, confidence


# ============================================================================
# CLASIFICADOR POR ÁRBOL DE DECISIÓN
# ============================================================================

class DecisionTreeClassifier:
    """
    Clasificador basado en árbol de decisión entrenado.
    """

    def __init__(self, model_path=None):
        """
        Parameters
        ----------
        model_path : str, optional
            Ruta al modelo pickle entrenado
            Si None, buscar en models/decision_tree.pkl
        """
        if model_path is None:
            model_path = 'models/decision_tree.pkl'

        self.model = self._load_model(model_path)
        self.is_trained = (self.model is not None)

    def _load_model(self, model_path):
        """Carga modelo entrenado"""
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"[!] Modelo no encontrado: {model_path}")
            print("   Ejecutar: python train_and_validate.py --catalog elodie/")
            return None

    def _extract_features(self, measurements):
        """Extrae features en el orden esperado por el modelo"""
        # Features en el mismo orden que train_and_validate.py
        EW_FEATURES = [
            'He_II_4686', 'He_I_4471', 'H_beta', 'H_gamma', 'H_delta', 'H_epsilon',
            'Si_IV_4089', 'Si_III_4553', 'Si_II_4128', 'Mg_II_4481',
            'Ca_II_K', 'Ca_II_H', 'Ca_I_4227',
            'Fe_I_4046', 'Fe_I_4144', 'Fe_I_4383', 'Fe_I_4957'
        ]

        features = []

        # EWs
        for line_name in EW_FEATURES:
            ew = measurements.get(line_name, {}).get('ew', 0.0)
            features.append(ew)

        # Ratios
        He_II_4686 = measurements.get('He_II_4686', {}).get('ew', 0.0)
        He_I_4471 = measurements.get('He_I_4471', {}).get('ew', 0.0)
        H_beta = measurements.get('H_beta', {}).get('ew', 0.0)
        H_gamma = measurements.get('H_gamma', {}).get('ew', 0.0)
        H_delta = measurements.get('H_delta', {}).get('ew', 0.0)
        H_epsilon = measurements.get('H_epsilon', {}).get('ew', 0.0)
        Si_III = measurements.get('Si_III_4553', {}).get('ew', 0.0)
        Si_II = measurements.get('Si_II_4128', {}).get('ew', 0.0)
        Ca_K = measurements.get('Ca_II_K', {}).get('ew', 0.0)
        Mg_II = measurements.get('Mg_II_4481', {}).get('ew', 0.0)
        Ca_I = measurements.get('Ca_I_4227', {}).get('ew', 0.0)
        Fe_I_4046 = measurements.get('Fe_I_4046', {}).get('ew', 0.0)
        Fe_I_4144 = measurements.get('Fe_I_4144', {}).get('ew', 0.0)
        Fe_I_4383 = measurements.get('Fe_I_4383', {}).get('ew', 0.0)

        H_avg = (H_beta + H_gamma + H_delta) / 3.0 if (H_beta + H_gamma + H_delta) > 0 else 0.01
        Fe_I_avg = (Fe_I_4046 + Fe_I_4144 + Fe_I_4383) / 3.0

        features.append(He_I_4471 / (He_II_4686 + 0.01))  # ratio_He_I_He_II
        features.append(Si_III / (Si_II + 0.01))  # ratio_Si_III_Si_II
        features.append(Ca_K / (H_epsilon + 0.01))  # ratio_Ca_K_H_epsilon
        features.append(Mg_II / (He_I_4471 + 0.01))  # ratio_Mg_He_I
        features.append(Fe_I_avg / H_avg)  # ratio_Fe_I_avg_H_avg
        features.append(Ca_I / (H_gamma + 0.01))  # ratio_Ca_I_H_gamma

        # H_avg
        features.append(H_avg)

        return np.array(features).reshape(1, -1)

    def classify(self, measurements):
        """
        Clasifica espectro usando árbol de decisión.

        Returns
        -------
        predicted_type : str
            Tipo predicho
        probabilities : dict
            Probabilidades para cada tipo
        confidence : float
            Confianza (0-100%)
        """
        if not self.is_trained:
            return None, {}, 0.0

        # Extraer features
        X = self._extract_features(measurements)

        # Predecir
        predicted_type = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]

        # Diccionario de probabilidades
        prob_dict = {cls: prob * 100 for cls, prob in zip(self.model.classes_, probs)}

        # Confianza = probabilidad del tipo predicho
        confidence = prob_dict[predicted_type]

        return predicted_type, prob_dict, confidence


# ============================================================================
# VALIDADOR MULTI-MÉTODO
# ============================================================================

class SpectralValidator:
    """
    Sistema de validación multi-método con votación ponderada.

    Combina 3 clasificadores:
    1. Físico (reglas espectroscópicas)
    2. Árbol de decisión (ML)
    3. Template matching

    Y genera clasificación final por votación ponderada con métricas de confianza.
    """

    def __init__(self, models_dir='models/', templates_path=None, weights=None,
                 use_neural=True, preferred_neural='auto'):
        """
        Parameters
        ----------
        models_dir : str
            Directorio con modelos entrenados
        templates_path : str, optional
            Ruta a templates JSON
        weights : dict, optional
            Pesos personalizados para votación
        use_neural : bool
            Si intentar cargar clasificadores neuronales (KNN/CNN)
        preferred_neural : str
            'auto' = elegir el de mayor accuracy disponible
            'knn'  = forzar KNN
            'cnn_1d' = forzar CNN 1D
        """
        self.models_dir = models_dir

        # Inicializar clasificadores
        self.template_classifier = TemplateMatchingClassifier(templates_path)

        dt_path = os.path.join(models_dir, 'decision_tree.pkl') if models_dir else None
        self.dt_classifier = DecisionTreeClassifier(dt_path)

        # Inicializar clasificadores neuronales
        self.neural_manager = None
        self.neural_type = None  # 'knn', 'cnn_1d', o 'cnn_2d'

        if use_neural and NEURAL_AVAILABLE:
            try:
                self.neural_manager = NeuralClassifierManager(models_dir)
                available_models = self.neural_manager.get_available_models()
                if available_models:
                    if preferred_neural != 'auto' and preferred_neural in available_models:
                        # Usar el modelo que eligió el usuario
                        self.neural_type = preferred_neural
                    else:
                        # Elegir automáticamente el de mayor accuracy
                        self.neural_type, _ = self.neural_manager.get_best_available()
            except Exception as e:
                print(f"[!] Error cargando clasificadores neuronales: {e}")
                self.neural_manager = None

        # Ajustar pesos según disponibilidad de neural
        if weights:
            self.weights = weights
        elif self.neural_manager and self.neural_manager.get_available_models():
            self.weights = DEFAULT_WEIGHTS.copy()
        else:
            self.weights = DEFAULT_WEIGHTS_NO_NEURAL.copy()

        print(f"[OK] SpectralValidator inicializado")
        print(f"   - Clasificador físico: Activo")
        print(f"   - Árbol de decisión: {'Activo' if self.dt_classifier.is_trained else 'Inactivo (modelo no encontrado)'}")
        print(f"   - Template matching: Activo")
        if self.neural_manager and self.neural_manager.get_available_models():
            models = self.neural_manager.get_available_models()
            print(f"   - Neural ({self.neural_type}): Activo ({', '.join(models)} disponibles)")
        else:
            print(f"   - Neural: Inactivo (entrena modelos KNN/CNN en pestaña Redes Neuronales)")

    def classify_physical(self, wavelengths, flux_normalized, measurements):
        """
        Clasificación usando método físico.

        Returns
        -------
        spectral_type : str
        subtype : str
        confidence : float
        diagnostics : dict
        """
        spectral_type, subtype, diagnostics = classify_star_corrected(
            measurements, wavelengths, flux_normalized
        )

        # Estimar confianza basada en fuerza de líneas diagnóstico
        confidence = self._estimate_physical_confidence(
            spectral_type, measurements, diagnostics
        )

        return spectral_type, subtype, confidence, diagnostics

    def _estimate_physical_confidence(self, spectral_type, measurements, diagnostics):
        """
        Estima confianza del clasificador físico basada en:
        - Fuerza de líneas diagnóstico
        - Presencia de líneas esperadas
        - Ausencia de características contradictorias
        """
        confidence = 100.0

        # Extraer EWs relevantes
        He_II = measurements.get('He_II_4686', {}).get('ew', 0.0)
        He_I = measurements.get('He_I_4471', {}).get('ew', 0.0)
        H_avg = (measurements.get('H_beta', {}).get('ew', 0.0) +
                 measurements.get('H_gamma', {}).get('ew', 0.0) +
                 measurements.get('H_delta', {}).get('ew', 0.0)) / 3.0
        Ca_K = measurements.get('Ca_II_K', {}).get('ew', 0.0)
        Fe_I = measurements.get('Fe_I_4046', {}).get('ew', 0.0)

        # Criterios específicos por tipo
        if spectral_type == 'O':
            # O debe tener He II fuerte
            if He_II < 0.1:
                confidence -= 30.0  # He II muy débil para O
            if He_II > 0.5:
                confidence += 0.0  # He II fuerte, excelente
            elif He_II > 0.2:
                confidence -= 10.0  # He II moderado, OK
            else:
                confidence -= 20.0  # He II débil, dudoso

        elif spectral_type == 'B':
            # B debe tener He I fuerte, He II ausente
            if He_I < 0.2:
                confidence -= 25.0  # He I débil para B
            if He_II > 0.1:
                confidence -= 30.0  # He II presente → tal vez O tardío

        elif spectral_type == 'A':
            # A debe tener H fuerte
            if H_avg < 4.0:
                confidence -= 30.0  # H débil para A
            if He_I > 0.1:
                confidence -= 40.0  # He I presente → probablemente B

        elif spectral_type in ['F', 'G', 'K']:
            # F-G-K deben tener metales
            if Ca_K < 1.0 and Fe_I < 0.3:
                confidence -= 35.0  # Metales muy débiles
            if H_avg > 6.0:
                confidence -= 30.0  # H muy fuerte → probablemente A

        elif spectral_type == 'M':
            # M debe tener metales muy fuertes y/o TiO
            if Fe_I < 0.5:
                confidence -= 25.0  # Metales débiles para M

        # Penalización si Ca II K está fuera del rango espectral
        if Ca_K < 0.05 and spectral_type in ['A', 'F', 'G', 'K']:
            confidence -= 15.0  # Sin Ca II K reduce confianza en A-K

        # Limitar a [0, 100]
        confidence = max(0.0, min(100.0, confidence))

        return confidence

    def classify(self, wavelengths, flux, verbose=False):
        """
        Clasificación multi-método con votación ponderada.

        Parameters
        ----------
        wavelengths : array
            Longitudes de onda (Å)
        flux : array
            Flujo (unidades arbitrarias)
        verbose : bool
            Mostrar detalles de clasificación

        Returns
        -------
        result : dict
            Diccionario con:
            - tipo_final: Tipo por votación
            - subtipo_final: Subtipo del método físico
            - confianza: Confianza global (0-100%)
            - alternativas: Top 3 tipos con justificación
            - detalles: Resultados de cada método
        """
        # 1. Normalizar
        flux_normalized, continuum = normalize_to_continuum(wavelengths, flux)

        # 2. Medir líneas
        measurements = measure_diagnostic_lines(wavelengths, flux_normalized)

        # 3. Clasificar con cada método
        results = {}

        # Método físico
        tipo_phys, subtipo_phys, conf_phys, diag_phys = self.classify_physical(
            wavelengths, flux_normalized, measurements
        )
        results['physical'] = {
            'tipo': tipo_phys,
            'subtipo': subtipo_phys,
            'confianza': conf_phys,
            'diagnostics': diag_phys
        }

        # Árbol de decisión
        if self.dt_classifier.is_trained:
            tipo_dt, prob_dt, conf_dt = self.dt_classifier.classify(measurements)
            results['decision_tree'] = {
                'tipo': tipo_dt,
                'probabilidades': prob_dt,
                'confianza': conf_dt
            }
        else:
            results['decision_tree'] = None

        # Template matching
        tipo_tm, chi2_tm, conf_tm = self.template_classifier.classify(measurements)
        results['template_matching'] = {
            'tipo': tipo_tm,
            'chi2': chi2_tm,
            'confianza': conf_tm
        }

        # Clasificadores neuronales (KNN/CNN)
        if self.neural_manager and self.neural_type:
            try:
                model_type = self.neural_type

                if model_type == 'knn' and self.neural_manager.has_model('knn'):
                    # KNN usa features extraídas de las mediciones
                    neural_result = self.neural_manager.predict('knn', measurements=measurements)
                elif model_type == 'cnn_1d' and self.neural_manager.has_model('cnn_1d'):
                    # CNN 1D usa el espectro normalizado directamente
                    neural_result = self.neural_manager.predict('cnn_1d', spectrum=flux_normalized)
                else:
                    # CNN 2D requeriría una imagen (no aplicable en este contexto)
                    neural_result = None

                if neural_result:
                    results['neural'] = {
                        'tipo': neural_result['tipo'],
                        'confianza': neural_result['confianza'],
                        'probabilidades': neural_result.get('probabilidades', {}),
                        'metodo': neural_result.get('metodo', model_type)
                    }
                else:
                    results['neural'] = None

            except Exception as e:
                print(f"[!] Error en clasificacion neural: {e}")
                results['neural'] = None
        else:
            results['neural'] = None

        # 4. Votación ponderada
        tipo_final, alternativas, confianza_global = self._weighted_vote(results)

        # 5. Subtipo coherente con el tipo ganador
        if tipo_phys == tipo_final:
            subtipo_coherente = subtipo_phys
        else:
            subtipo_coherente = self._estimate_subtype(tipo_final, measurements)

        # 6. Añadir subtipo a cada alternativa
        for alt in alternativas:
            t = alt['tipo']
            if t == tipo_phys:
                alt['subtipo'] = subtipo_phys
            elif t == tipo_final:
                alt['subtipo'] = subtipo_coherente
            else:
                alt['subtipo'] = self._estimate_subtype(t, measurements)

        # 7. Generar resultado
        result = {
            'tipo_final':     tipo_final,
            'subtipo_final':  subtipo_coherente,
            'tipo_fisico':    tipo_phys,
            'subtipo_fisico': subtipo_phys,
            'confianza':      confianza_global,
            'alternativas':   alternativas,
            'detalles':       results,
            'measurements':   measurements
        }

        if verbose:
            self._print_classification_details(result)

        return result

    def _estimate_subtype(self, tipo_final, measurements):
        """
        Estima un subtipo numérico coherente con tipo_final usando las mediciones.
        Se usa cuando el tipo ganador del voto difiere del tipo físico.

        Returns  str como 'F5', 'K3', 'B2', etc.
        """
        H_avg  = (measurements.get('H_beta',  {}).get('ew', 0.0) +
                  measurements.get('H_gamma', {}).get('ew', 0.0) +
                  measurements.get('H_delta', {}).get('ew', 0.0)) / 3.0
        Ca_K   = measurements.get('Ca_II_K',   {}).get('ew', 0.0)
        Fe_avg = (measurements.get('Fe_I_4046', {}).get('ew', 0.0) +
                  measurements.get('Fe_I_4383', {}).get('ew', 0.0)) / 2.0
        He_I   = measurements.get('He_I_4471',  {}).get('ew', 0.0)
        He_II  = measurements.get('He_II_4686', {}).get('ew', 0.0)
        Mg_II  = measurements.get('Mg_II_4481', {}).get('ew', 0.0)
        TiO    = measurements.get('TiO_4955',   {}).get('ew', 0.0)

        if tipo_final == 'O':
            ratio = He_I / (He_II + 0.01)
            if ratio < 0.5:  return 'O5'
            elif ratio < 1.0: return 'O7'
            else:             return 'O9'

        elif tipo_final == 'B':
            if Mg_II < 0.05:            return 'B2'
            ratio = Mg_II / (He_I + 0.01)
            if ratio < 1.5:   return 'B6'
            elif ratio < 3.0: return 'B8'
            else:             return 'B9'

        elif tipo_final == 'A':
            if H_avg > 7.0:   return 'A1'
            elif H_avg > 5.5: return 'A3'
            else:             return 'A7'

        elif tipo_final == 'F':
            ratio = H_avg / (Fe_avg + 0.01)
            if ratio > 5:   return 'F2'
            elif ratio > 2: return 'F5'
            else:           return 'F8'

        elif tipo_final == 'G':
            ratio = H_avg / (Ca_K + 0.01)
            if ratio > 3:   return 'G0'
            elif ratio > 1.5: return 'G5'
            else:           return 'G8'

        elif tipo_final == 'K':
            if Ca_K > 8:   return 'K5'
            elif Ca_K > 5: return 'K3'
            else:          return 'K0'

        elif tipo_final == 'M':
            if TiO > 1.0:   return 'M5'
            elif TiO > 0.5: return 'M2'
            else:           return 'M0'

        return tipo_final

    def _weighted_vote(self, results):
        """
        Votación ponderada entre métodos.

        Returns
        -------
        tipo_final : str
        alternativas : list
            Top 3 con justificación
        confianza : float
        """
        # Acumular votos ponderados
        votes = defaultdict(float)
        confidences = defaultdict(list)

        # Votos del clasificador físico
        if 'physical' in results and results['physical']:
            tipo = results['physical']['tipo']
            conf = results['physical']['confianza']
            weight = self.weights['physical']
            votes[tipo] += weight * (conf / 100.0)
            confidences[tipo].append(conf)

        # Votos del árbol de decisión
        if results.get('decision_tree'):
            tipo = results['decision_tree']['tipo']
            conf = results['decision_tree']['confianza']
            weight = self.weights['decision_tree']
            votes[tipo] += weight * (conf / 100.0)
            confidences[tipo].append(conf)

        # Votos de template matching
        if 'template_matching' in results and results['template_matching']:
            tipo = results['template_matching']['tipo']
            conf = results['template_matching']['confianza']
            weight = self.weights['template_matching']
            votes[tipo] += weight * (conf / 100.0)
            confidences[tipo].append(conf)

        # Votos de clasificadores neuronales (KNN/CNN)
        if results.get('neural') and 'neural' in self.weights:
            tipo = results['neural']['tipo']
            conf = results['neural']['confianza']
            weight = self.weights['neural']
            votes[tipo] += weight * (conf / 100.0)
            confidences[tipo].append(conf)

        # Ordenar por votos
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)

        # Tipo con más votos
        tipo_final = sorted_votes[0][0] if sorted_votes else 'Unknown'

        # Top 3 alternativas
        alternativas = []
        for i, (tipo, vote_score) in enumerate(sorted_votes[:3]):
            # Confianza promedio de los métodos que votaron por este tipo
            avg_conf = np.mean(confidences[tipo]) if confidences[tipo] else 0.0

            # Justificación
            justification = self._generate_justification(tipo, results)

            alternativas.append({
                'tipo': tipo,
                'confianza': avg_conf,
                'votos_ponderados': vote_score * 100,
                'justificacion': justification
            })

        # Confianza global basada en consenso
        if len(sorted_votes) >= 2:
            # Si hay consenso (top voto >> segundo voto), alta confianza
            ratio = sorted_votes[0][1] / (sorted_votes[1][1] + 0.01)
            if ratio > 2.0:
                confianza_global = min(95.0, alternativas[0]['confianza'] * 1.1)
            else:
                confianza_global = alternativas[0]['confianza'] * 0.85
        else:
            confianza_global = alternativas[0]['confianza'] if alternativas else 0.0

        return tipo_final, alternativas, confianza_global

    def _generate_justification(self, tipo, results):
        """Genera justificación textual para clasificación"""
        justifications = []

        # Físico
        if results.get('physical') and results['physical']['tipo'] == tipo:
            justifications.append("Clasificador físico")

        # ML
        if results.get('decision_tree') and results['decision_tree']['tipo'] == tipo:
            prob = results['decision_tree']['probabilidades'].get(tipo, 0)
            justifications.append(f"ML ({prob:.0f}% prob.)")

        # Template
        if results.get('template_matching') and results['template_matching']['tipo'] == tipo:
            chi2 = results['template_matching']['chi2'].get(tipo, 999)
            justifications.append(f"Template (χ²={chi2:.1f})")

        # Neural (KNN/CNN)
        if results.get('neural') and results['neural']['tipo'] == tipo:
            metodo = results['neural'].get('metodo', 'neural')
            conf = results['neural']['confianza']
            justifications.append(f"{metodo.upper()} ({conf:.0f}%)")

        if not justifications:
            return "Clasificación alternativa"

        return ", ".join(justifications)

    def _print_classification_details(self, result):
        """Imprime detalles de clasificación (modo verbose)"""
        print(f"\n{'='*70}")
        print("CLASIFICACIÓN MULTI-MÉTODO")
        print(f"{'='*70}")

        print(f"\n>>> RESULTADO FINAL:")
        print(f"   Tipo: {result['tipo_final']}")
        print(f"   Subtipo: {result['subtipo_final']}")
        print(f"   Confianza: {result['confianza']:.1f}%")

        print(f"\n--- ALTERNATIVAS:")
        for i, alt in enumerate(result['alternativas'], 1):
            print(f"   {i}. {alt['tipo']:2s} - {alt['confianza']:.1f}% - {alt['justificacion']}")

        print(f"\n🔬 DETALLES POR MÉTODO:")

        # Físico
        phys = result['detalles'].get('physical')
        if phys:
            print(f"   Físico: {phys['tipo']} ({phys['subtipo']}) - {phys['confianza']:.1f}%")

        # ML
        dt = result['detalles'].get('decision_tree')
        if dt:
            print(f"   ML:     {dt['tipo']} - {dt['confianza']:.1f}%")

        # Template
        tm = result['detalles'].get('template_matching')
        if tm:
            print(f"   Template: {tm['tipo']} - {tm['confianza']:.1f}%")

        # Neural
        nn = result['detalles'].get('neural')
        if nn:
            print(f"   Neural ({nn.get('metodo', 'knn')}): {nn['tipo']} - {nn['confianza']:.1f}%")

        print(f"{'='*70}\n")


# ============================================================================
# FUNCIÓN DE CONVENIENCIA
# ============================================================================

def validate_spectrum(wavelengths, flux, models_dir='models/', verbose=False):
    """
    Función de conveniencia para clasificar un espectro.

    Parameters
    ----------
    wavelengths : array
    flux : array
    models_dir : str
    verbose : bool

    Returns
    -------
    result : dict
        Resultado de clasificación
    """
    validator = SpectralValidator(models_dir=models_dir)
    return validator.classify(wavelengths, flux, verbose=verbose)


# ============================================================================
# MAIN (TESTING)
# ============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Uso: python spectral_validation.py <spectrum.txt>")
        sys.exit(1)

    spectrum_file = sys.argv[1]

    # Cargar espectro
    data = np.loadtxt(spectrum_file, delimiter=',', skiprows=1)
    wavelengths = data[:, 0]
    flux = data[:, 1]

    # Clasificar
    result = validate_spectrum(wavelengths, flux, verbose=True)

    print("\n[OK] Clasificacion completada")
    print(f"   Archivo: {os.path.basename(spectrum_file)}")
    print(f"   Tipo: {result['tipo_final']} ({result['subtipo_final']})")
    print(f"   Confianza: {result['confianza']:.1f}%")
