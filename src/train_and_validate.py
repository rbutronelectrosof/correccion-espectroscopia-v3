#!/usr/bin/env python3
"""
ENTRENAMIENTO Y VALIDACIÓN - Clasificación Espectral
=====================================================

Script para entrenar y validar modelos de clasificación espectral:
- Split 80/20 del catálogo etiquetado
- Entrenamiento de árbol de decisión
- Validación cruzada k-fold
- Optimización de umbrales
- Generación de métricas y reportes

Uso:
    python train_and_validate.py --catalog elodie/ --output-dir models/

    # Solo validar (sin reentrenar)
    python train_and_validate.py --catalog elodie/ --validate-only

    # Optimizar umbrales del clasificador físico
    python train_and_validate.py --catalog elodie/ --optimize-thresholds
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Configurar matplotlib para modo sin interfaz grafica (evita errores en servidores)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import pickle
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Asegurar que el directorio src/ esté en el path para imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Importar módulo de clasificación
from spectral_classification_corrected import (
    normalize_to_continuum,
    measure_diagnostic_lines,
    classify_star_corrected,
    SPECTRAL_LINES
)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Tipos espectrales principales y subtipos
MAIN_TYPES = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
SPECTRAL_TYPE_ORDER = MAIN_TYPES  # Para ordenar en matriz de confusión

# Features a extraer (anchos equivalentes + ratios diagnósticos)
EW_FEATURES = [
    'He_II_4686', 'He_I_4471', 'H_beta', 'H_gamma', 'H_delta', 'H_epsilon',
    'Si_IV_4089', 'Si_III_4553', 'Si_II_4128', 'Mg_II_4481',
    'Ca_II_K', 'Ca_II_H', 'Ca_I_4227',
    'Fe_I_4046', 'Fe_I_4144', 'Fe_I_4383', 'Fe_I_4957'
]

RATIO_FEATURES = [
    'He_I_He_II', 'Si_III_Si_II', 'Ca_K_H_epsilon', 'Mg_He_I',
    'Fe_I_avg_H_avg', 'Ca_I_H_gamma'
]


# ============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================================

def extract_spectral_type_from_filename(filename):
    """
    Extrae el tipo espectral del nombre del archivo.

    Ejemplos:
        HD015570_tipoO4.txt → O
        BD+023375_tipoA5.txt → A
        HD000400_tipo_F8IV.txt → F

    Returns
    -------
    main_type : str
        Tipo espectral principal (O, B, A, F, G, K, M)
    subtype : str
        Subtipo completo (ej: 'O4', 'A5', 'F8IV')
    """
    if '_tipo' not in filename and '_tipo_' not in filename:
        return None, None

    # Extraer parte después de "tipo"
    if '_tipo_' in filename:
        parts = filename.split('_tipo_')
    else:
        parts = filename.split('_tipo')

    if len(parts) < 2:
        return None, None

    # Limpiar subtipo (quitar .txt, espacios, etc.)
    subtype_raw = parts[1].replace('.txt', '').replace('....', '').strip()

    # Extraer tipo principal (primera letra)
    if len(subtype_raw) == 0:
        return None, None

    main_type = subtype_raw[0].upper()

    if main_type not in MAIN_TYPES:
        return None, None

    return main_type, subtype_raw


def load_spectrum_txt(filepath):
    """
    Carga espectro desde archivo .txt (formato 2 columnas).
    Maneja múltiples formatos: CSV, TSV, espacios, con o sin header.
    """
    try:
        # Intentar con pandas primero (más robusto con encodings y formatos)
        try:
            # Intentar CSV con header
            df = pd.read_csv(filepath, encoding='utf-8')
            if df.shape[1] >= 2:
                wavelengths = df.iloc[:, 0].values.astype(float)
                flux = df.iloc[:, 1].values.astype(float)
                return wavelengths, flux
        except:
            pass

        try:
            # Intentar CSV con encoding latin-1
            df = pd.read_csv(filepath, encoding='latin-1')
            if df.shape[1] >= 2:
                wavelengths = df.iloc[:, 0].values.astype(float)
                flux = df.iloc[:, 1].values.astype(float)
                return wavelengths, flux
        except:
            pass

        try:
            # Intentar sin header
            df = pd.read_csv(filepath, header=None, encoding='utf-8')
            if df.shape[1] >= 2:
                # Verificar si la primera fila es numérica
                try:
                    float(df.iloc[0, 0])
                    wavelengths = df.iloc[:, 0].values.astype(float)
                    flux = df.iloc[:, 1].values.astype(float)
                except:
                    # La primera fila es header, saltar
                    wavelengths = df.iloc[1:, 0].values.astype(float)
                    flux = df.iloc[1:, 1].values.astype(float)
                return wavelengths, flux
        except:
            pass

        # Último intento: numpy con diferentes delimitadores
        for delimiter in [',', '\t', ' ', ';']:
            for skip in [0, 1]:
                try:
                    data = np.loadtxt(filepath, delimiter=delimiter, skiprows=skip)
                    if len(data.shape) == 2 and data.shape[1] >= 2:
                        return data[:, 0], data[:, 1]
                except:
                    continue

        print(f"[!] No se pudo cargar {os.path.basename(filepath)}: formato no reconocido")
        return None, None

    except Exception as e:
        print(f"[!] Error cargando {os.path.basename(filepath)}: {e}")
        return None, None


def extract_features(wavelengths, flux, measurements):
    """
    Extrae features (EWs + ratios) de un espectro.

    Returns
    -------
    features : dict
        Diccionario con features extraídos
    """
    features = {}

    # 1. Anchos equivalentes directos
    for line_name in EW_FEATURES:
        ew = measurements.get(line_name, {}).get('ew', 0.0)
        features[f'ew_{line_name}'] = ew

    # 2. Ratios diagnósticos
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

    features['ratio_He_I_He_II'] = He_I_4471 / (He_II_4686 + 0.01)
    features['ratio_Si_III_Si_II'] = Si_III / (Si_II + 0.01)
    features['ratio_Ca_K_H_epsilon'] = Ca_K / (H_epsilon + 0.01)
    features['ratio_Mg_He_I'] = Mg_II / (He_I_4471 + 0.01)
    features['ratio_Fe_I_avg_H_avg'] = Fe_I_avg / H_avg
    features['ratio_Ca_I_H_gamma'] = Ca_I / (H_gamma + 0.01)

    # 3. Features adicionales
    features['H_avg'] = H_avg
    features['wavelength_range'] = wavelengths[-1] - wavelengths[0]
    features['n_points'] = len(wavelengths)

    return features


def load_catalog(catalog_dir, max_files=None, verbose=True):
    """
    Carga catálogo completo de espectros etiquetados.

    Parameters
    ----------
    catalog_dir : str
        Directorio con archivos .txt
    max_files : int, optional
        Límite de archivos a procesar (para testing)
    verbose : bool
        Mostrar progreso

    Returns
    -------
    data : pd.DataFrame
        DataFrame con features y etiquetas
    """
    catalog_path = Path(catalog_dir)

    # Verificar que el directorio existe
    if not catalog_path.exists():
        print(f"[ERROR] El directorio no existe: {catalog_dir}")
        print(f"   Ruta absoluta: {catalog_path.absolute()}")
        return pd.DataFrame()

    if not catalog_path.is_dir():
        print(f"[ERROR] La ruta no es un directorio: {catalog_dir}")
        return pd.DataFrame()

    txt_files = sorted(catalog_path.glob("*.txt"))

    if len(txt_files) == 0:
        print(f"[ERROR] No se encontraron archivos .txt en: {catalog_dir}")
        return pd.DataFrame()

    if max_files:
        txt_files = txt_files[:max_files]

    data_rows = []
    errors = []

    if verbose:
        print(f"\n{'='*70}", flush=True)
        print(f"CARGA DE CATALOGO: {catalog_dir}", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"Archivos encontrados: {len(txt_files)}", flush=True)

    for i, filepath in enumerate(txt_files):
        filename = filepath.name

        try:
            # Extraer tipo espectral del nombre
            main_type, subtype = extract_spectral_type_from_filename(filename)

            if main_type is None:
                continue  # Saltar archivos sin tipo en nombre

            # Cargar espectro
            wavelengths, flux = load_spectrum_txt(str(filepath))

            if wavelengths is None or len(wavelengths) == 0:
                errors.append(filename)
                continue

            # Verificar datos válidos
            if len(wavelengths) < 10 or np.any(np.isnan(wavelengths)) or np.any(np.isnan(flux)):
                if verbose and len(errors) < 5:
                    print(f"[!] Datos invalidos en {filename}")
                errors.append(filename)
                continue

            # Normalizar
            try:
                flux_normalized, continuum = normalize_to_continuum(wavelengths, flux)
            except Exception as e:
                if verbose and len(errors) < 10:
                    print(f"[!] Error normalizando {filename}: {e}")
                errors.append(filename)
                continue

            # Medir líneas
            try:
                measurements = measure_diagnostic_lines(wavelengths, flux_normalized)
            except Exception as e:
                if verbose and len(errors) < 10:
                    print(f"[!] Error midiendo lineas {filename}: {e}")
                errors.append(filename)
                continue

            # Extraer features
            features = extract_features(wavelengths, flux, measurements)

            # Agregar metadatos
            features['filename'] = filename
            features['main_type'] = main_type
            features['subtype'] = subtype

            data_rows.append(features)

        except Exception as e:
            if verbose and len(errors) < 10:
                print(f"[!] Error inesperado procesando {filename}: {e}")
            errors.append(filename)
            continue

        if verbose and (i + 1) % 50 == 0:
            print(f"  Procesados: {i + 1}/{len(txt_files)} ({100*(i+1)/len(txt_files):.1f}%)", flush=True)

    df = pd.DataFrame(data_rows)

    if verbose:
        print(f"\n[OK] Espectros procesados exitosamente: {len(df)}", flush=True)
        print(f"[ERROR] Errores: {len(errors)}", flush=True)
        print(f"\nDistribucion por tipo:", flush=True)
        print(df['main_type'].value_counts().sort_index(), flush=True)
        print(f"{'='*70}\n", flush=True)

    return df


# ============================================================================
# ENTRENAMIENTO DE MODELOS
# ============================================================================

def train_decision_tree(X_train, y_train, X_test, y_test, max_depth=10):
    """
    Entrena árbol de decisión para clasificación de tipos espectrales.

    Parameters
    ----------
    X_train, y_train : arrays
        Datos de entrenamiento
    X_test, y_test : arrays
        Datos de prueba
    max_depth : int
        Profundidad máxima del árbol

    Returns
    -------
    model : DecisionTreeClassifier
        Modelo entrenado
    accuracy : float
        Accuracy en conjunto de prueba
    """
    print(f"\n{'='*70}")
    print("ENTRENAMIENTO DE ÁRBOL DE DECISIÓN")
    print(f"{'='*70}")

    # Entrenar
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluar
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    print(f"Accuracy (entrenamiento): {acc_train*100:.2f}%")
    print(f"Accuracy (prueba): {acc_test*100:.2f}%")
    print(f"Profundidad del árbol: {model.get_depth()}")
    print(f"Número de hojas: {model.get_n_leaves()}")

    # Feature importance
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'f{i}' for i in range(X_train.shape[1])]
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 features más importantes:")
    print(importances.head(10).to_string(index=False))
    print(f"{'='*70}\n")

    return model, acc_test, importances


def cross_validate_model(X, y, model, k=5):
    """
    Validación cruzada k-fold.

    Returns
    -------
    scores : array
        Accuracy en cada fold
    mean_score : float
        Accuracy promedio
    """
    print(f"\n{'='*70}")
    print(f"VALIDACIÓN CRUZADA ({k}-FOLD)")
    print(f"{'='*70}")

    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

    print(f"Accuracy por fold: {[f'{s*100:.2f}%' for s in scores]}")
    print(f"Accuracy promedio: {scores.mean()*100:.2f}% (± {scores.std()*100:.2f}%)")
    print(f"{'='*70}\n")

    return scores, scores.mean()


# ============================================================================
# VALIDACIÓN DEL CLASIFICADOR FÍSICO
# ============================================================================

def validate_physical_classifier(df, catalog_dir=None):
    """
    Valida el clasificador físico actual con el catálogo.

    Returns
    -------
    accuracy : float
        Accuracy global
    results : pd.DataFrame
        Resultados detallados
    """
    print(f"\n{'='*70}")
    print("VALIDACIÓN DEL CLASIFICADOR FÍSICO")
    print(f"{'='*70}")

    correct = 0
    total = 0
    results_list = []

    for idx, row in df.iterrows():
        filename = row['filename']
        true_type = row['main_type']

        # Cargar espectro completo (necesario para clasificar)
        if catalog_dir is None:
            catalog_dir = "elodie/"
        filepath = os.path.join(catalog_dir, filename)

        wavelengths, flux = load_spectrum_txt(filepath)
        if wavelengths is None:
            continue

        flux_normalized, _ = normalize_to_continuum(wavelengths, flux)
        measurements = measure_diagnostic_lines(wavelengths, flux_normalized)

        # Clasificar
        pred_type, pred_subtype, diagnostics = classify_star_corrected(
            measurements, wavelengths, flux_normalized
        )

        # Comparar
        is_correct = (pred_type == true_type)
        if is_correct:
            correct += 1
        total += 1

        results_list.append({
            'filename': filename,
            'true_type': true_type,
            'true_subtype': row['subtype'],
            'pred_type': pred_type,
            'pred_subtype': pred_subtype,
            'correct': is_correct
        })

        if total % 50 == 0:
            print(f"  Validados: {total}/{len(df)} ({100*total/len(df):.1f}%)")

    accuracy = correct / total if total > 0 else 0
    results_df = pd.DataFrame(results_list)

    print(f"\n[OK] Accuracy global: {accuracy*100:.2f}% ({correct}/{total})")

    # Accuracy por tipo
    print(f"\nAccuracy por tipo espectral:")
    for stype in MAIN_TYPES:
        subset = results_df[results_df['true_type'] == stype]
        if len(subset) > 0:
            acc = subset['correct'].mean()
            print(f"  {stype}: {acc*100:.2f}% ({subset['correct'].sum()}/{len(subset)})")

    print(f"{'='*70}\n")

    return accuracy, results_df


# ============================================================================
# VISUALIZACIÓN
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """Genera matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Número de espectros'})
    plt.xlabel('Tipo Predicho', fontsize=12)
    plt.ylabel('Tipo Verdadero', fontsize=12)
    plt.title('Matriz de Confusión - Clasificación Espectral', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Matriz de confusion guardada: {output_path}")
    plt.close()


def plot_decision_tree_diagram(model, feature_names, output_path, max_depth=3):
    """Genera diagrama del árbol de decisión"""
    plt.figure(figsize=(20, 10))
    plot_tree(model,
              feature_names=feature_names,
              class_names=MAIN_TYPES,
              filled=True,
              rounded=True,
              fontsize=8,
              max_depth=max_depth)
    plt.title(f'Árbol de Decisión (primeros {max_depth} niveles)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Diagrama del arbol guardado: {output_path}")
    plt.close()


def generate_report(df, results_physical, model_dt, acc_dt, importances, output_dir):
    """Genera reporte completo de validación"""
    report_path = os.path.join(output_dir, 'validation_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("REPORTE DE VALIDACIÓN - CLASIFICACIÓN ESPECTRAL\n")
        f.write("="*70 + "\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Catálogo: {len(df)} espectros\n\n")

        f.write("DISTRIBUCIÓN POR TIPO:\n")
        f.write(df['main_type'].value_counts().sort_index().to_string() + "\n\n")

        f.write("CLASIFICADOR FÍSICO:\n")
        if len(results_physical) > 0 and 'correct' in results_physical.columns:
            f.write(f"Accuracy global: {results_physical['correct'].mean()*100:.2f}%\n")
            f.write("\nAccuracy por tipo:\n")
            for stype in MAIN_TYPES:
                subset = results_physical[results_physical['true_type'] == stype]
                if len(subset) > 0:
                    acc = subset['correct'].mean()
                    f.write(f"  {stype}: {acc*100:.2f}% ({subset['correct'].sum()}/{len(subset)})\n")
        else:
            f.write("No se pudieron validar espectros del clasificador físico\n")

        model_name = type(model_dt).__name__
        f.write(f"\nMODELO ({model_name}):\n")
        f.write(f"Accuracy: {acc_dt*100:.2f}%\n")
        if hasattr(model_dt, 'get_depth'):
            f.write(f"Profundidad: {model_dt.get_depth()}\n")
        if hasattr(model_dt, 'get_n_leaves'):
            f.write(f"Numero de hojas: {model_dt.get_n_leaves()}\n")
        if hasattr(model_dt, 'n_estimators'):
            f.write(f"Estimadores: {model_dt.n_estimators}\n")
        if hasattr(model_dt, 'max_depth') and model_dt.max_depth is not None:
            f.write(f"Profundidad maxima: {model_dt.max_depth}\n")
        f.write("\n")

        f.write("TOP 10 FEATURES MÁS IMPORTANTES:\n")
        f.write(importances.head(10).to_string(index=False) + "\n\n")

        f.write("="*70 + "\n")

    print(f"[OK] Reporte guardado: {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Forzar line-buffering en stdout (necesario cuando Python corre en pipe en Windows)
    sys.stdout.reconfigure(line_buffering=True)
    print("=== Modulos cargados. Preparando entrenamiento... ===", flush=True)

    parser = argparse.ArgumentParser(description='Entrenar y validar clasificadores espectrales')
    parser.add_argument('--catalog', type=str, default='data/elodie/',
                        help='Directorio del catálogo de espectros')
    parser.add_argument('--output', '--output-dir', type=str, default='models/',
                        help='Directorio de salida para modelos y reportes')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Límite de archivos a procesar (para testing)')
    parser.add_argument('--validate-only', action='store_true',
                        help='Solo validar clasificador físico (no entrenar)')
    parser.add_argument('--max-depth', '--tree-depth', type=int, default=10,
                        help='Profundidad máxima del árbol de decisión')
    parser.add_argument('--k-folds', type=int, default=5,
                        help='Número de folds para validación cruzada')
    parser.add_argument('--model', type=str, default='decision_tree',
                        choices=['decision_tree', 'random_forest', 'gradient_boosting'],
                        help='Tipo de modelo a entrenar')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proporción de datos para test (0.0-1.0)')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Número de estimadores para Random Forest/Gradient Boosting')

    args = parser.parse_args()

    # Compatibilidad con argumentos alias
    args.output_dir = args.output
    args.tree_depth = args.max_depth

    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Catalogo: {args.catalog}  |  Modelo: {args.model}  |  Test: {args.test_size*100:.0f}%", flush=True)

    # 1. Cargar catálogo
    df = load_catalog(args.catalog, max_files=args.max_files)

    if len(df) == 0:
        print("[ERROR] No se pudieron cargar espectros del catalogo")
        return

    # 2. Validar clasificador físico
    acc_physical, results_physical = validate_physical_classifier(df, catalog_dir=args.catalog)

    if args.validate_only:
        print("\n[OK] Validacion completada (solo clasificador fisico)")
        return

    # 3. Preparar datos para ML
    feature_cols = [col for col in df.columns if col.startswith('ew_') or col.startswith('ratio_') or col in ['H_avg']]
    X = df[feature_cols]
    y = df['main_type']

    # Filtrar clases con menos de 2 muestras (requerido por stratify)
    class_counts = y.value_counts()
    clases_invalidas = class_counts[class_counts < 2]
    if len(clases_invalidas) > 0:
        print(f"\n[!] Tipos con solo 1 muestra (excluidos del split): {dict(clases_invalidas)}")
        mascara = y.isin(class_counts[class_counts >= 2].index)
        X = X[mascara]
        y = y[mascara]

    if len(X) < 2:
        print("[ERROR] No hay suficientes espectros para dividir en entrenamiento/prueba")
        return

    # Split según test_size especificado
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"[!] Stratify fallo ({e}). Dividiendo sin estratificacion.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42
        )

    print(f"\nDatos de entrenamiento: {len(X_train)} espectros")
    print(f"Datos de prueba: {len(X_test)} espectros")
    print(f"Tipo de modelo: {args.model}")

    # 4. Entrenar modelo según tipo seleccionado
    if args.model == 'decision_tree':
        model_dt, acc_dt, importances = train_decision_tree(
            X_train, y_train, X_test, y_test, max_depth=args.tree_depth
        )
    elif args.model == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        model_dt = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.tree_depth,
            random_state=42,
            n_jobs=-1
        )
        model_dt.fit(X_train, y_train)
        y_pred = model_dt.predict(X_test)
        acc_dt = accuracy_score(y_test, y_pred)
        importances = pd.Series(model_dt.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print(f"\n{'='*50}")
        print(f"RANDOM FOREST ({args.n_estimators} árboles)")
        print(f"{'='*50}")
        print(f"Accuracy en test: {acc_dt*100:.2f}%")
    elif args.model == 'gradient_boosting':
        from sklearn.ensemble import GradientBoostingClassifier
        model_dt = GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.tree_depth,
            random_state=42
        )
        model_dt.fit(X_train, y_train)
        y_pred = model_dt.predict(X_test)
        acc_dt = accuracy_score(y_test, y_pred)
        importances = pd.Series(model_dt.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print(f"\n{'='*50}")
        print(f"GRADIENT BOOSTING ({args.n_estimators} estimadores)")
        print(f"{'='*50}")
        print(f"Accuracy en test: {acc_dt*100:.2f}%")

    # 5. Validación cruzada
    scores_cv, mean_cv = cross_validate_model(X, y, model_dt, k=args.k_folds)

    # 6. Guardar modelo
    model_path = os.path.join(args.output_dir, 'decision_tree.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_dt, f)
    print(f"[OK] Modelo guardado: {model_path}")

    # 7. Guardar metadata
    metadata = {
        'n_train': len(X_train),
        'n_test': len(X_test),
        'accuracy_test': acc_dt,
        'accuracy_cv_mean': mean_cv,
        'accuracy_cv_std': scores_cv.std(),
        'accuracy_physical': acc_physical,
        'feature_names': feature_cols,
        'timestamp': datetime.now().isoformat()
    }

    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Metadata guardada: {metadata_path}")

    # 8. Visualizaciones
    y_pred = model_dt.predict(X_test)

    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, MAIN_TYPES, cm_path)

    if args.model == 'decision_tree':
        tree_path = os.path.join(args.output_dir, 'decision_tree.png')
        plot_decision_tree_diagram(model_dt, feature_cols, tree_path, max_depth=3)

    # 9. Reporte
    generate_report(df, results_physical, model_dt, acc_dt, importances, args.output_dir)

    print(f"\n{'='*70}")
    print("[OK] ENTRENAMIENTO Y VALIDACION COMPLETADOS")
    print(f"{'='*70}")
    print(f"Accuracy clasificador físico: {acc_physical*100:.2f}%")
    print(f"Accuracy árbol de decisión: {acc_dt*100:.2f}%")
    print(f"Accuracy validación cruzada: {mean_cv*100:.2f}% (± {scores_cv.std()*100:.2f}%)")
    print(f"\nResultados guardados en: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
