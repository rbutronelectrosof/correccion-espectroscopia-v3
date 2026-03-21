"""
Script para Procesamiento por Lotes - Clasificación Espectral Corregida
========================================================================

Este script procesa MÚLTIPLES espectros estelares mostrando:
1. Informe detallado individual de cada estrella (con cuadro bonito)
2. Estadísticas consolidadas al final del procesamiento

Uso:
    python procesar_lote_estrellas.py

Configuración:
    - Modifica ARCHIVOS_A_PROCESAR para especificar los archivos
    - O usa START_INDEX y END_INDEX para procesar un rango
    - Los resultados se guardan en OUTPUT_DIR
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from pathlib import Path

# Añadir directorio src/ al path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Importar funciones del módulo corregido
try:
    from spectral_classification_corrected import (
        normalize_to_continuum,
        measure_diagnostic_lines,
        classify_star_corrected,
        plot_spectrum_corrected,
        SPECTRAL_LINES
    )
except ImportError as e:
    print(f"❌ Error al importar módulo: {e}")
    print("   Asegúrate de que spectral_classification_corrected.py está en el directorio")
    sys.exit(1)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Directorios (relativos al script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "elodie")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_lote")

# OPCIÓN 1: Lista específica de archivos (comenta/descomenta según necesites)
ARCHIVOS_A_PROCESAR = [
    'HD015570_tipoO4....txt',
    'HD015558_tipoO5e.txt',
    # Añade más archivos aquí
]

# OPCIÓN 2: Procesar un rango de archivos (descomenta para usar)
# START_INDEX = 0
# END_INDEX = 10  # None para procesar todos

# Configuración de visualización
GUARDAR_GRAFICOS = True
MOSTRAR_GRAFICOS = False  # True = mostrar gráficos en pantalla (ralentiza el proceso)

# Crear directorio de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def load_txt_spectrum(file_path):
    """Carga un espectro desde archivo .txt"""
    try:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        wavelengths = data[:, 0]
        flux = data[:, 1]
        return wavelengths, flux, None
    except Exception as e:
        return None, None, str(e)


def extract_metadata(filename):
    """Extrae metadatos del nombre del archivo"""
    if '_tipo_' in filename or '_tipo' in filename:
        parts = filename.replace('_tipo_', '_tipo').split('_tipo')
        object_name = parts[0]
        original_type = parts[1].replace('.txt', '').replace('....', '').replace('.', '')
    else:
        object_name = filename.replace('.txt', '')
        original_type = 'Desconocido'

    return object_name, original_type


def print_individual_report(archivo, object_name, spectral_type, subtype,
                           diagnostics, detected_lines_count, original_type):
    """
    Imprime el informe bonito individual de una estrella
    """
    tipo_completo = f"{spectral_type} {subtype}"

    print("\n\n" + "╔" + "="*78 + "╗")
    print("║" + " "*25 + "CLASIFICACIÓN FINAL" + " "*34 + "║")
    print("╠" + "="*78 + "╣")
    print("║" + " "*78 + "║")
    print(f"║  Archivo:  {archivo:<62} ║")
    print(f"║  Objeto:   {object_name:<63} ║")
    print("║" + " "*78 + "║")

    # RESULTADO PRINCIPAL CON DESTACADO VISUAL
    espacios_antes = (78 - len("TIPO ESPECTRAL:") - len(tipo_completo) - 4) // 2
    espacios_despues = 78 - espacios_antes - len("TIPO ESPECTRAL:") - len(tipo_completo) - 4

    print("║" + " "*78 + "║")
    print("║  " + "┏" + "━"*74 + "┓" + "  ║")
    print(f"║  ┃{' '*espacios_antes}TIPO ESPECTRAL: {tipo_completo}{' '*espacios_despues}┃  ║")
    print("║  " + "┗" + "━"*74 + "┛" + "  ║")
    print("║" + " "*78 + "║")

    # Valores característicos
    He_II = diagnostics['He_II']
    He_I = diagnostics['He_I']
    H_avg = diagnostics['H_avg']
    Ca_II_K = diagnostics['Ca_II_K']

    print("║  Valores característicos:" + " "*52 + "║")
    print("║" + " "*78 + "║")
    print(f"║    • He II promedio:    {He_II:6.3f} Å{' '*41}║")
    print(f"║    • He I promedio:     {He_I:6.3f} Å{' '*41}║")
    print(f"║    • H promedio:        {H_avg:6.3f} Å{' '*41}║")
    print(f"║    • Ca II K:           {Ca_II_K:6.3f} Å{' '*41}║")
    print(f"║    • Líneas detectadas: {detected_lines_count:<4}{' '*47}║")
    print("║" + " "*78 + "║")

    # Comparación con tipo esperado
    if original_type != 'Desconocido':
        print(f"║  Tipo esperado:  {original_type:<60} ║")
        print("║" + " "*78 + "║")

        letra_clasificada = spectral_type.split('-')[0]
        letra_esperada = original_type[0]

        if letra_clasificada == letra_esperada:
            print("║  " + " "*74 + "  ║")
            print(f"║  ✅ COINCIDE (ambos tipo {letra_clasificada}){' '*50}║")
            print("║  " + " "*74 + "  ║")
        else:
            print("║  " + " "*74 + "  ║")
            print(f"║  ⚠️  DISCREPANCIA{' '*59}║")
            print(f"║      Clasificado: {letra_clasificada}    Esperado: {letra_esperada}{' '*43}║")
            print("║  " + " "*74 + "  ║")

    print(f"║  Resultados guardados en:{' '*52}║")
    print(f"║  {OUTPUT_DIR:<76} ║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝\n")


def print_consolidated_statistics(results_df):
    """
    Imprime estadísticas consolidadas de todas las estrellas procesadas
    """
    print("\n\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "ESTADÍSTICAS CONSOLIDADAS" + " "*33 + "║")
    print("╠" + "="*78 + "╣")
    print("║" + " "*78 + "║")

    # Total procesadas
    total = len(results_df)
    print(f"║  Total de estrellas procesadas: {total:<44} ║")
    print("║" + " "*78 + "║")

    # Distribución por tipo espectral
    print("║  " + "─"*74 + "  ║")
    print("║  DISTRIBUCIÓN POR TIPO ESPECTRAL" + " "*44 + "║")
    print("║  " + "─"*74 + "  ║")

    # Extraer letra del tipo
    results_df['Tipo_letra'] = results_df['Tipo Clasificado'].str.split().str[0].str.split('-').str[0]
    type_counts = results_df['Tipo_letra'].value_counts().sort_index()

    for tipo in ['O', 'B', 'A', 'F', 'G', 'K', 'M']:
        count = type_counts.get(tipo, 0)
        porcentaje = (count / total * 100) if total > 0 else 0
        barra = "█" * int(porcentaje / 2)  # Escala: 2% = 1 carácter
        print(f"║    {tipo}:  {count:3d} ({porcentaje:5.1f}%)  {barra:<40} ║")

    print("║" + " "*78 + "║")

    # Comparación con tipos esperados (si existen)
    if 'Tipo Original' in results_df.columns:
        con_tipo = results_df[results_df['Tipo Original'] != 'Desconocido']
        if len(con_tipo) > 0:
            con_tipo['Original_letra'] = con_tipo['Tipo Original'].str[0]
            coincidencias = (con_tipo['Tipo_letra'] == con_tipo['Original_letra']).sum()
            total_con_tipo = len(con_tipo)
            tasa_acierto = (coincidencias / total_con_tipo * 100) if total_con_tipo > 0 else 0

            print("║  " + "─"*74 + "  ║")
            print("║  COMPARACIÓN CON TIPOS ESPERADOS" + " "*44 + "║")
            print("║  " + "─"*74 + "  ║")
            print(f"║    Estrellas con tipo conocido:  {total_con_tipo:<42} ║")
            print(f"║    Coincidencias:                 {coincidencias:<42} ║")
            print(f"║    Discrepancias:                 {total_con_tipo - coincidencias:<42} ║")
            print(f"║    Tasa de acierto:               {tasa_acierto:.1f}%{' '*38}║")
            print("║" + " "*78 + "║")

            # Mostrar discrepancias
            discrepancias = con_tipo[con_tipo['Tipo_letra'] != con_tipo['Original_letra']]
            if len(discrepancias) > 0:
                print("║  Discrepancias encontradas:" + " "*50 + "║")
                for _, row in discrepancias.iterrows():
                    obj = row['Objeto'][:20]
                    esperado = row['Tipo Original'][:5]
                    clasif = row['Tipo Clasificado'][:10]
                    print(f"║    • {obj:<20} → Esperado: {esperado:<5}  Clasificado: {clasif:<10} ║")
                print("║" + " "*78 + "║")

    # Promedios de anchos equivalentes por tipo
    print("║  " + "─"*74 + "  ║")
    print("║  ANCHOS EQUIVALENTES PROMEDIO POR TIPO" + " "*38 + "║")
    print("║  " + "─"*74 + "  ║")

    for tipo in ['O', 'B', 'A', 'F', 'G', 'K', 'M']:
        subset = results_df[results_df['Tipo_letra'] == tipo]
        if len(subset) > 0:
            he2_avg = subset['He II (Å)'].mean()
            he1_avg = subset['He I (Å)'].mean()
            h_avg = subset['H avg (Å)'].mean()
            ca_avg = subset['Ca II K (Å)'].mean()
            print(f"║    {tipo}: He II={he2_avg:5.2f}  He I={he1_avg:5.2f}  H={h_avg:5.2f}  Ca II K={ca_avg:5.2f}{' '*17}║")

    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝\n")


# ============================================================================
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO
# ============================================================================

def procesar_lote(archivos):
    """
    Procesa un lote de espectros estelares

    Parameters
    ----------
    archivos : list
        Lista de nombres de archivos a procesar

    Returns
    -------
    results : list
        Lista de diccionarios con resultados
    errors : list
        Lista de diccionarios con errores
    """
    results = []
    errors = []

    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " PROCESAMIENTO POR LOTES - CLASIFICACIÓN ESPECTRAL ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print(f"\nTotal de archivos a procesar: {len(archivos)}")
    print(f"Directorio de salida: {OUTPUT_DIR}\n")
    print("=" * 80)

    for i, filename in enumerate(archivos, 1):
        print(f"\n{'='*80}")
        print(f"PROCESANDO [{i}/{len(archivos)}]: {filename}")
        print(f"{'='*80}")

        try:
            # ================================================================
            # PASO 1: Cargar espectro
            # ================================================================
            print(f"\n{'─'*80}")
            print(f"► PASO 1: Cargando espectro")
            print(f"{'─'*80}")

            filepath = os.path.join(INPUT_DIR, filename)

            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

            wavelengths, flux, error = load_txt_spectrum(filepath)

            if error:
                raise Exception(f"Error al cargar: {error}")

            print(f"✓ Espectro cargado exitosamente")
            print(f"  Puntos espectrales: {len(wavelengths)}")
            print(f"  Rango λ: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} Å")
            print(f"  Rango de flujo: {flux.min():.2e} - {flux.max():.2e}")

            # ================================================================
            # PASO 2: Normalizar al continuo
            # ================================================================
            print(f"\n{'─'*80}")
            print(f"► PASO 2: Normalizando al continuo")
            print(f"{'─'*80}")
            print(f"  Aplicando corrección crítica #1:")
            print(f"  - Sigma-clipping para rechazar rayos cósmicos")
            print(f"  - Ajuste de continuo (continuo = 1.0)")

            flux_normalized, continuum = normalize_to_continuum(wavelengths, flux)

            print(f"✓ Normalización completada")
            print(f"  Continuo mediano: {np.median(continuum):.2e}")
            print(f"  Flujo normalizado: min={flux_normalized.min():.3f}, max={flux_normalized.max():.3f}")

            # ================================================================
            # PASO 3: Medir anchos equivalentes
            # ================================================================
            print(f"\n{'─'*80}")
            print(f"► PASO 3: Midiendo anchos equivalentes")
            print(f"{'─'*80}")
            print(f"  Aplicando corrección crítica #2:")
            print(f"  - Cálculo de EW = ∫(1 - F_λ) dλ (intensidad de línea)")
            print(f"  - NO usa FWHM (que mide ensanchamiento)")

            measurements = measure_diagnostic_lines(wavelengths, flux_normalized)
            detected_lines = {name: data for name, data in measurements.items() if data['ew'] > 0.05}

            print(f"✓ Anchos equivalentes medidos")
            print(f"  Líneas detectadas (EW > 0.05 Å): {len(detected_lines)}")

            # Mostrar líneas más intensas
            lineas_fuertes = {n: d for n, d in detected_lines.items() if d['ew'] > 0.3}
            if lineas_fuertes:
                print(f"\n  Líneas más intensas (EW > 0.3 Å):")
                for line_name, data in sorted(lineas_fuertes.items(), key=lambda x: x[1]['ew'], reverse=True):
                    print(f"    {line_name:15s}: EW = {data['ew']:6.2f} Å, profundidad = {data['depth']:.3f}")

            # ================================================================
            # PASO 4: Clasificar
            # ================================================================
            print(f"\n{'─'*80}")
            print(f"► PASO 4: Clasificación espectral")
            print(f"{'─'*80}")
            print(f"  Aplicando corrección crítica #3:")
            print(f"  - Clasificación basada en INTENSIDADES (EW), no longitudes de onda")
            print(f"  - Usando criterios espectroscópicos estándar")

            # Mostrar parámetros diagnóstico antes de clasificar
            He_II_diag = sum(measurements.get(l, {}).get('ew', 0.0)
                            for l in ['He_II_4686', 'He_II_4542', 'He_II_4200', 'He_II_5411']) / 4
            He_I_diag  = sum(measurements.get(l, {}).get('ew', 0.0)
                            for l in ['He_I_4471', 'He_I_4026', 'He_I_4922', 'He_I_5016']) / 4
            H_avg_diag = (measurements.get('H_beta', {}).get('ew', 0.0) +
                          measurements.get('H_gamma', {}).get('ew', 0.0) +
                          measurements.get('H_delta', {}).get('ew', 0.0)) / 3.0
            Ca_diag    = measurements.get('Ca_II_K', {}).get('ew', 0.0)

            print(f"\n  Parámetros diagnóstico:")
            print(f"    He II promedio:  {He_II_diag:.3f} Å")
            print(f"    He I promedio:   {He_I_diag:.3f} Å")
            print(f"    H promedio:      {H_avg_diag:.3f} Å")
            print(f"    Ca II K:         {Ca_diag:.3f} Å")

            spectral_type, subtype, diagnostics = classify_star_corrected(
                measurements, wavelengths, flux_normalized
            )
            print(f"\n✓ Clasificación completada: {spectral_type} {subtype}")

            # ================================================================
            # PASO 5: Extraer metadatos
            # ================================================================
            print(f"\n{'─'*80}")
            print(f"► PASO 5: Extrayendo metadatos y guardando resultados")
            print(f"{'─'*80}")

            object_name, original_type = extract_metadata(filename)
            print(f"  Objeto:         {object_name}")
            print(f"  Tipo esperado:  {original_type}")

            # ================================================================
            # PASO 6: Guardar gráfico
            # ================================================================
            if GUARDAR_GRAFICOS:
                output_file = os.path.join(OUTPUT_DIR, filename.replace('.txt', '_clasificado.png'))
                plot_spectrum_corrected(
                    wavelengths, flux_normalized, measurements,
                    spectral_type, subtype,
                    object_name, original_type,
                    save_path=output_file
                )
                print(f"✓ Gráfico guardado: {output_file}")

            # ================================================================
            # MOSTRAR INFORME INDIVIDUAL BONITO
            # ================================================================
            print_individual_report(
                filename, object_name, spectral_type, subtype,
                diagnostics, len(detected_lines), original_type
            )

            # ================================================================
            # Guardar resultado
            # ================================================================
            results.append({
                'Archivo': filename,
                'Objeto': object_name,
                'Tipo Original': original_type,
                'Tipo Clasificado': f"{spectral_type} {subtype}",
                'He II (Å)': diagnostics['He_II'],
                'He I (Å)': diagnostics['He_I'],
                'H avg (Å)': diagnostics['H_avg'],
                'Ca II K (Å)': diagnostics['Ca_II_K'],
                'Líneas detectadas': len(detected_lines)
            })

        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            errors.append({'file': filename, 'error': str(e)})

    return results, errors


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":

    # Determinar qué archivos procesar
    if 'START_INDEX' in globals() and 'END_INDEX' in globals():
        # Opción 2: Rango de archivos
        if os.path.exists(INPUT_DIR):
            all_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')])
            archivos_a_procesar = all_files[START_INDEX:END_INDEX]
            print(f"Modo: Rango de archivos ({START_INDEX} a {END_INDEX})")
        else:
            print(f"❌ Directorio {INPUT_DIR} no encontrado")
            sys.exit(1)
    else:
        # Opción 1: Lista específica
        archivos_a_procesar = ARCHIVOS_A_PROCESAR
        print(f"Modo: Lista específica de archivos")

    if len(archivos_a_procesar) == 0:
        print("❌ No hay archivos para procesar")
        print("   Configura ARCHIVOS_A_PROCESAR o START_INDEX/END_INDEX en el script")
        sys.exit(1)

    # Procesar lote
    results, errors = procesar_lote(archivos_a_procesar)

    # ========================================================================
    # ESTADÍSTICAS CONSOLIDADAS
    # ========================================================================

    if len(results) > 0:
        df_results = pd.DataFrame(results)

        # Imprimir estadísticas consolidadas
        print_consolidated_statistics(df_results)

        # Guardar resultados en CSV
        csv_path = os.path.join(OUTPUT_DIR, 'clasificacion_lote_resultados.csv')
        df_results.to_csv(csv_path, index=False)
        print(f"✓ Resultados guardados en CSV: {csv_path}")

    # Guardar errores si los hay
    if len(errors) > 0:
        error_df = pd.DataFrame(errors)
        error_path = os.path.join(OUTPUT_DIR, 'errores_log.csv')
        error_df.to_csv(error_path, index=False)
        print(f"⚠️  Log de errores guardado: {error_path}")

    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================

    print("\n" + "="*80)
    print("RESUMEN FINAL DEL PROCESAMIENTO")
    print("="*80)
    print(f"✓ Procesados exitosamente: {len(results)}")
    print(f"✗ Errores:                 {len(errors)}")
    if GUARDAR_GRAFICOS:
        print(f"📊 Gráficos generados:      {len(results)}")
    print(f"\n📁 Todos los archivos guardados en: {OUTPUT_DIR}")
    print("="*80)

    print("\n✅ Procesamiento por lotes completado\n")
