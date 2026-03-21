"""


MODOS DE OPERACIÓN:
------------------
1. MODO INDIVIDUAL: Procesa una sola estrella con gráficos detallados
2. MODO LOTE: Procesa múltiples estrellas y genera UN SOLO PDF consolidado

Uso:
    python procesar_una_estrella.py

Configuración:
    - Modifica MODO para elegir 'INDIVIDUAL' o 'LOTE'
    - En modo INDIVIDUAL: modifica ARCHIVO_ESPECTRO
    - En modo LOTE: modifica LISTA_ARCHIVOS o PROCESAR_TODOS
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import sys
from pathlib import Path
from datetime import datetime
from glob import glob

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

# MODO DE OPERACIÓN: 'INDIVIDUAL' o 'LOTE'
MODO = 'INDIVIDUAL'  # Cambiar a 'LOTE' para procesar múltiples estrellas

# Directorios (relativos al script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "elodie")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_estrella_individual")

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN MODO INDIVIDUAL
# ──────────────────────────────────────────────────────────────────────────
# Archivo a procesar en modo individual
ARCHIVO_ESPECTRO = 'HD048279_tipoO8.txt'  #  (test clasificación)

# Opciones de visualización (solo modo individual)
MOSTRAR_GRAFICOS_INDIVIDUAL = True   # Mostrar gráficos en pantalla
GUARDAR_GRAFICOS_INDIVIDUAL = True   # Guardar PNGs individuales

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN MODO LOTE
# ──────────────────────────────────────────────────────────────────────────
# Nombre del PDF consolidado de salida
PDF_CONSOLIDADO = os.path.join(SCRIPT_DIR, 'informe_clasificacion_consolidado.pdf')

# Opción 1: Procesar TODOS los archivos .txt del directorio
PROCESAR_TODOS = False

# Opción 2: Lista específica de archivos a procesar
LISTA_ARCHIVOS = [
    'HD166734_tipoO8e.txt',           # Tipo O
    'HD015570_tipo_O4....txt',        # Tipo O
    'HD161677_tipo_B6V.txt',          # Tipo B
    'BD+023375_tipoA5.txt',           # Tipo A
    'BD+024651_tipoF5.txt',           # Tipo F
    'BD+233130_tipoG0.txt',           # Tipo G
    'BD+302611_tipoG8III.txt',        # Tipo G-K
    'BD+321561_tipoK2V.txt',          # Tipo K
    'BD+362219_tipoM1.txt',           # Tipo M
]

# Crear directorio de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def load_txt_spectrum(file_path):
    """
    Carga un espectro desde archivo .txt (formato: wavelength, flux)

    Parameters
    ----------
    file_path : str
        Ruta al archivo .txt

    Returns
    -------
    wavelengths : array
        Longitudes de onda en Å
    flux : array
        Flujo observado
    error : str or None
        Mensaje de error si ocurre algún problema
    """
    try:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        wavelengths = data[:, 0]
        flux = data[:, 1]
        return wavelengths, flux, None
    except Exception as e:
        return None, None, str(e)


def extract_metadata(filename):
    """
    Extrae metadatos del nombre del archivo.

    Parameters
    ----------
    filename : str
        Nombre del archivo

    Returns
    -------
    object_name : str
        Nombre del objeto
    original_type : str
        Tipo espectral original (si está en el nombre)
    """
    if '_tipo_' in filename:
        object_name = filename.split('_tipo_')[0]
        original_type = filename.split('_tipo_')[1].replace('.txt', '').replace('....', '')
    else:
        object_name = filename.replace('.txt', '')
        original_type = 'Desconocido'

    return object_name, original_type


def print_header(text, char='='):
    """Imprime un encabezado formateado"""
    width = 70
    print('\n' + char * width)
    print(text.center(width))
    print(char * width)


def print_section(text):
    """Imprime un título de sección"""
    print(f'\n{"─" * 70}')
    print(f'► {text}')
    print(f'{"─" * 70}')


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def procesar_estrella(archivo, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR,
                      mostrar_graficos=True, guardar_graficos=True):
    """
    Procesa un espectro estelar completo con todas las correcciones.

    Parameters
    ----------
    archivo : str
        Nombre del archivo a procesar
    input_dir : str
        Directorio de entrada
    output_dir : str
        Directorio de salida
    mostrar_graficos : bool
        Si True, muestra los gráficos en pantalla
    guardar_graficos : bool
        Si True, guarda los gráficos en archivos PNG

    Returns
    -------
    results : dict
        Diccionario con todos los resultados del procesamiento
    """

    print_header('PROCESAMIENTO DE ESPECTRO ESTELAR INDIVIDUAL')
    print(f'Archivo: {archivo}')
    print(f'Directorio entrada: {input_dir}')
    print(f'Directorio salida: {output_dir}')

    # ========================================================================
    # PASO 1: CARGAR ESPECTRO
    # ========================================================================

    print_section('PASO 1: Cargando espectro')

    file_path = os.path.join(input_dir, archivo)

    if not os.path.exists(file_path):
        print(f'❌ ERROR: El archivo no existe: {file_path}')
        return None

    wavelengths, flux, error = load_txt_spectrum(file_path)

    if error:
        print(f'❌ ERROR al cargar: {error}')
        return None

    print(f'✓ Espectro cargado exitosamente')
    print(f'  Puntos espectrales: {len(wavelengths)}')
    print(f'  Rango de λ: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} Å')
    print(f'  Rango de flujo: {flux.min():.2e} - {flux.max():.2e}')

    # Graficar espectro crudo
    if mostrar_graficos or guardar_graficos:
        fig1, ax = plt.subplots(figsize=(14, 4))
        ax.plot(wavelengths, flux, 'k-', linewidth=0.5)
        ax.set_xlabel('Longitud de Onda (Å)', fontsize=12)
        ax.set_ylabel('Flujo (crudo)', fontsize=12)
        ax.set_title(f'Espectro Crudo - {archivo}', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if guardar_graficos:
            save_path = os.path.join(output_dir, f'{archivo.replace(".txt", "")}_1_crudo.png')
            plt.savefig(save_path, dpi=150)
            print(f'  → Guardado: {save_path}')

        if mostrar_graficos:
            plt.show()
        else:
            plt.close()

    # ========================================================================
    # PASO 2: NORMALIZACIÓN AL CONTINUO
    # ========================================================================

    print_section('PASO 2: Normalizando al continuo')
    print('  Aplicando corrección crítica #1:')
    print('  - Sigma-clipping para rechazar rayos cósmicos')
    print('  - Ajuste de continuo (continuo = 1.0)')

    flux_normalized, continuum = normalize_to_continuum(wavelengths, flux)

    print(f'✓ Normalización completada')
    print(f'  Continuo mediano: {np.median(continuum):.2e}')
    print(f'  Flujo normalizado: min={flux_normalized.min():.3f}, max={flux_normalized.max():.3f}')

    # Comparar normalización
    if mostrar_graficos or guardar_graficos:
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        # Antes
        ax1.plot(wavelengths, flux, 'b-', linewidth=0.5, label='Flujo crudo')
        ax1.plot(wavelengths, continuum, 'r-', linewidth=1.5, label='Continuo ajustado')
        ax1.set_ylabel('Flujo', fontsize=11)
        ax1.set_title('ANTES: Espectro crudo + continuo detectado', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)

        # Después
        ax2.plot(wavelengths, flux_normalized, 'g-', linewidth=0.5)
        ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Continuo = 1.0')
        ax2.set_xlabel('Longitud de Onda (Å)', fontsize=12)
        ax2.set_ylabel('Flujo normalizado', fontsize=11)
        ax2.set_title('DESPUÉS: Espectro normalizado al continuo', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if guardar_graficos:
            save_path = os.path.join(output_dir, f'{archivo.replace(".txt", "")}_2_normalizacion.png')
            plt.savefig(save_path, dpi=150)
            print(f'  → Guardado: {save_path}')

        if mostrar_graficos:
            plt.show()
        else:
            plt.close()

    # ========================================================================
    # PASO 3: MEDICIÓN DE ANCHOS EQUIVALENTES
    # ========================================================================

    print_section('PASO 3: Midiendo anchos equivalentes')
    print('  Aplicando corrección crítica #2:')
    print('  - Cálculo de EW = ∫(1 - F_λ) dλ (intensidad de línea)')
    print('  - NO usa FWHM (que mide ensanchamiento)')

    measurements = measure_diagnostic_lines(wavelengths, flux_normalized)

    # Filtrar líneas detectadas
    detected_lines = {name: data for name, data in measurements.items() if data['ew'] > 0.05}

    print(f'✓ Anchos equivalentes medidos')
    print(f'  Líneas detectadas (EW > 0.05 Å): {len(detected_lines)}')

    if detected_lines:
        print('\n  Líneas más intensas (EW > 0.3 Å):')
        for line_name, data in sorted(detected_lines.items(), key=lambda x: x[1]['ew'], reverse=True):
            if data['ew'] > 0.3:
                print(f'    {line_name:15s}: EW = {data["ew"]:6.2f} Å, profundidad = {data["depth"]:.3f}')

    # Visualizar líneas detectadas
    if (mostrar_graficos or guardar_graficos) and detected_lines:
        fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

        # Panel superior: Espectro con líneas marcadas
        ax1.plot(wavelengths, flux_normalized, 'k-', linewidth=0.5, alpha=0.7, label='Espectro normalizado')
        ax1.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        colors = plt.cm.tab20(np.linspace(0, 1, len(detected_lines)))
        for i, (line_name, data) in enumerate(detected_lines.items()):
            center = data['wavelength']
            ew = data['ew']

            # Línea vertical
            ax1.axvline(center, color=colors[i], linestyle='-', linewidth=2, alpha=0.8)

            # Región de integración
            width = ew * 2
            ax1.axvspan(center - width/2, center + width/2, alpha=0.2, color=colors[i])

            # Etiqueta
            ax1.text(center, 1.05, f'{line_name}\n{ew:.2f}Å',
                    rotation=90, fontsize=8, ha='right', va='bottom', color=colors[i])

        ax1.set_xlabel('Longitud de Onda (Å)', fontsize=12)
        ax1.set_ylabel('Flujo Normalizado', fontsize=12)
        ax1.set_title(f'Líneas Espectrales Detectadas - {archivo}', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)
        ax1.set_ylim([0, 1.3])
        ax1.legend(loc='upper right')

        # Panel inferior: Barras de EW
        line_names = list(detected_lines.keys())
        ews = [data['ew'] for data in detected_lines.values()]

        bars = ax2.barh(line_names, ews, color=colors)
        ax2.set_xlabel('Ancho Equivalente (Å)', fontsize=10)
        ax2.set_ylabel('Línea espectral', fontsize=10)
        ax2.set_title('Intensidades de Líneas (Anchos Equivalentes)', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3, axis='x')

        for j, (bar, ew) in enumerate(zip(bars, ews)):
            ax2.text(ew + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{ew:.2f} Å', va='center', fontsize=6)

        plt.tight_layout()

        if guardar_graficos:
            save_path = os.path.join(output_dir, f'{archivo.replace(".txt", "")}_3_lineas.png')
            plt.savefig(save_path, dpi=150)
            print(f'  → Guardado: {save_path}')

        if mostrar_graficos:
            plt.show()
        else:
            plt.close()

    # ========================================================================
    # PASO 4: CLASIFICACIÓN ESPECTRAL
    # ========================================================================

    print_section('PASO 4: Clasificación espectral')
    print('  Aplicando corrección crítica #3:')
    print('  - Clasificación basada en INTENSIDADES (EW), no longitudes de onda')
    print('  - Usando criterios espectroscópicos estándar')

    # Calcular parámetros diagnóstico
    He_II = sum(measurements.get(line, {}).get('ew', 0.0)
                for line in ['He_II_4686', 'He_II_4542', 'He_II_4200', 'He_II_5411']) / 4
    He_I = sum(measurements.get(line, {}).get('ew', 0.0)
               for line in ['He_I_4471', 'He_I_4026', 'He_I_4922', 'He_I_5016']) / 4
    # IMPORTANTE: H_avg usa solo 3 líneas principales (NO H_epsilon que puede estar fuera de rango)
    # Debe coincidir con spectral_classification_corrected.py línea 583
    H_beta = measurements.get('H_beta', {}).get('ew', 0.0)
    H_gamma = measurements.get('H_gamma', {}).get('ew', 0.0)
    H_delta = measurements.get('H_delta', {}).get('ew', 0.0)
    H_avg = (H_beta + H_gamma + H_delta) / 3.0
    Ca_II_K = measurements.get('Ca_II_K', {}).get('ew', 0.0)

    print('\n  Parámetros diagnóstico:')
    print(f'    He II promedio:  {He_II:.2f} Å')
    print(f'    He I promedio:   {He_I:.2f} Å')
    print(f'    H promedio:      {H_avg:.2f} Å')
    print(f'    Ca II K:         {Ca_II_K:.2f} Å')

    # Clasificar (con detección mejorada de tipo M)
    spectral_type, subtype, diagnostics = classify_star_corrected(
        measurements, wavelengths, flux_normalized
    )

    print(f'\n✓ Clasificación completada')
    print(f'  Tipo espectral: {spectral_type} {subtype}')

    # Comparar con tipo original si existe
    object_name, original_type = extract_metadata(archivo)

    if original_type != 'Desconocido':
        letra_clasificada = spectral_type.split('-')[0]  # Por si es "G-K"
        letra_esperada = original_type[0]

        print(f'  Tipo esperado:  {original_type}')

        if letra_clasificada == letra_esperada:
            print(f'  ✅ COINCIDE (ambos tipo {letra_clasificada})')
        else:
            print(f'  ⚠️  DISCREPANCIA: clasificado como {letra_clasificada}, esperado {letra_esperada}')

    # ========================================================================
    # PASO 5: VISUALIZACIÓN COMPLETA
    # ========================================================================

    print_section('PASO 5: Generando visualización completa')

    if mostrar_graficos or guardar_graficos:
        output_file = os.path.join(output_dir, f'{archivo.replace(".txt", "")}_clasificado.png')

        plot_spectrum_corrected(
            wavelengths, flux_normalized, measurements,
            spectral_type, subtype,
            object_name, original_type,
            save_path=output_file if guardar_graficos else None
        )

        if guardar_graficos:
            print(f'  → Guardado: {output_file}')

        if mostrar_graficos and not guardar_graficos:
            # Si solo mostramos, plot_spectrum_corrected ya lo hace
            pass

    # ========================================================================
    # RESUMEN FINAL CON FORMATO DESTACADO
    # ========================================================================

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
    print("║  Valores característicos:" + " "*52 + "║")
    print("║" + " "*78 + "║")
    print(f"║    • He II promedio:    {He_II:6.3f} Å{' '*41}║")
    print(f"║    • He I promedio:     {He_I:6.3f} Å{' '*41}║")
    print(f"║    • H promedio:        {H_avg:6.3f} Å{' '*41}║")
    print(f"║    • Ca II K:           {Ca_II_K:6.3f} Å{' '*41}║")
    print(f"║    • Líneas detectadas: {len(detected_lines):<4}{' '*47}║")
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
    print(f"║  {output_dir:<76} ║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝\n")

    # Preparar diccionario de resultados
    results = {
        'archivo': archivo,
        'objeto': object_name,
        'tipo_original': original_type,
        'tipo_clasificado': spectral_type,
        'subtipo': subtype,
        'wavelengths': wavelengths,
        'flux': flux,
        'flux_normalized': flux_normalized,
        'continuum': continuum,
        'measurements': measurements,
        'diagnostics': {
            'He_II': He_II,
            'He_I': He_I,
            'H_avg': H_avg,
            'Ca_II_K': Ca_II_K
        },
        'detected_lines': detected_lines
    }

    return results


# ============================================================================
# FUNCIONES PARA MODO LOTE (PDF CONSOLIDADO)
# ============================================================================

def create_pdf_page(pdf, archivo, wavelengths, flux_normalized, measurements,
                    spectral_type, subtype, object_name, original_type, diagnostics):
    """
    Crea una página completa para una estrella en el PDF consolidado.

    Parameters
    ----------
    pdf : PdfPages
        Objeto PdfPages para añadir la página
    archivo : str
        Nombre del archivo
    wavelengths, flux_normalized : arrays
        Datos espectrales normalizados
    measurements : dict
        Mediciones de anchos equivalentes
    spectral_type, subtype : str
        Clasificación
    object_name, original_type : str
        Metadatos
    diagnostics : dict
        Valores diagnóstico
    """

    # Crear figura con 3 paneles verticales (tamaño carta)
    fig = plt.figure(figsize=(8.5, 11))

    gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 1.5, 1.5], hspace=0.4,
                          left=0.1, right=0.95, top=0.93, bottom=0.05)

    # ========================================================================
    # PANEL 1: ESPECTRO CON LÍNEAS DETECTADAS
    # ========================================================================

    ax1 = fig.add_subplot(gs[0, 0])

    # Espectro normalizado
    ax1.plot(wavelengths, flux_normalized, 'k-', linewidth=0.4, alpha=0.7)
    ax1.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Detectar líneas significativas (EW > 0.1 Å)
    detected_lines = {name: data for name, data in measurements.items() if data['ew'] > 0.1}

    # Colores por tipo de línea
    line_colors = {
        'He_II': 'red',
        'He_I': 'orange',
        'H_': 'blue',
        'Ca_II': 'green',
        'Ca_I': 'darkgreen',
        'Fe_I': 'purple',
        'Mg_II': 'brown',
        'Si_': 'darkblue',
        'N_V': 'darkred',
        'Cr_I': 'olive',
    }

    if detected_lines:
        for line_name, data in detected_lines.items():
            center = data['wavelength']
            ew = data['ew']

            # Determinar color
            color = 'gray'
            for key, col in line_colors.items():
                if line_name.startswith(key):
                    color = col
                    break

            # Línea vertical
            ax1.axvline(center, color=color, linestyle='-', linewidth=1.5, alpha=0.7)

            # Etiqueta (solo para líneas fuertes EW > 0.3)
            if ew > 0.3:
                label = line_name.replace('_', ' ')
                ax1.text(center, 1.08, f'{label}\n{ew:.2f}Å',
                        rotation=90, fontsize=6, ha='right', va='bottom',
                        color=color, weight='bold')

    ax1.set_xlabel('Longitud de Onda (Å)', fontsize=10)
    ax1.set_ylabel('Flujo Normalizado', fontsize=10)
    ax1.set_ylim([0, 1.25])
    ax1.grid(alpha=0.3, linewidth=0.5)

    # Título con clasificación
    tipo_completo = f"{spectral_type} {subtype}"
    coincide = (spectral_type[0] == original_type[0]) if original_type != 'Desconocido' else None

    if coincide:
        match_symbol = "✓"
        title_color = 'darkgreen'
    elif coincide is False:
        match_symbol = "✗"
        title_color = 'darkred'
    else:
        match_symbol = ""
        title_color = 'black'

    title = f"{object_name} - Clasificación: {tipo_completo}"
    if original_type != 'Desconocido':
        title += f"  |  Esperado: {original_type}  {match_symbol}"

    ax1.set_title(title, fontsize=11, fontweight='bold', color=title_color, pad=10)

    # ========================================================================
    # PANEL 2: GRÁFICO DE BARRAS DE ANCHOS EQUIVALENTES
    # ========================================================================

    ax2 = fig.add_subplot(gs[1, 0])

    if detected_lines:
        # Ordenar por EW (mayor a menor) y tomar top 15
        sorted_lines = sorted(detected_lines.items(), key=lambda x: x[1]['ew'], reverse=True)
        top_lines = sorted_lines[:15]

        line_names = [name.replace('_', ' ') for name, _ in top_lines]
        ews = [data['ew'] for _, data in top_lines]

        # Colores por tipo
        colors_list = []
        for name, _ in top_lines:
            color = 'gray'
            for key, col in line_colors.items():
                if name.startswith(key):
                    color = col
                    break
            colors_list.append(color)

        bars = ax2.barh(line_names, ews, color=colors_list, alpha=0.7,
                       edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Ancho Equivalente (Å)', fontsize=10)
        ax2.set_ylabel('Línea Espectral', fontsize=9)
        ax2.set_title('Líneas Espectrales Más Intensas (Top 15)',
                     fontsize=11, fontweight='bold')
        ax2.grid(alpha=0.3, axis='x', linewidth=0.5)

        # Añadir valores en las barras
        for bar, ew in zip(bars, ews):
            width = bar.get_width()
            ax2.text(width + max(ews)*0.02, bar.get_y() + bar.get_height()/2,
                    f'{ew:.2f}', va='center', fontsize=7)

        ax2.tick_params(axis='y', labelsize=7)
    else:
        ax2.text(0.5, 0.5, 'No se detectaron líneas significativas (EW > 0.1 Å)',
                ha='center', va='center', fontsize=10, transform=ax2.transAxes)
        ax2.axis('off')

    # ========================================================================
    # PANEL 3: TABLA DE DIAGNÓSTICO
    # ========================================================================

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis('off')

    # Preparar datos de la tabla
    table_data = []

    # Encabezado
    table_data.append(['PARÁMETRO DIAGNÓSTICO', 'Valor', 'Umbral', 'Estado'])
    table_data.append(['─' * 30, '─' * 8, '─' * 10, '─' * 8])

    # He II (tipo O)
    He_II_avg = diagnostics.get('He_II_avg', 0.0)
    He_II_status = '✓ FUERTE' if He_II_avg > 0.15 else '✗ débil'
    table_data.append([' He II promedio', f'{He_II_avg:.3f} Å', '> 0.15 Å', He_II_status])

    # He I (tipo B)
    He_I_avg = (diagnostics.get('He_I_4471', 0.0) + diagnostics.get('He_I_4387', 0.0)) / 2
    He_I_status = '✓ FUERTE' if He_I_avg > 0.2 else '✗ débil'
    table_data.append([' He I promedio', f'{He_I_avg:.3f} Å', '> 0.20 Å', He_I_status])

    # H promedio (tipos A-F)
    H_avg = diagnostics.get('H_avg', 0.0)
    H_status = '✓ FUERTE' if H_avg > 3.0 else ('moderado' if H_avg > 1.0 else '✗ débil')
    table_data.append([' H promedio (Balmer)', f'{H_avg:.3f} Å', '> 3.0 Å', H_status])

    # Ca II K (tipos A-K)
    Ca_II_K = diagnostics.get('Ca_II_K', 0.0)
    Ca_status = '✓ FUERTE' if Ca_II_K > 2.0 else ('moderado' if Ca_II_K > 0.5 else '✗ débil')
    table_data.append([' Ca II K (3933 Å)', f'{Ca_II_K:.3f} Å', '> 2.0 Å', Ca_status])

    # Fe I (tipos F-M)
    Fe_I_avg = (diagnostics.get('Fe_I_4046', 0.0) + diagnostics.get('Fe_I_4144', 0.0)) / 2
    Fe_status = '✓ presente' if Fe_I_avg > 0.3 else '✗ ausente'
    table_data.append([' Fe I promedio', f'{Fe_I_avg:.3f} Å', '> 0.30 Å', Fe_status])

    table_data.append(['', '', '', ''])

    # Criterio de clasificación
    table_data.append(['CRITERIO DE CLASIFICACIÓN', '', '', ''])
    table_data.append(['─' * 30, '─' * 8, '─' * 10, '─' * 8])

    # Determinar el criterio usado
    if diagnostics.get('has_He_II', False):
        criterio = 'He II presente → Tipo O'
    elif diagnostics.get('has_He_I', False):
        criterio = 'He I presente (sin He II) → Tipo B'
    elif H_avg > 5.0 and Ca_II_K < H_avg * 0.5:
        criterio = 'H fuerte, Ca II K débil → Tipo A'
    elif H_avg > 3.0:
        criterio = 'Líneas de Balmer bien definidas → Tipo F'
    elif Ca_II_K > 2.0 or Fe_I_avg > 0.3:
        criterio = 'Metales dominantes → Tipo G-K-M'
    else:
        criterio = 'Clasificación basada en ratios de intensidad'

    table_data.append([f' {criterio}', '', '', ''])

    # Crear tabla
    cell_colors = []
    for i, row in enumerate(table_data):
        if i == 0 or i == len(table_data) - 3:  # Headers
            cell_colors.append(['lightgray'] * 4)
        elif '─' in row[0]:  # Separadores
            cell_colors.append(['white'] * 4)
        else:
            cell_colors.append(['white'] * 4)

    table = ax3.table(cellText=table_data, cellLoc='left',
                     loc='center', cellColours=cell_colors,
                     colWidths=[0.5, 0.15, 0.2, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)

    # Añadir información de archivo en el pie
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.5, 0.01, f'Archivo: {archivo}  |  Procesado: {fecha}',
             ha='center', fontsize=7, style='italic', color='gray')

    # Guardar página en PDF
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def procesar_lote():
    """
    Procesa múltiples estrellas y genera un PDF consolidado.

    Returns
    -------
    bool
        True si se procesó exitosamente, False en caso de error
    """

    print("\n" + "═" * 70)
    print("MODO LOTE: INFORME PDF CONSOLIDADO".center(70))
    print("Clasificación Espectral Corregida".center(70))
    print("═" * 70 + "\n")

    # Verificar directorio de entrada
    if not os.path.exists(INPUT_DIR):
        print(f"❌ ERROR: El directorio no existe: {INPUT_DIR}")
        return False

    # Seleccionar archivos a procesar
    if PROCESAR_TODOS:
        archivos = [os.path.basename(f) for f in glob(os.path.join(INPUT_DIR, "*.txt"))]
        print(f"📂 Procesando TODOS los archivos .txt del directorio")
        print(f"   Total: {len(archivos)} archivos")
    else:
        archivos = LISTA_ARCHIVOS
        print(f"📂 Procesando lista específica de archivos")
        print(f"   Total: {len(archivos)} archivos en la lista")

    # Filtrar archivos que existen
    archivos_existentes = []
    for archivo in archivos:
        if os.path.exists(os.path.join(INPUT_DIR, archivo)):
            archivos_existentes.append(archivo)
        else:
            print(f"⚠️  Archivo no encontrado (omitido): {archivo}")

    if not archivos_existentes:
        print("❌ ERROR: No hay archivos para procesar")
        return False

    print(f"\n✓ {len(archivos_existentes)} archivos encontrados")
    print(f"📄 Generando PDF: {PDF_CONSOLIDADO}\n")
    print("─" * 70)

    # Crear PDF consolidado
    with PdfPages(PDF_CONSOLIDADO) as pdf:

        # Procesar cada estrella
        for i, archivo in enumerate(archivos_existentes, 1):
            print(f"[{i}/{len(archivos_existentes)}] {archivo}...", end=" ", flush=True)

            file_path = os.path.join(INPUT_DIR, archivo)

            try:
                # Cargar espectro
                wavelengths, flux, error = load_txt_spectrum(file_path)
                if error:
                    print(f"❌ Error al cargar: {error}")
                    continue

                # Normalizar
                flux_normalized, continuum = normalize_to_continuum(wavelengths, flux)

                # Medir líneas
                measurements = measure_diagnostic_lines(wavelengths, flux_normalized)

                # Clasificar (con detección mejorada de tipo M)
                spectral_type, subtype, diagnostics = classify_star_corrected(
                    measurements, wavelengths, flux_normalized
                )

                # Extraer metadatos
                object_name, original_type = extract_metadata(archivo)

                # Añadir He_II_avg a diagnostics si no está
                if 'He_II_avg' not in diagnostics:
                    He_II_avg = (diagnostics.get('He_II_4686', 0.0) +
                                diagnostics.get('He_II_4542', 0.0) +
                                measurements.get('He_II_4200', {}).get('ew', 0.0)) / 3.0
                    diagnostics['He_II_avg'] = He_II_avg

                # Crear página en PDF
                create_pdf_page(pdf, archivo, wavelengths, flux_normalized,
                              measurements, spectral_type, subtype,
                              object_name, original_type, diagnostics)

                # Verificar clasificación
                if original_type != 'Desconocido':
                    letra_clasificada = spectral_type[0]
                    letra_esperada = original_type[0]
                    if letra_clasificada == letra_esperada:
                        print(f"✓ {spectral_type}{subtype} (coincide con {original_type})")
                    else:
                        print(f"⚠️  {spectral_type}{subtype} (esperado {original_type})")
                else:
                    print(f"✓ {spectral_type}{subtype}")

            except Exception as e:
                print(f"❌ ERROR: {str(e)}")
                continue

        # Añadir metadatos al PDF
        d = pdf.infodict()
        d['Title'] = 'Informe de Clasificación Espectral Estelar'
        d['Author'] = 'Clasificación Espectral Corregida'
        d['Subject'] = f'Procesamiento de {len(archivos_existentes)} espectros estelares'
        d['Keywords'] = 'Espectroscopia, Clasificación Estelar, Anchos Equivalentes'
        d['CreationDate'] = datetime.now()

    print("─" * 70)
    print(f"\n✅ Informe PDF generado exitosamente")
    print(f"📄 Archivo: {PDF_CONSOLIDADO}")
    print(f"📊 Total de estrellas procesadas: {len(archivos_existentes)}")
    print("\n" + "═" * 70 + "\n")

    return True


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":

    # ========================================================================
    # SELECCIONAR MODO DE OPERACIÓN
    # ========================================================================

    if MODO.upper() == 'LOTE':
        # ────────────────────────────────────────────────────────────────────
        # MODO LOTE: Procesar múltiples estrellas → PDF consolidado
        # ────────────────────────────────────────────────────────────────────
        try:
            exito = procesar_lote()
            sys.exit(0 if exito else 1)

        except KeyboardInterrupt:
            print("\n\n⚠️  Proceso interrumpido por el usuario")
            sys.exit(1)

        except Exception as e:
            print(f"\n\n❌ ERROR INESPERADO: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif MODO.upper() == 'INDIVIDUAL':
        # ────────────────────────────────────────────────────────────────────
        # MODO INDIVIDUAL: Procesar una sola estrella con gráficos detallados
        # ────────────────────────────────────────────────────────────────────

        print("\n")
        print("╔" + "═" * 68 + "╗")
        print("║" + " MODO INDIVIDUAL: PROCESAMIENTO DETALLADO ".center(68) + "║")
        print("║" + " Clasificación Espectral Corregida ".center(68) + "║")
        print("╚" + "═" * 68 + "╝")
        print("\n")

        # Verificar que el archivo existe
        archivo_completo = os.path.join(INPUT_DIR, ARCHIVO_ESPECTRO)

        if not os.path.exists(archivo_completo):
            print(f'❌ ERROR: El archivo no existe')
            print(f'   Ruta: {archivo_completo}')
            print(f'\nArchivos disponibles en {INPUT_DIR}:')

            if os.path.exists(INPUT_DIR):
                archivos = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
                for i, f in enumerate(archivos[:10], 1):
                    print(f'  {i:2d}. {f}')
                if len(archivos) > 10:
                    print(f'  ... y {len(archivos) - 10} más')
            else:
                print(f'  (El directorio {INPUT_DIR} no existe)')

            sys.exit(1)

        # Procesar la estrella
        resultados = procesar_estrella(
            ARCHIVO_ESPECTRO,
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            mostrar_graficos=MOSTRAR_GRAFICOS_INDIVIDUAL,
            guardar_graficos=GUARDAR_GRAFICOS_INDIVIDUAL
        )

        if resultados:
            print("\n✅ Procesamiento exitoso")
        else:
            print("\n❌ Error en el procesamiento")
            sys.exit(1)

    else:
        # ────────────────────────────────────────────────────────────────────
        # MODO INVÁLIDO
        # ────────────────────────────────────────────────────────────────────
        print("\n❌ ERROR: MODO inválido")
        print(f"   MODO configurado: '{MODO}'")
        print(f"   Valores válidos: 'INDIVIDUAL' o 'LOTE'")
        print("\nEdita el archivo y cambia la variable MODO en la línea 54")
        sys.exit(1)
