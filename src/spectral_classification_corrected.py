"""
Clasificación Espectral Corregida
==================================

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import UnivariateSpline
from astropy.io import fits
import pandas as pd
import os

# Módulo de clasificación de luminosidad MK (Ia, Ib, II, III, IV, V)
try:
    from luminosity_classification import (
        estimate_luminosity_class,
        combine_spectral_and_luminosity
    )
    _LUMINOSITY_AVAILABLE = True
except ImportError:
    _LUMINOSITY_AVAILABLE = False

    def estimate_luminosity_class(measurements, spectral_type):
        return 'V'

    def combine_spectral_and_luminosity(spectral_type, luminosity_class):
        return spectral_type + luminosity_class


# ============================================================================
# CORRECCIÓN 1: Normalización al continuo con sigma-clipping
# ============================================================================

def sigma_clip(data, sigma=3.0, max_iterations=5):
    """
    Rechaza outliers (rayos cósmicos) usando sigma-clipping robusto (MAD).

    Usa la Desviación Absoluta de la Mediana (MAD) en lugar de la desviación
    estándar clásica, lo que lo hace más resistente a distribuciones asimétricas
    como las que producen los rayos cósmicos en espectros CCD.

    Parameters
    ----------
    data : array
        Datos a limpiar
    sigma : float
        Número de desviaciones estándar robustas para rechazo
    max_iterations : int
        Número máximo de iteraciones

    Returns
    -------
    mask : array
        Máscara booleana (True = dato válido, False = outlier)
    """
    mask = np.ones(len(data), dtype=bool)

    for _ in range(max_iterations):
        if np.sum(mask) < 3:
            break
        median = np.median(data[mask])
        mad = np.median(np.abs(data[mask] - median))
        std_robust = 1.4826 * mad  # Factor de normalización para distribución Gaussiana

        if std_robust < 1e-10:
            break

        new_mask = np.abs(data - median) < sigma * std_robust

        if np.array_equal(mask, new_mask):
            break
        mask = new_mask

    return mask


def normalize_to_continuum(wavelengths, flux, window_size=50, poly_order=3, sigma=3.0):
    """
    Normaliza el espectro al continuo (continuo = 1.0) con enfoque de dos pasos.

    Algoritmo mejorado:
    1. Estimación RÁPIDA del continuo sin sigma-clip global (primer pase)
    2. Rechazo de rayos cósmicos RELATIVO AL CONTINUO (no global), por ventanas
    3. Detección de puntos de continuo excluyendo regiones de absorción
    4. Ajuste final con spline suavizado (s < len(points)) para evitar sobreajuste

    Este orden es crítico: el sigma-clipping global previo no funciona bien en
    espectros sin normalizar donde el flujo varía fuertemente con λ (más azul
    que rojo en estrellas tempranas, o lo contrario en tardías).

    Parameters
    ----------
    wavelengths : array
        Longitudes de onda en Å
    flux : array
        Flujo observado (puede ser en cuentas, erg/s/cm²/Å, etc.)
    window_size : int
        Tamaño de ventana en píxeles para detección de continuo (default: 50)
    poly_order : int
        Orden del spline de ajuste (default: 3 = cúbico)
    sigma : float
        Umbral para sigma-clipping de rayos cósmicos (default: 3.0)

    Returns
    -------
    flux_normalized : array
        Flujo normalizado (continuo ≈ 1.0)
    continuum : array
        Continuo estimado en las mismas unidades que flux
    """
    n_windows = max(len(flux) // window_size, 1)

    # =========================================================================
    # PASO 1: Estimación RÁPIDA del continuo (primer pase sin sigma-clip)
    # =========================================================================
    # Calcular percentil 90 en cada ventana sin rechazar rayos cósmicos aún.
    # Esto da una referencia local del nivel del flujo.
    cont_pts_rough = []
    cont_wave_rough = []

    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, len(flux))
        window_flux = flux[start:end]
        window_wave = wavelengths[start:end]
        if len(window_flux) > 0:
            cont_pts_rough.append(np.percentile(window_flux, 90))
            cont_wave_rough.append(np.median(window_wave))

    if len(cont_pts_rough) < 4:
        rough_continuum = np.full_like(flux, float(np.percentile(flux, 90)))
    else:
        try:
            spl_rough = UnivariateSpline(
                cont_wave_rough, cont_pts_rough,
                k=min(poly_order, len(cont_pts_rough) - 1),
                s=len(cont_pts_rough)
            )
            rough_continuum = spl_rough(wavelengths)
        except Exception:
            rough_continuum = np.interp(wavelengths, cont_wave_rough, cont_pts_rough)

    # Evitar continuo negativo o cero
    rough_continuum = np.maximum(rough_continuum, np.percentile(flux, 5))

    # =========================================================================
    # PASO 2: Rechazo de rayos cósmicos RELATIVO AL CONTINUO (por ventanas)
    # =========================================================================
    # Normalizar provisionalmente para hacer el sigma-clip independiente del
    # nivel de flujo local (necesario para espectros con gradiente azul-rojo).
    flux_ratio = flux / rough_continuum

    clean_mask = np.ones(len(flux), dtype=bool)
    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, len(flux))
        win_ratio = flux_ratio[start:end]

        local_median = np.median(win_ratio)
        local_mad = np.median(np.abs(win_ratio - local_median))
        local_std = 1.4826 * local_mad

        if local_std > 1e-6:
            # Los rayos cósmicos son picos POSITIVOS por encima del continuo
            spike_mask = win_ratio > (local_median + sigma * local_std)
            clean_mask[start:end][spike_mask] = False

    # =========================================================================
    # PASO 3: Detectar puntos de continuo (excluye absorción y rayos cósmicos)
    # =========================================================================
    # Solo se usan puntos LIMPIOS y que estén en el percentil alto local,
    # lo que excluye automáticamente las regiones de absorción.
    continuum_points = []
    continuum_wavelengths = []

    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, len(flux))
        window_flux = flux[start:end]
        window_wave = wavelengths[start:end]
        window_clean = clean_mask[start:end]

        n_clean = np.sum(window_clean)
        if n_clean > 3:
            # Percentil 90 de los puntos limpios = estimación del continuo local
            continuum_level = np.percentile(window_flux[window_clean], 90)
            continuum_points.append(continuum_level)
            continuum_wavelengths.append(np.median(window_wave))

    # =========================================================================
    # PASO 4: Ajuste final del continuo con spline suavizado (s < len(points))
    # =========================================================================
    # s < len(points) previene que el spline pase exactamente por todos los
    # puntos (lo que produciría oscilaciones). El factor 0.5 asegura suavizado
    # adicional en zonas con alta densidad de puntos de absorción.
    if len(continuum_points) < 4:
        valid_flux = flux[clean_mask] if np.sum(clean_mask) > 0 else flux
        continuum = np.full_like(flux, float(np.percentile(valid_flux, 90)))
    else:
        try:
            s_factor = max(len(continuum_points) * 0.5, 1.0)  # s < len(points)
            spline = UnivariateSpline(
                continuum_wavelengths, continuum_points,
                k=min(poly_order, len(continuum_points) - 1),
                s=s_factor
            )
            continuum = spline(wavelengths)
        except Exception:
            continuum = np.interp(wavelengths, continuum_wavelengths, continuum_points)

    # Garantizar continuo positivo (evitar división por cero)
    continuum = np.maximum(continuum, np.percentile(flux, 5))

    # =========================================================================
    # PASO 5: Normalizar
    # =========================================================================
    flux_normalized = flux / continuum

    return flux_normalized, continuum


# ============================================================================
# CORRECCIÓN 2: Cálculo de anchos equivalentes con banderas de calidad
# ============================================================================

# Tolerancia estricta para identificación de líneas (Mejora 1)
LINE_ID_TOLERANCE = 2.0  # Å — |λ_obs - λ_lab| < 2 Å

# Profundidad mínima para considerar absorción real
LINE_MIN_DEPTH = 0.03   # 3% bajo el continuo

# Separación mínima para detectar blend
BLEND_SEPARATION = 3.0  # Å — si otra línea está a < 3 Å: blend

# Ventana reducida en presencia de blend
BLEND_WINDOW = 6.0      # Å

# Posibles valores del campo quality_flag:
#   'OK'                  Medición válida, sin problemas
#   'NOT_DETECTED'        No se detectó absorción significativa en la ventana
#   'UNRELIABLE_POSITION' Mínimo fuera de ±2 Å del centro teórico
#   'TOO_SHALLOW'         Profundidad < 3% (puede ser ruido)
#   'BLENDED'             Medición válida pero contaminada por línea vecina
#   'WINDOW_TOO_SMALL'    Pocos píxeles en la ventana (resolución insuficiente)


def measure_equivalent_width(wavelengths, flux_normalized, line_center, window_width=10.0,
                              tolerance=LINE_ID_TOLERANCE, min_depth=LINE_MIN_DEPTH):
    """
    Mide el ancho equivalente de una línea de absorción con validación y banderas de calidad.

    EW = ∫(1 - F_λ/F_continuo) dλ

    Con el espectro normalizado al continuo (F_continuo = 1.0):
    EW = ∫(1 - F_λ) dλ

    Algoritmo:
    1. Definir ventana de medición (window_width, ±window_width/2 desde line_center)
    2. Localizar el mínimo de absorción dentro de la ventana
    3. Verificar que el mínimo esté dentro de la tolerancia (|Δλ| < tolerance)
    4. Verificar profundidad mínima (> min_depth)
    5. Detectar líneas vecinas: si hay otra a < 3 Å, reducir ventana a 6 Å
    6. Integrar EW por método trapezoidal sobre la ventana final

    Parameters
    ----------
    wavelengths : array
        Longitudes de onda en Å
    flux_normalized : array
        Flujo normalizado al continuo (continuo ≈ 1.0)
    line_center : float
        Longitud de onda laboratorio de la línea en Å
    window_width : float
        Ancho total de la ventana de integración en Å (default: 10.0)
    tolerance : float
        Tolerancia máxima |λ_obs − λ_lab| en Å (default: 2.0 Å)
    min_depth : float
        Profundidad mínima para considerar absorción real (default: 0.03 = 3%)

    Returns
    -------
    ew : float
        Ancho equivalente en Å (positivo para absorción, 0.0 si rechazada)
    line_depth : float
        Profundidad de la línea relativa al continuo (0.0 si rechazada)
    quality_flag : str
        Estado de la medición: 'OK', 'NOT_DETECTED', 'UNRELIABLE_POSITION',
        'TOO_SHALLOW', 'BLENDED', 'WINDOW_TOO_SMALL'
    lambda_measured : float
        Longitud de onda observada del mínimo de absorción (0.0 si no detectada)
    """
    # ── Paso 1: Definir ventana de integración inicial ─────────────────────
    mask = np.abs(wavelengths - line_center) <= window_width / 2

    if np.sum(mask) < 3:
        return 0.0, 0.0, 'WINDOW_TOO_SMALL', 0.0

    wave_window = wavelengths[mask]
    flux_window = flux_normalized[mask]

    # ── Validación 1: Estimar continuo local ──────────────────────────────
    p90 = np.percentile(flux_window, 90)
    cont_pts = flux_window[flux_window > p90]
    continuum_local = float(np.median(cont_pts)) if len(cont_pts) > 0 else 1.0

    # Si el continuo local se desvía mucho de 1.0, usar valor nominal
    if continuum_local < 0.80 or continuum_local > 1.20:
        continuum_local = 1.0

    # ── Paso 2: Localizar el mínimo de absorción ──────────────────────────
    inverted_flux = -flux_window
    peaks_idx, _ = find_peaks(inverted_flux, prominence=0.02, width=1)

    if len(peaks_idx) == 0:
        return 0.0, 0.0, 'NOT_DETECTED', 0.0

    detected_lines = wave_window[peaks_idx]
    detected_depths_flux = flux_window[peaks_idx]  # Flujo en cada mínimo

    # ── Paso 3: Verificar posición — tolerancia estricta |Δλ| < 2 Å ───────
    distances = np.abs(detected_lines - line_center)
    closest_idx = np.argmin(distances)
    closest_line_wave = float(detected_lines[closest_idx])
    closest_line_flux = float(detected_depths_flux[closest_idx])

    if distances[closest_idx] > tolerance:
        # El mínimo de absorción está fuera de la tolerancia → bandera de posición
        return 0.0, 0.0, 'UNRELIABLE_POSITION', closest_line_wave

    # ── Paso 4: Verificar profundidad mínima ──────────────────────────────
    line_depth = 1.0 - closest_line_flux / continuum_local

    if line_depth < min_depth:
        return 0.0, 0.0, 'TOO_SHALLOW', closest_line_wave

    # ── Paso 5: Detectar líneas vecinas (blend) ───────────────────────────
    quality_flag = 'OK'
    other_lines = detected_lines[np.arange(len(detected_lines)) != closest_idx]

    if len(other_lines) > 0:
        min_separation = float(np.min(np.abs(other_lines - closest_line_wave)))
        if min_separation < BLEND_SEPARATION:
            # Otra absorción a menos de 3 Å → blend confirmado
            # Reducir ventana de integración para limitar contaminación
            window_width = min(window_width, BLEND_WINDOW)
            quality_flag = 'BLENDED'

    # ── Cálculo final: integración trapezoidal centrada en λ_observada ────
    mask = np.abs(wavelengths - closest_line_wave) <= window_width / 2
    wave_window = wavelengths[mask]
    flux_window = flux_normalized[mask]

    if len(wave_window) < 3:
        return 0.0, 0.0, 'WINDOW_TOO_SMALL', closest_line_wave

    # EW = ∫(1 - F/F_cont) dλ
    integrand = 1.0 - (flux_window / continuum_local)

    # Detectar perfiles de emisión (supergigantes O/B) — EW negativo permitido
    has_emission = float(np.max(flux_window)) > continuum_local * 1.05
    if not has_emission:
        integrand = np.maximum(integrand, 0.0)

    ew = float(np.trapz(integrand, wave_window))

    return ew, line_depth, quality_flag, closest_line_wave


# ============================================================================
# ESTIMACIÓN DE FWHM PARA VENTANAS ADAPTATIVAS (Mejora 4)
# ============================================================================

def estimate_fwhm(wavelengths, flux_normalized, line_center, window_width=12.0):
    """
    Estima el FWHM (Full Width at Half Maximum) de una línea espectral.

    Se aplica principalmente a líneas metálicas estrechas para calcular
    ventanas de integración adaptativas: window ≈ 3–5 × FWHM.

    Parameters
    ----------
    wavelengths : array
        Longitudes de onda en Å
    flux_normalized : array
        Flujo normalizado al continuo
    line_center : float
        Longitud de onda central aproximada de la línea en Å
    window_width : float
        Ancho de la ventana de búsqueda (default: 12.0 Å)

    Returns
    -------
    fwhm : float
        FWHM estimado en Å. Devuelve 0.0 si no puede estimarse.
    """
    mask = np.abs(wavelengths - line_center) <= window_width / 2
    if np.sum(mask) < 5:
        return 0.0

    wave_win = wavelengths[mask]
    flux_win = flux_normalized[mask]

    # Nivel del continuo local
    continuum_local = float(np.percentile(flux_win, 90))
    if continuum_local < 1e-6:
        return 0.0

    # Localizar el mínimo de la línea
    min_idx = int(np.argmin(flux_win))
    min_flux = float(flux_win[min_idx])

    # Profundidad: si la línea es demasiado superficial, FWHM no es fiable
    depth = 1.0 - min_flux / continuum_local
    if depth < 0.05:
        return 0.0

    # Nivel de semi-máximo (mitad de la profundidad de absorción)
    half_max_flux = min_flux + depth * continuum_local / 2.0

    # Cruce a la izquierda del mínimo
    left_part_flux = flux_win[:min_idx]
    left_cross = np.where(left_part_flux > half_max_flux)[0]
    if len(left_cross) == 0:
        return 0.0
    left_boundary = float(wave_win[left_cross[-1]])

    # Cruce a la derecha del mínimo
    right_part_flux = flux_win[min_idx:]
    right_cross = np.where(right_part_flux > half_max_flux)[0]
    if len(right_cross) == 0:
        return 0.0
    right_boundary = float(wave_win[min_idx + right_cross[0]])

    fwhm = right_boundary - left_boundary

    # Sanidad: FWHM debe ser positivo y menor que la ventana
    if fwhm <= 0.2 or fwhm >= window_width:
        return 0.0

    return fwhm


# ============================================================================
# ESTIMACIÓN DE SNR (Mejora 9)
# ============================================================================

# Umbrales mínimos recomendados
SNR_MINIMUM = 30.0          # SNR mínimo para clasificación fiable
RESOLUTION_MINIMUM = 2000.0  # Resolución espectral mínima R


def compute_snr(wavelengths, flux_normalized, region=None):
    """
    Estima la relación señal-ruido (SNR) del espectro normalizado.

    Usa la variación píxel a píxel en una región relativamente libre de líneas
    para separar señal (mediana) de ruido (dispersión de diferencias sucesivas).

    Parameters
    ----------
    wavelengths : array
        Longitudes de onda en Å
    flux_normalized : array
        Flujo normalizado al continuo
    region : tuple (wave_min, wave_max) or None
        Región de λ para estimar el SNR. Si es None, se busca una región
        limpia automáticamente.

    Returns
    -------
    snr : float
        Relación señal-ruido estimada (0.0 si no se puede calcular)
    """
    # Regiones candidatas relativamente libres de líneas fuertes
    candidate_regions = [(5500, 5600), (4800, 4830), (4400, 4450), (6100, 6200)]

    if region is not None:
        candidate_regions = [region] + candidate_regions

    selected_flux = None
    for r_min, r_max in candidate_regions:
        mask = (wavelengths >= r_min) & (wavelengths <= r_max)
        if np.sum(mask) >= 20:
            selected_flux = flux_normalized[mask]
            break

    if selected_flux is None:
        # Usar tercio central del espectro
        n = len(flux_normalized)
        selected_flux = flux_normalized[n // 3: 2 * n // 3]

    if len(selected_flux) < 10:
        return 0.0

    signal = float(np.median(selected_flux))
    # Ruido estimado por diferencias sucesivas (elimina tendencias de líneas)
    noise = float(np.std(np.diff(selected_flux)) / np.sqrt(2))

    if noise < 1e-10:
        return 999.0  # Espectro perfectamente limpio o sintético

    return signal / noise


# ============================================================================
# Líneas diagnóstico 
# ============================================================================

# Líneas espectrales para clasificación (longitudes de onda en Å)
SPECTRAL_LINES = {
    # Serie de Balmer (Hidrógeno)
    'H_alpha': 6562.8,
    'H_beta': 4861.3,
    'H_gamma': 4340.5,
    'H_delta': 4101.7,
    'H_epsilon': 3970.1,

    # Helio I (estrellas tipo B)
    'He_I_4471': 4471.5,
    'He_I_4387': 4387.9,  # Para O tardías O8-O9 y luminosidad B
    'He_I_4026': 4026.2,
    'He_I_4922': 4921.9,
    'He_I_5016': 5015.7,
    'He_I_5876': 5875.6,
    'He_I_4713': 4713.2,  # Luminosidad O tardía: He II 4686 vs He I 4713
    'He_I_4121': 4120.8,  # Ratio Si IV 4116 / He I 4121 (luminosidad O-B)
    'He_I_4144': 4143.8,  # Temperatura O8-O9: He I 4144 vs He II 4200
                          # (distinto de Fe I 4143.9; en estrellas O domina He I)

    # Helio II (estrellas tipo O)
    'He_II_4686': 4685.7,
    'He_II_4542': 4541.6,
    'He_II_4200': 4199.8,
    'He_II_5411': 5411.5,

    # Nitrógeno (O tempranas)
    'N_V_4604': 4604.0,
    'N_V_4620': 4620.0,
    'N_IV_4058': 4057.8,  # Frecuente en emisión en O tempranas
    'N_III_4634': 4634.0,  # Débil en O3-O5, fuerte en O6-O7
    'N_III_4640': 4640.6,  # Triplete N III: 4634 + 4640 + 4641
    'N_III_4641': 4641.0,  # Par con N III 4634/4640
    'N_II_3995': 3995.0,

    # Oxígeno (O9-B2)
    'O_II_4070': 4069.9,  # Luminosidad B1-B4: He I 4026 / O II 4070-4076
    'O_II_4076': 4075.9,  # Par con O II 4070 (blend en baja resolución)
    'O_II_4348': 4347.4,  # Luminosidad B: ratio Hγ / O II 4348
    'O_II_4591': 4591.0,
    'O_II_4596': 4596.0,
    'O_V_5114': 5114.0,   # Presente en O tempranas (T > 43 kK)
    'O_III_5592': 5592.4, # Muy débil o ausente en O tempranas

    # Silicio (B tempranas)
    'Si_IV_4089': 4088.9,
    'Si_IV_4116': 4116.1,
    'Si_III_4553': 4552.7,
    'Si_III_4568': 4567.9,
    'Si_III_4575': 4574.8,
    'Si_II_4128': 4128.1,
    'Si_II_4131': 4130.9,
    'Si_II_5041': 5041.0,
    'Si_II_5056': 5056.1,

    # Magnesio II (B tardías)
    'Mg_II_4481': 4481.2,

    # Calcio II - importante para tipos A-G
    'Ca_II_K': 3933.7,
    'Ca_II_H': 3968.5,

    # Hierro (B tardías y tipos tardíos)
    'Fe_II_4923': 4923.9,
    'Fe_II_5018': 5018.4,
    'Fe_I_4046': 4045.8,  # Para tipos F-G-K
    'Fe_I_4144': 4143.9,  # Para tipos G-K
    'Fe_I_4260': 4260.5,  # Comparar con Cr I
    'Fe_I_4325': 4325.8,  # Comparar con H gamma
    'Fe_I_4383': 4383.5,  # Para tipos F
    'Fe_I_4957': 4957.5,  # Para tipos K-M (banda TiO)

    # Titanio (B tardías y K-M)
    'Ti_II_4468': 4468.0,

    # Bandas TiO (estrellas tipo M - diagnóstico crítico)
    'TiO_4762': 4762.0,   # Primera banda TiO (K tardío - M temprano)
    'TiO_4955': 4955.0,   # Banda TiO principal (cerca de Fe I 4957)
    'TiO_5167': 5167.0,   # Banda TiO tardía (M intermedio-tardío)

    # Aluminio (B tempranas)
    'Al_III_4529': 4529.0,

    # Carbono (todas O y B)
    'C_II_4267': 4267.2,
    'C_III_4647': 4647.4,  # Calificador "fc" en O: C III 4647-4652 ≈ N III 4634-4642
    'C_III_4652': 4651.5,  # Par con C III 4647
    'C_IV_5801': 5801.3,  # Fuerte en absorción en O tempranas
    'C_IV_5812': 5812.0,  # Par con C IV 5801
    'C_IV_4658': 4658.0,  # Ocasional en O tempranas

    # Calcio I (estrellas F-G-K)
    'Ca_I_4227': 4226.7,  # Importante para F-G-K

    # Estroncio II (indicador de luminosidad A-F: ratio Sr II / Fe I)
    'Sr_II_4077': 4077.7,  # Resonante Sr II — aumenta en supergigantes A-F

    # Hierro I adicional (par de referencia para ratio Sr II / Fe I)
    'Fe_I_4071': 4071.7,   # Fe I 4071 — referencia de gravedad en A-F

    # Cromo (triplete Cr I para temperatura K)
    'Cr_I_4254': 4254.3,
    'Cr_I_4275': 4274.8,  # Triplete Cr I: Cr I 4254 / Fe I 4271 y Cr I 4275 / Fe I 4271
    'Cr_I_4290': 4289.7,  # Triplete Cr I: Cr I 4290 / Fe I 4326

    # Otras metálicas (estrellas tardías)
    'Mg_I_5167': 5167.3,  # Triplete Mg b componente 1 (la más intensa del triplete)
    'Mg_I_5173': 5172.7,
    'Na_I_D1': 5895.9,
    'Na_I_D2': 5889.9,
    # Manganeso (tipo A — blend Mn I indicador de subtipo)
    'Mn_I_4030': 4030.7,  # Blend Mn I 4030 Å — indicador de temperatura en A
    # Hierro I adicional para Cr/Fe ratio en K
    'Fe_I_4250': 4250.1,  # Cociente Cr I 4254 / Fe I 4250 para temperatura K
    # Itrio ionizado — mejor indicador de luminosidad en G-K
    'Y_II_4376': 4376.1,  # Y II 4376 Å vs Fe I 4383 — independiente de metalicidad

    # ── Ampliación: tipo O ──────────────────────────────────────────────
    # C III 4647-4651 blend — presente en O4-O6 (fotosférico en absorción)
    'C_III_4650': 4650.0,

    # ── Ampliación: tipo B ──────────────────────────────────────────────
    # O II en azul: par 4415/4417 Å (Δλ = 2 Å) — B2–B5
    # Útil para refinar subtipo B1-B3 junto a C II 4267 y Si III
    'O_II_4415': 4414.9,   # O II 4415 — fuerte en B2-B3
    'O_II_4417': 4416.9,   # O II 4417 — par con 4415 (blend en baja resolución)

    # ── Ampliación: tipo A ──────────────────────────────────────────────
    # Fe II en Am/supergigantes A: más fuerte que en A estándar
    'Fe_II_4173': 4173.5,  # Fe II 4173 — indicador de metalicidad en A
    'Fe_II_4233': 4233.2,  # Fe II 4233 — crece en A tardías y F tempranas
    # Sr II segundo resonante (par con Sr II 4077)
    'Sr_II_4216': 4215.5,  # Sr II 4216 — luminosidad A-F, par con 4077
    # Mg I 4703: primer Mg I visible en A frías/F tempranas
    'Mg_I_4703': 4702.9,   # Mg I 4703 — aparece en ~8 000 K, crece hacia G
    # Ti II — blends Fe II/Ti II para luminosidad A tardía y F temprana
    'Ti_II_4178': 4177.5,  # Blend Fe II/Ti II 4172-4179 — luminosidad A7-F5
    'Ti_II_4399': 4399.8,  # Blend Fe II/Ti II 4395-4400 — luminosidad A-F
    'Ti_II_4444': 4443.8,  # Blend Ti II/Fe II 4444 — luminosidad F temprana

    # ── Ampliación: tipo F ──────────────────────────────────────────────
    # CH G-band: sistema molecular de CH, centrado en ~4300 Å
    # Aparece en F5 y se refuerza progresivamente hasta G5
    'CH_Gband': 4300.0,    # Banda CH G-band — indicador F/G
    # Fe I adicionales para afinar F/G
    'Fe_I_4271': 4271.2,   # Fe I 4271 — diagnóstico F-G junto con Fe I 4260
    # Ca I 4455: línea de calcio neutro, sensible a T en F-K
    'Ca_I_4455': 4454.8,   # Ca I 4455 — temperatura F-G-K

    # ── Ampliación: tipo G ──────────────────────────────────────────────
    # Triplete Mg I b (5167.3 / 5172.7 / 5183.6 Å)
    # Mg_I_5167 se añadió antes; aquí añadimos b1 (más intensa) y la ya existente b2
    'Mg_I_5183': 5183.6,   # Mg I b1 — componente más intensa del triplete Mg b
    # Hierro en verde-rojo: diagnóstico G-K en espectros de alta cobertura
    'Fe_I_5270': 5269.5,   # Fe I 5270 — referencia en ventana óptico-rojo
    'Fe_I_5328': 5328.0,   # Fe I 5328 — par con Fe I 5270 en G-K
    # Ca I rojo
    'Ca_I_6162': 6162.2,   # Ca I 6162 — gravedad y temperatura en G-K
    # Cianógeno (CN) — luminosidad G-K (efecto positivo en gigantes/supergigantes)
    'CN_4215': 4215.0,     # Cabeza de banda CN violeta 4215 Å (próx. a Sr II 4216)
    'CN_3883': 3883.0,     # Cabeza de banda CN violeta 3883 Å — supergigantes G-K

    # ── Ampliación: tipo K ──────────────────────────────────────────────
    # Ba II 4554: sensible a luminosidad en G-K (mayor en gigantes)
    'Ba_II_4554': 4554.0,  # Ba II 4554 — indicador de luminosidad G-K
    # Ca I rojo adicional (par con Ca I 6162)
    'Ca_I_6122': 6122.2,   # Ca I 6122 — gravedad en K
    # Fe I rojo tardío
    'Fe_I_6495': 6495.0,   # Fe I 6495 — referencia en K tardías / M tempranas

    # ── Ampliación: tipo M — bandas moleculares adicionales ─────────────
    # TiO adicionales:
    'TiO_5448': 5448.0,    # TiO banda 5448 Å — M0–M3
    'TiO_6158': 6158.0,    # TiO roja — crece en M2+
    'TiO_6651': 6651.0,    # TiO roja tardía — M3+
    # VO (óxido de vanadio): aparece en M4-M5, dominante en M6+
    'VO_5736': 5736.0,     # VO 5736 Å — aparece en M5, temperatura temprana M
    'VO_7434': 7434.0,     # VO 7434 Å — aparece en M4-M5
    'VO_7865': 7865.0,     # VO 7865 Å — más fuerte en M5+
    # CaH (hidruro de calcio): enanas M vs gigantes M
    'CaH_6382': 6382.0,    # CaH 6382 Å — más fuerte en enanas M
    'CaH_6750': 6750.0,    # CaH 6750 Å — discrimina subtipos M tardías
    'CaH_6908': 6908.0,    # CaH banda-A 6908 Å — luminosidad M (fuerte en enanas)
    'CaH_6946': 6945.0,    # CaH banda-A 6946 Å — par con 6908, luminosidad M
    # MgH (hidruro de magnesio): enanas K-M vs gigantes
    'MgH_4770': 4770.0,    # MgH + TiO blend 4770 Å — muy sensible a log g en K-M
}

# ---------------------------------------------------------------------------
# Ventanas de integración por línea (Å, ventana TOTAL → ±window/2 del centro)
#
# Balmer (Hidrógeno): líneas anchas por efecto Stark → ventana amplia 20–30 Å
#   Hβ  (4861) → ±10 Å = 20 Å total (recomendado ±15-20 Å en atlas estándar)
#   Hγ  (4340) → ±10 Å = 20 Å total (recomendado ±12-18 Å)
#   Hδ  (4102) → ±10 Å = 20 Å total (recomendado ±10-15 Å)
#   Hε  (3970) → ±8  Å = 16 Å (ligeramente más estrecha, vecindad de Ca II H)
#
# Pares cercanos metálicos: evitar contaminación cruzada
#   Fe I 4071 / Sr II 4077 → sólo 6 Å de separación → ventana 8 Å (±4 Å)
#   Fe I 4071: cubre 4067.7–4075.7  │  Sr II 4077: cubre 4073.7–4081.7
#   (solapamiento mínimo en bordes, aceptable para el ratio Sr/Fe)
#
# Por defecto (todas las demás líneas): 12 Å (±6 Å)
#   Reducido de 15 → 12 para mejorar la selectividad en regiones densas
# ---------------------------------------------------------------------------
LINE_WINDOWS = {
    # Balmer — ventana amplia para capturar las alas ensanchadas por Stark
    'H_alpha':   30.0,   # ±15 Å — más ancha: región libre de otras líneas
    'H_beta':    20.0,   # ±10 Å
    'H_gamma':   20.0,   # ±10 Å
    'H_delta':   20.0,   # ±10 Å
    'H_epsilon': 16.0,   # ±8  Å — más estrecha: cercana a Ca II H (3968.5)

    # Pares metálicos cercanos — ventana estrecha para evitar contaminación
    'Fe_I_4071':  8.0,   # ±4 Å  — par con Sr II 4077 (Δλ = 6 Å)
    'Sr_II_4077': 8.0,   # ±4 Å  — par con Fe I 4071

    # O II cercanas entre sí (4591 / 4596, Δλ = 5 Å)
    'O_II_4591':  8.0,
    'O_II_4596':  8.0,

    # N III par (4634 / 4641, Δλ = 7 Å)
    'N_III_4634': 10.0,
    'N_III_4641': 10.0,

    # O II par muy cercano (Δλ = 2 Å) → ventana mínima para evitar blend total
    'O_II_4415':  6.0,
    'O_II_4417':  6.0,

    # CH G-band: banda difusa de ~20 Å de anchura real → ventana amplia
    'CH_Gband':  25.0,

    # Sr II 4216 / Fe II 4173 / Fe II 4233: sin vecinos problemáticos → defecto (12)
    # Mg I 4703: aislada → defecto (12)

    # Bandas moleculares adicionales — amplia para capturar el contorno de banda
    'TiO_5448':  20.0,
    'TiO_6158':  20.0,
    'TiO_6651':  20.0,
    'VO_7434':   20.0,
    'VO_7865':   20.0,
    'CaH_6382':  15.0,
    'CaH_6750':  15.0,
}

# ---------------------------------------------------------------------------
# Clasificación de líneas por tipo (Mejora 3) — define el rango de ventana
# y si se aplica FWHM adaptativo (solo para líneas metálicas estrechas).
#
# Categorías y ventanas nominales:
#   Balmer       → 20–60 Å (alas de Stark muy anchas)
#   HeI_strong   → 10–15 Å
#   HeI_weak     → 6–10 Å
#   metal_lines  → ≤ 4 Å con FWHM adaptativo: window ≈ 3–5 × FWHM
#   CaI_MgI_NaI → 10–15 Å
#   molecular    → 15–25 Å (contornos de banda molecular)
#   other        → 12 Å (defecto)
# ---------------------------------------------------------------------------
LINE_TYPES = {
    # Serie de Balmer
    'H_alpha':    'Balmer',
    'H_beta':     'Balmer',
    'H_gamma':    'Balmer',
    'H_delta':    'Balmer',
    'H_epsilon':  'Balmer',

    # Helio I — fuertes (visibles en O9 – B5)
    'He_I_4471':  'HeI_strong',
    'He_I_4026':  'HeI_strong',
    'He_I_5876':  'HeI_strong',
    'He_I_4713':  'HeI_strong',   # Luminosidad O tardía: He II 4686 vs He I 4713

    # Helio I — débiles
    'He_I_4387':  'HeI_weak',
    'He_I_4922':  'HeI_weak',
    'He_I_5016':  'HeI_weak',
    'He_I_4121':  'HeI_weak',     # Si IV 4116 / He I 4121 — luminosidad O-B0
    'He_I_4144':  'HeI_weak',     # He I 4144 — temperatura O8-O9 (distinto de Fe I)

    # Helio II — ionizado (tipo O)
    'He_II_4686': 'HeI_strong',
    'He_II_4542': 'HeI_strong',
    'He_II_4200': 'HeI_strong',
    'He_II_5411': 'HeI_strong',

    # Ca II H y K — resonantes, ventana como CaI_MgI_NaI
    'Ca_II_K':    'CaI_MgI_NaI',
    'Ca_II_H':    'CaI_MgI_NaI',

    # Ca I
    'Ca_I_4227':  'CaI_MgI_NaI',
    'Ca_I_4455':  'CaI_MgI_NaI',
    'Ca_I_6122':  'CaI_MgI_NaI',
    'Ca_I_6162':  'CaI_MgI_NaI',

    # Mg I — triplete b
    'Mg_I_5173':  'CaI_MgI_NaI',
    'Mg_I_5183':  'CaI_MgI_NaI',
    'Mg_I_4703':  'CaI_MgI_NaI',

    # Na I D — doblete resonante
    'Na_I_D1':    'CaI_MgI_NaI',
    'Na_I_D2':    'CaI_MgI_NaI',

    # Bandas moleculares
    'TiO_4762':   'molecular',
    'TiO_4955':   'molecular',
    'TiO_5167':   'molecular',
    'TiO_5448':   'molecular',
    'TiO_6158':   'molecular',
    'TiO_6651':   'molecular',
    'VO_5736':    'molecular',
    'VO_7434':    'molecular',
    'VO_7865':    'molecular',
    'CaH_6382':   'molecular',
    'CaH_6750':   'molecular',
    'CaH_6908':   'molecular',
    'CaH_6946':   'molecular',
    'MgH_4770':   'molecular',
    'CN_4215':    'molecular',
    'CN_3883':    'molecular',
    'CH_Gband':   'molecular',

    # Líneas metálicas estrechas — FWHM adaptativo
    'Fe_I_4046':  'metal_lines',
    'Fe_I_4071':  'metal_lines',
    'Fe_I_4144':  'metal_lines',
    'Fe_I_4260':  'metal_lines',
    'Fe_I_4271':  'metal_lines',
    'Fe_I_4325':  'metal_lines',
    'Fe_I_4383':  'metal_lines',
    'Fe_I_4957':  'metal_lines',
    'Fe_I_5270':  'metal_lines',
    'Fe_I_5328':  'metal_lines',
    'Fe_I_6495':  'metal_lines',
    'Fe_II_4173': 'metal_lines',
    'Fe_II_4233': 'metal_lines',
    'Fe_II_4923': 'metal_lines',
    'Fe_II_5018': 'metal_lines',
    'Cr_I_4254':  'metal_lines',
    'Cr_I_4275':  'metal_lines',
    'Cr_I_4290':  'metal_lines',
    'Fe_I_4250':  'metal_lines',
    'Y_II_4376':  'metal_lines',
    'Mn_I_4030':  'metal_lines',
    'Ti_II_4178': 'metal_lines',
    'Ti_II_4399': 'metal_lines',
    'Ti_II_4444': 'metal_lines',
    'Mg_I_5167':  'CaI_MgI_NaI',
    'Ba_II_4554': 'metal_lines',
    'Sr_II_4077': 'metal_lines',
    'Sr_II_4216': 'metal_lines',
    'Ti_II_4468': 'metal_lines',
    'Mg_II_4481': 'metal_lines',
    'Si_II_4128': 'metal_lines',
    'Si_II_4131': 'metal_lines',
    'Si_II_5041': 'metal_lines',
    'Si_II_5056': 'metal_lines',
    'Si_III_4553':'metal_lines',
    'Si_III_4568':'metal_lines',
    'Si_III_4575':'metal_lines',
    'Si_IV_4089': 'metal_lines',
    'Si_IV_4116': 'metal_lines',
    'C_II_4267':  'metal_lines',
    'C_IV_5801':  'metal_lines',
    'N_II_3995':  'metal_lines',
    'N_III_4634': 'metal_lines',
    'N_III_4640': 'metal_lines',
    'N_III_4641': 'metal_lines',
    'N_IV_4058':  'metal_lines',
    'N_V_4604':   'metal_lines',
    'N_V_4620':   'metal_lines',
    'O_II_4070':  'metal_lines',
    'O_II_4076':  'metal_lines',
    'O_II_4348':  'metal_lines',
    'O_II_4415':  'metal_lines',
    'O_II_4417':  'metal_lines',
    'O_II_4591':  'metal_lines',
    'O_II_4596':  'metal_lines',
    'O_III_5592': 'metal_lines',
    'Al_III_4529':'metal_lines',
    'C_III_4647': 'metal_lines',
    'C_III_4650': 'metal_lines',
    'C_III_4652': 'metal_lines',
    'C_IV_4658':  'metal_lines',
    'O_V_5114':   'metal_lines',
}


def measure_diagnostic_lines(wavelengths, flux_normalized, use_fwhm_adaptation=True,
                              min_snr=0.0, spectral_resolution=0.0):
    """
    Mide anchos equivalentes de líneas diagnóstico para clasificación espectral.

    Mejoras implementadas:
    - Devuelve quality_flag y lambda_measured para cada línea
    - Usa ventanas dependientes del tipo de línea (LINE_TYPES / LINE_WINDOWS)
    - Adapta ventanas de líneas metálicas basándose en FWHM estimado (Mejora 4)
    - Registra el ancho de ventana usado para diagnóstico

    Parameters
    ----------
    wavelengths : array
        Longitudes de onda en Å
    flux_normalized : array
        Flujo normalizado al continuo
    use_fwhm_adaptation : bool
        Si True, adapta ventanas de líneas metálicas por FWHM (default: True)
    min_snr : float
        SNR mínimo esperado. Solo informativo (no filtra mediciones).
    spectral_resolution : float
        Resolución espectral R (λ/Δλ). Solo informativa.

    Returns
    -------
    measurements : dict
        Diccionario con, para cada línea:
          'ew'            : float — ancho equivalente en Å
          'depth'         : float — profundidad relativa al continuo
          'wavelength'    : float — longitud de onda de laboratorio en Å
          'lambda_measured': float — longitud de onda del mínimo observado en Å
          'quality_flag'  : str   — 'OK', 'NOT_DETECTED', 'UNRELIABLE_POSITION', etc.
          'window_width'  : float — ventana de integración usada en Å
    """
    measurements = {}
    wave_min, wave_max = float(wavelengths[0]), float(wavelengths[-1])

    for line_name, line_wave in SPECTRAL_LINES.items():
        if wave_min <= line_wave <= wave_max:
            # Ventana base según tipo de línea y diccionario específico
            win = LINE_WINDOWS.get(line_name, 12.0)

            # ── Adaptación de ventana por FWHM (solo para líneas metálicas) ──
            if use_fwhm_adaptation and LINE_TYPES.get(line_name) == 'metal_lines':
                fwhm = estimate_fwhm(wavelengths, flux_normalized, line_wave,
                                     window_width=win)
                if fwhm > 0.5:  # FWHM válido estimado
                    # Ventana adaptativa: 3–5 × FWHM, acotada al rango [4.0, win]
                    adaptive_win = np.clip(4.0 * fwhm, 4.0, win)
                    win = float(adaptive_win)

            ew, depth, quality_flag, lambda_measured = measure_equivalent_width(
                wavelengths, flux_normalized, line_wave, window_width=win
            )
            measurements[line_name] = {
                'ew': ew,
                'depth': depth,
                'wavelength': line_wave,
                'lambda_measured': lambda_measured,
                'quality_flag': quality_flag,
                'window_width': win,
            }
        else:
            # Línea fuera del rango espectral cubierto
            measurements[line_name] = {
                'ew': 0.0,
                'depth': 0.0,
                'wavelength': line_wave,
                'lambda_measured': 0.0,
                'quality_flag': 'OUT_OF_RANGE',
                'window_width': LINE_WINDOWS.get(line_name, 12.0),
            }

    return measurements


def compute_spectral_ratios(measurements: dict) -> dict:
    """
    Calcula razones diagnóstico entre anchos equivalentes de líneas espectrales.

    Las razones de líneas son más robustas que los EW individuales porque
    cancelan parcialmente los efectos de la normalización del continuo, el
    enrojecimiento interestelar y la relación señal/ruido.

    Fundamento astrofísico
    ----------------------
    Cada razón está calibrada con el criterio MK estándar (Gray & Corbally 2009):

    * ``HeI_HeII``   — termómetro de temperatura en tipo O (EW He I 4471 / He II 4542).
                       Aumenta de O3 → O9 al bajar T y despoblarse He II.
    * ``MgII_HeI``   — temperatura en tipo B (EW Mg II 4481 / He I 4471).
                       Mg II crece al bajar T porque los metales se pueden excitar
                       a T < 20 000 K.
    * ``SiIII_SiII`` — temperatura en B temprana (EW Si III 4553 / Si II 4128).
                       Secuencia de ionización: Si IV > Si III > Si II al bajar T.
    * ``SiIV_SiIII`` — temperatura en B0 / O9 (EW Si IV 4089 / Si III 4553).
    * ``CaIIK_Heps`` — temperatura A → G (EW Ca II K / Hε 3970).
                       Ca II K crece fuertemente de A0 a F; Hε decrece.
    * ``CrI_FeI``    — temperatura G (EW Cr I 4254 / Fe I 4260).
                       Cr I crece relativo a Fe I al bajar T (G0→K).
    * ``FeI_H``      — temperatura G → K: promedio Fe I / promedio Balmer.
    * ``NIV_NIII``   — temperatura O temprana (N IV 4058 / N III 4634-4641 avg).
                       N IV solo en O3-O4 (T ≥ 40 000 K).
    * ``NIII_HeII``  — luminosidad en O (N III / He II): mayor en supergigantes O.
    * ``SrII_FeI``   — luminosidad en A-F (Sr II 4077 / Fe I 4071).
                       Sr II crece a menor log g.
    * ``CaI_FeI``    — luminosidad en G-K (Ca I 4227 / Fe I 4383).
                       Ca I disminuye relativo a Fe I en supergigantes G-K.
    * ``BaII_FeI``   — luminosidad en G-K (Ba II 4554 / Fe I 4260).
                       Ba II crece en enanas y gigantes frías.
    * ``TiO_index``  — índice de temperatura M: suma EW bandas TiO.
    * ``VO_index``   — índice de temperatura M tardía: suma EW bandas VO.
    * ``CaH_index``  — discrimina enanas M de gigantes M (CaH fuerte → enana).
    * ``CHband_FeI`` — temperatura F/G (CH G-band / Fe I 4260).

    Parameters
    ----------
    measurements : dict
        Diccionario de mediciones tal como lo devuelve
        ``measure_diagnostic_lines()``.  Cada valor puede ser:
        * Formato interno: ``{"ew": float, "depth": float}``
        * Formato plano: ``{"He_I_4471": float, ...}``

    Returns
    -------
    dict
        Diccionario con claves de ratio (str) → valor (float).
        Razones indefinidas (denominador ≈ 0) se devuelven como ``0.0``
        o ``99.0`` según corresponda (ver ``safe_ratio``).

    Examples
    --------
    >>> from src.spectral_classification_corrected import compute_spectral_ratios
    >>> meas = {'He_I_4471': {'ew': 0.8}, 'He_II_4542': {'ew': 0.4}}
    >>> r = compute_spectral_ratios(meas)
    >>> r['HeI_HeII']   # → 2.0
    2.0
    """
    # ── Extractor interno compatible con ambos formatos ───────────────────
    def ew(key: str) -> float:
        entry = measurements.get(key)
        if entry is None:
            return 0.0
        if isinstance(entry, dict):
            return float(entry.get('ew', 0.0) or 0.0)
        return float(entry)

    def safe_ratio(num: float, den: float, max_val: float = 99.0) -> float:
        """Razón segura que evita división por cero."""
        if den < 1e-4:
            return max_val if num > 1e-4 else 0.0
        return min(num / den, max_val)

    # ── Promedios auxiliares ─────────────────────────────────────────────
    # He II: promedio de las dos líneas más usadas
    he2_avg = (ew('He_II_4542') + ew('He_II_4686')) / 2.0 or ew('He_II_4542')

    # N III: promedio del doblete 4634/4641
    niii_avg = (ew('N_III_4634') + ew('N_III_4641')) / 2.0

    # Fe I: promedio de las tres líneas de referencia azul
    fe1_avg = (ew('Fe_I_4260') + ew('Fe_I_4383') + ew('Fe_I_4046')) / 3.0

    # Balmer: promedio de Hβ, Hγ, Hδ
    h_avg = (ew('H_beta') + ew('H_gamma') + ew('H_delta')) / 3.0

    # Si II: promedio del doblete 4128/4131
    si2_avg = (ew('Si_II_4128') + ew('Si_II_4131')) / 2.0

    # TiO total (azul + rojo)
    tio_total = (ew('TiO_4762') + ew('TiO_4955') + ew('TiO_5167')
                 + ew('TiO_5448') + ew('TiO_6158') + ew('TiO_6651'))

    # ── Razones de temperatura ────────────────────────────────────────────
    ratios = {}

    # Tipo O
    ratios['HeI_HeII']    = safe_ratio(ew('He_I_4471'), he2_avg)
    ratios['NIV_NIII']    = safe_ratio(ew('N_IV_4058'), niii_avg)
    ratios['NIII_HeII']   = safe_ratio(niii_avg, he2_avg)

    # Tipo B
    ratios['MgII_HeI']    = safe_ratio(ew('Mg_II_4481'), ew('He_I_4471'))
    ratios['SiIV_SiIII']  = safe_ratio(ew('Si_IV_4089'), ew('Si_III_4553'))
    ratios['SiIII_SiII']  = safe_ratio(ew('Si_III_4553'), si2_avg)
    ratios['CII_OII']     = safe_ratio(ew('C_II_4267'),   ew('O_II_4415'))

    # Tipos A-F
    ratios['CaIIK_Heps']  = safe_ratio(ew('Ca_II_K'),     ew('H_epsilon'))
    ratios['SrII_FeI']    = safe_ratio(ew('Sr_II_4077'),  ew('Fe_I_4071'))
    ratios['SrII4216_FeI'] = safe_ratio(ew('Sr_II_4216'), ew('Fe_I_4383'))

    # Tipo G
    ratios['CrI_FeI']     = safe_ratio(ew('Cr_I_4254'),   ew('Fe_I_4260'))
    ratios['FeI_H']        = safe_ratio(fe1_avg,            h_avg)
    ratios['CHband_FeI']  = safe_ratio(ew('CH_Gband'),    ew('Fe_I_4260'))
    ratios['MgIb_FeI']    = safe_ratio(ew('Mg_I_5183') + ew('Mg_I_5173'),
                                        ew('Fe_I_5270') + 0.01)

    # Luminosidad G-K
    ratios['CaI_FeI']     = safe_ratio(ew('Ca_I_4227'),   ew('Fe_I_4383'))
    ratios['BaII_FeI']    = safe_ratio(ew('Ba_II_4554'),  ew('Fe_I_4260'))
    ratios['NaI_CaI']     = safe_ratio(
        (ew('Na_I_D1') + ew('Na_I_D2')) / 2.0, ew('Ca_I_4227'))

    # Tipo M — índices moleculares
    ratios['TiO_index']   = tio_total
    ratios['VO_index']    = ew('VO_7434') + ew('VO_7865')
    ratios['CaH_index']   = ew('CaH_6382') + ew('CaH_6750')
    # Razón TiO/CaH: gigantes M tienen TiO fuerte y CaH débil
    ratios['TiO_CaH']     = safe_ratio(tio_total,
                                        ratios['CaH_index'] + 0.01)

    # ── Razones adicionales para tipos O-B tempranos (Mejora 10) ─────────
    # Estas razones refinan la clasificación de subtipos O y B.
    #
    # He I 4471 / He II 4541 (Gray & Corbally 2009, criterio canónico O)
    #   Aumenta de O3 → O9 al disminuir T y despoblarse He II.
    #   < 1.0 → O temprana; ≈ 1 → O7; > 1 → O tardía / B0
    ratios['HeI4471_HeII4541'] = safe_ratio(ew('He_I_4471'), ew('He_II_4542'))

    # N III 4640 / He II 4686  — luminosidad en tipo O (criterio "Onfp")
    #   Mayor en supergigantes O (viento estelar más intenso)
    ratios['NIII4640_HeII4686'] = safe_ratio(niii_avg, ew('He_II_4686'))

    # Si IV 4089 / He I 4026  — temperatura B0–B2
    #   Si IV crece a mayor T; He I 4026 referencia en tipos O tardíos–B
    ratios['SiIV4089_HeI4026']  = safe_ratio(ew('Si_IV_4089'), ew('He_I_4026'))

    # N V 4604 / N III 4634   — temperatura O temprana muy caliente
    #   N V solo aparece en O3-O5 (T > 40 000 K)
    nv_avg = (ew('N_V_4604') + ew('N_V_4620')) / 2.0
    ratios['NV_NIII']           = safe_ratio(nv_avg, niii_avg)

    # O III 5592 / He I 5876  — indicador secundario en B0-B2
    ratios['OIII5592_HeI5876']  = safe_ratio(ew('O_III_5592'), ew('He_I_5876'))

    # Si IV 4116 / Si III 4553 — temperatura B0.5 (complemento SiIV_SiIII)
    ratios['SiIV4116_SiIII4553'] = safe_ratio(ew('Si_IV_4116'), ew('Si_III_4553'))

    # ── Razones añadidas para completar criterios MK ──────────────────────

    # Luminosidad O tardía: He II 4686 vs He I 4713
    #   Clase V: He II >> He I; Clase I: He II en emisión (EW < 0)
    ratios['HeII4686_HeI4713']  = safe_ratio(ew('He_II_4686'), ew('He_I_4713') + 0.01)

    # Temperatura O8-O9: He I 4387 / He I 4144
    #   En O9: He I 4387 ≈ He II 4542; en O8: He II 4200 > He I 4387 y 4144
    ratios['HeI4387_HeI4144']   = safe_ratio(ew('He_I_4387'),  ew('He_I_4144') + 0.01)

    # Luminosidad O-B0: Si IV 4116 / He I 4121
    #   Si IV crece con luminosidad en B0–B0.7; válido hasta B0.7
    ratios['SiIV4116_HeI4121']  = safe_ratio(ew('Si_IV_4116'), ew('He_I_4121') + 0.01)

    # Luminosidad B1-B4: O II 4070-4076 vs He I 4026
    #   O II crece fuertemente en supergigantes B tempranas
    oii_4070_avg = (ew('O_II_4070') + ew('O_II_4076')) / 2.0
    ratios['OII4070_HeI4026']   = safe_ratio(oii_4070_avg, ew('He_I_4026') + 0.01)

    # Luminosidad B: Hγ / O II 4348
    #   Hγ decrece con luminosidad; O II 4348 crece → ratio disminuye en supergigantes
    ratios['Hgamma_OII4348']    = safe_ratio(ew('H_gamma'), ew('O_II_4348') + 0.01)

    # Luminosidad G-K: Y II 4376 / Fe I 4383
    #   Y II crece con luminosidad (menor log g); independiente de metalicidad
    ratios['YII4376_FeI4383']   = safe_ratio(ew('Y_II_4376'), ew('Fe_I_4383') + 0.01)

    # Temperatura K: triplete Cr I
    #   Cr I 4254 / Fe I 4250 y Cr I 4275 / Fe I 4271 crecen de G → K
    ratios['CrI4275_FeI4271']   = safe_ratio(ew('Cr_I_4275'), ew('Fe_I_4271') + 0.01)
    ratios['CrI4290_FeI4326']   = safe_ratio(ew('Cr_I_4290'), ew('Fe_I_4325') + 0.01)
    ratios['CrI4254_FeI4250']   = safe_ratio(ew('Cr_I_4254'), ew('Fe_I_4250') + 0.01)

    # Luminosidad A-F: blends Fe II/Ti II
    #   Estos blends crecen fuertemente en supergigantes A tardías y F tempranas
    tiif_avg = (ew('Ti_II_4178') + ew('Ti_II_4399')) / 2.0
    ratios['TiIIFeII_FeI']      = safe_ratio(tiif_avg, ew('Fe_I_4046') + 0.01)

    # Luminosidad M: CaH vs TiO
    #   CaH se debilita en gigantes; ratio CaH/(CaH+TiO) → 0 en gigantes
    cah_total = ew('CaH_6382') + ew('CaH_6750') + ew('CaH_6908') + ew('CaH_6946')
    ratios['CaH_extended']      = cah_total
    ratios['CaH_TiO_ratio']     = safe_ratio(cah_total, tio_total + 0.01)

    # Índice VO extendido (incluyendo VO 5736 para M tempranas)
    ratios['VO_index_ext']      = ew('VO_5736') + ew('VO_7434') + ew('VO_7865')

    return ratios


def detect_FeI_4957_asymmetry(wavelengths, flux_normalized):
    """
    Detecta la asimetría de Fe I 4957 causada por TiO 4955 (diagnóstico tipo M).

    Según el esquema de clasificación estándar:
    - K5: Fe I 4957 simétrica (sin TiO significativo)
    - M0: Fe I 4957 asimétrica (flujo izquierda > derecha, diferencia ~50% de la profundidad)
    - M4: Fe I 4957 casi desaparece, dominada por escalón TiO 4955
    - M6+: Espectro dominado por bandas TiO (apariencia de "dientes de sierra")

    Parameters
    ----------
    wavelengths : array
        Longitudes de onda en Å
    flux_normalized : array
        Flujo normalizado al continuo

    Returns
    -------
    asymmetry_ratio : float
        Ratio de asimetría (0 = simétrica, >0.1 = asimétrica tipo M)
    flux_left : float
        Flujo promedio a la izquierda de 4957 (4952-4957 Å)
    flux_right : float
        Flujo promedio a la derecha de 4957 (4957-4962 Å)
    has_tio_band : bool
        True si se detecta banda TiO significativa
    """
    # Definir región alrededor de Fe I 4957
    mask_region = (wavelengths >= 4950) & (wavelengths <= 4965)

    if np.sum(mask_region) < 10:
        # Región no cubierta
        return 0.0, 1.0, 1.0, False

    wave_region = wavelengths[mask_region]
    flux_region = flux_normalized[mask_region]

    # Buscar el mínimo de la línea Fe I 4957 (±2 Å de tolerancia)
    mask_fe = (wave_region >= 4955) & (wave_region <= 4960)
    if np.sum(mask_fe) < 3:
        return 0.0, 1.0, 1.0, False

    idx_fe_min = np.argmin(flux_region[mask_fe])
    idx_fe_global = np.where(mask_region)[0][np.where(mask_fe)[0][idx_fe_min]]
    wave_fe_min = wavelengths[idx_fe_global]

    # Definir ventanas a izquierda y derecha del mínimo (±2.5 Å)
    # Izquierda: región 4952-4957 (puede estar elevada por TiO 4955 si es tipo M)
    # Derecha:  región 4957-4962 (menos afectada por TiO)
    mask_left = (wavelengths >= wave_fe_min - 5.0) & (wavelengths < wave_fe_min)
    mask_right = (wavelengths > wave_fe_min) & (wavelengths <= wave_fe_min + 5.0)

    if np.sum(mask_left) < 3 or np.sum(mask_right) < 3:
        return 0.0, 1.0, 1.0, False

    # Calcular flujo promedio en cada lado
    # Usar percentil 75 para minimizar efecto de otras líneas
    flux_left = np.percentile(flux_normalized[mask_left], 75)
    flux_right = np.percentile(flux_normalized[mask_right], 75)

    # Calcular profundidad de la línea
    line_depth = 1.0 - flux_normalized[idx_fe_global]

    # Ratio de asimetría: (flujo_izq - flujo_der) / profundidad_línea
    # En tipo M, TiO 4955 "eleva" el flujo a la izquierda de Fe I 4957
    # creando asimetría característica
    if line_depth > 0.05:
        asymmetry_ratio = (flux_left - flux_right) / line_depth
    else:
        # Línea muy débil o ausente (tipo M tardío donde Fe I desaparece)
        asymmetry_ratio = flux_left - flux_right

    # Detectar si hay banda TiO significativa
    # Criterio: asimetría > 0.3 (30% de la profundidad) O flujo_izq - flujo_der > 0.1
    has_tio_band = (asymmetry_ratio > 0.3) or (flux_left - flux_right > 0.1)

    return asymmetry_ratio, flux_left, flux_right, has_tio_band


def evaluate_tio_bands(wavelengths, flux_normalized, measurements):
    """
    Evalúa la presencia e intensidad de bandas TiO para clasificación tipo M.

    Según el esquema:
    - TiO visible en 4762, 4955, 5167 → tipo M
    - TiO moderado → M0-M2
    - TiO fuerte → M2-M4
    - TiO muy fuerte (dientes de sierra) → M4-M7

    Parameters
    ----------
    wavelengths : array
        Longitudes de onda
    flux_normalized : array
        Flujo normalizado
    measurements : dict
        Mediciones de líneas (de measure_diagnostic_lines)

    Returns
    -------
    tio_strength : str
        'ausente', 'debil', 'moderado', 'fuerte', 'muy_fuerte'
    n_bands_detected : int
        Número de bandas TiO detectadas (0-3)
    tio_avg_depth : float
        Profundidad promedio de bandas TiO
    """
    # Obtener mediciones de TiO
    TiO_4762 = measurements.get('TiO_4762', {}).get('ew', 0.0)
    TiO_4955 = measurements.get('TiO_4955', {}).get('ew', 0.0)
    TiO_5167 = measurements.get('TiO_5167', {}).get('ew', 0.0)

    TiO_4762_depth = measurements.get('TiO_4762', {}).get('depth', 0.0)
    TiO_4955_depth = measurements.get('TiO_4955', {}).get('depth', 0.0)
    TiO_5167_depth = measurements.get('TiO_5167', {}).get('depth', 0.0)

    # Contar bandas detectadas (umbral: profundidad > 5%)
    n_bands_detected = sum([
        TiO_4762_depth > 0.05,
        TiO_4955_depth > 0.05,
        TiO_5167_depth > 0.05
    ])

    # Promedio de profundidades (solo bandas detectadas)
    depths = [d for d in [TiO_4762_depth, TiO_4955_depth, TiO_5167_depth] if d > 0.05]
    tio_avg_depth = np.mean(depths) if depths else 0.0

    # Clasificar intensidad
    if n_bands_detected == 0 or tio_avg_depth < 0.05:
        tio_strength = 'ausente'
    elif n_bands_detected == 1 or tio_avg_depth < 0.15:
        tio_strength = 'debil'  # K tardío o M0
    elif n_bands_detected >= 2 and tio_avg_depth < 0.30:
        tio_strength = 'moderado'  # M0-M2
    elif tio_avg_depth < 0.50:
        tio_strength = 'fuerte'  # M2-M4
    else:
        tio_strength = 'muy_fuerte'  # M4-M7

    return tio_strength, n_bands_detected, tio_avg_depth


# ============================================================================
# CLASIFICADOR BASADO EN ÁRBOL DE DECISIÓN (ESQUEMA ESTÁNDAR)
# ============================================================================
# Árbol de decisión espectral obligatorio:
# 1. Clasificación gruesa (aspecto global)
# 2. Estrellas O, B y A (buscar He I)
# 3. Estrellas intermedias (He I → B tardía, Ca II ≈ Hε → F)
# 4. Estrellas tardías (Balmer → F, Ca I + TiO → K-M)
# Luego clasificación detallada por tipo (O, B, A, F, G-K, K-M)
# ============================================================================

def classify_star_decision_tree(measurements, wavelengths=None, flux_normalized=None):
    """
    Clasifica una estrella usando el árbol de decisión estándar OBLIGATORIO.

    Returns
    -------
    spectral_type : str
        Tipo espectral (O, B, A, F, G, K, M)
    subtype : str
        Subtipo (ej. B2, G5, M4)
    diagnostics : dict
        Contiene:
        - 'lineas_usadas': lista de líneas usadas en la clasificación
        - 'justificacion': texto explicando la clasificación
        - 'advertencias': lista de advertencias si hay ambigüedad
        - 'confianza': nivel de confianza 0-100
        - Todas las mediciones relevantes
    """
    # ============================================================
    # EXTRAER MEDICIONES
    # ============================================================
    # He II (tipo O): 4200, 4542, 4686 Å
    He_II_4686 = measurements.get('He_II_4686', {}).get('ew', 0.0)
    He_II_4542 = measurements.get('He_II_4542', {}).get('ew', 0.0)
    He_II_4200 = measurements.get('He_II_4200', {}).get('ew', 0.0)

    # He I (tipos O-B): 4026, 4471, 4922, 5016 Å
    He_I_4471 = measurements.get('He_I_4471', {}).get('ew', 0.0)
    He_I_4387 = measurements.get('He_I_4387', {}).get('ew', 0.0)
    He_I_4026 = measurements.get('He_I_4026', {}).get('ew', 0.0)
    He_I_4922 = measurements.get('He_I_4922', {}).get('ew', 0.0)
    He_I_5016 = measurements.get('He_I_5016', {}).get('ew', 0.0)

    # Hidrógeno Balmer: Hε 3970, Hδ 4102, Hγ 4341, Hβ 4861
    H_epsilon = measurements.get('H_epsilon', {}).get('ew', 0.0)
    H_delta = measurements.get('H_delta', {}).get('ew', 0.0)
    H_gamma = measurements.get('H_gamma', {}).get('ew', 0.0)
    H_beta = measurements.get('H_beta', {}).get('ew', 0.0)

    # Silicio (subtipos B)
    Si_IV_4089 = measurements.get('Si_IV_4089', {}).get('ew', 0.0)
    Si_III_4553 = measurements.get('Si_III_4553', {}).get('ew', 0.0)
    Si_II_4128 = measurements.get('Si_II_4128', {}).get('ew', 0.0)

    # Magnesio (B tardías): Mg II 4481
    Mg_II_4481 = measurements.get('Mg_II_4481', {}).get('ew', 0.0)

    # Calcio: Ca II K 3933, Ca I 4227
    Ca_II_K = measurements.get('Ca_II_K', {}).get('ew', 0.0)
    Ca_I_4227 = measurements.get('Ca_I_4227', {}).get('ew', 0.0)

    # Hierro: Fe I 4046, 4144, 4260, 4384, 4957
    Fe_I_4046 = measurements.get('Fe_I_4046', {}).get('ew', 0.0)
    Fe_I_4144 = measurements.get('Fe_I_4144', {}).get('ew', 0.0)
    Fe_I_4260 = measurements.get('Fe_I_4260', {}).get('ew', 0.0)
    Fe_I_4383 = measurements.get('Fe_I_4383', {}).get('ew', 0.0)
    Fe_I_4957 = measurements.get('Fe_I_4957', {}).get('ew', 0.0)

    # Cromo: Cr I 4254 (distinguir G vs K)
    Cr_I_4254 = measurements.get('Cr_I_4254', {}).get('ew', 0.0)

    # TiO (tipo M): 4762, 4955, 5167
    TiO_4955 = measurements.get('TiO_4955', {}).get('ew', 0.0)

    # Líneas de confirmación B
    O_II_4591 = measurements.get('O_II_4591', {}).get('ew', 0.0)
    C_II_4267 = measurements.get('C_II_4267', {}).get('ew', 0.0)
    Fe_II_4924 = measurements.get('Fe_II_4924', {}).get('ew', 0.0)
    Ti_II_4468 = measurements.get('Ti_II_4468', {}).get('ew', 0.0)

    # ============================================================
    # INICIALIZAR SALIDAS
    # ============================================================
    lineas_usadas = []
    justificacion = []
    advertencias = []
    confianza = 100

    # Rango espectral
    wave_min = float(wavelengths[0]) if wavelengths is not None and len(wavelengths) > 0 else 3900.0
    wave_max = float(wavelengths[-1]) if wavelengths is not None and len(wavelengths) > 0 else 7000.0
    covers_blue = wave_min <= 3950

    # Promedios
    H_avg = (H_beta + H_gamma + H_delta) / 3.0
    metales_avg = (Fe_I_4046 + Fe_I_4383 + Ca_I_4227) / 3.0

    # ============================================================
    # THRESHOLDS CALIBRADOS
    # ============================================================
    # He II debe ser FUERTE en tipo O (no débil)
    THRESHOLD_He_II = 0.30   # He II mínimo para tipo O
    # He I debe ser claramente visible, no ruido
    THRESHOLD_He_I = 0.15    # He I mínimo para detectar
    # Threshold general
    THRESHOLD = 0.08

    # ============================================================
    # PASO 1: CLASIFICACIÓN GRUESA (aspecto global)
    # ============================================================
    # - Pocas líneas, continuo dominante → temprana (OBA)
    # - Serie Balmer dominante → intermedia (B tardía/A/F)
    # - Muchas líneas metálicas → tardía (GK o KM)

    # He I: requiere al menos 2 líneas con EW significativo
    He_I_lines_detected = sum([
        He_I_4026 > THRESHOLD_He_I,
        He_I_4471 > THRESHOLD_He_I,
        He_I_4922 > THRESHOLD_He_I,
        He_I_5016 > THRESHOLD_He_I
    ])
    He_I_present = He_I_lines_detected >= 2

    # He II: debe ser FUERTE para diagnosticar tipo O (EW > 0.3 Å)
    He_II_strong = (He_II_4686 > THRESHOLD_He_II or
                   He_II_4542 > THRESHOLD_He_II or
                   He_II_4200 > THRESHOLD_He_II)
    # He II débil (puede ser ruido o contaminación)
    He_II_weak = (He_II_4686 > THRESHOLD or
                  He_II_4542 > THRESHOLD or
                  He_II_4200 > THRESHOLD) and not He_II_strong

    # VALIDACIÓN CRUZADA: Si Balmer es MUY fuerte, He débil es falso positivo
    # En estrellas A, H puede tener EW > 5-10 Å, He I debe ser << H
    balmer_muy_fuerte = H_avg > 4.0
    if balmer_muy_fuerte:
        # En A/F, si Balmer domina, cualquier He debe ser mucho menor
        ratio_He_H = max(He_I_4471, He_I_4026) / (H_avg + 0.01)
        if ratio_He_H < 0.15:  # He I < 15% de H → no es He I real
            He_I_present = False
            He_II_strong = False
            He_II_weak = False

    balmer_domina = H_avg > 2.0 and H_avg > metales_avg * 2
    muchos_metales = metales_avg > 0.3 or (Fe_I_4144 > 0.2 and Ca_I_4227 > 0.2)

    # Clasificación gruesa basada en características dominantes
    if He_II_strong:
        # He II fuerte → definitivamente tipo O
        clasificacion_gruesa = 'temprana'
    elif He_I_present and not balmer_muy_fuerte:
        # He I presente y Balmer no domina → O o B
        clasificacion_gruesa = 'temprana'
    elif balmer_domina or balmer_muy_fuerte:
        # Balmer dominante → A o F (o B tardía si hay He I débil)
        clasificacion_gruesa = 'intermedia'
    elif muchos_metales:
        clasificacion_gruesa = 'tardia'
    else:
        # Por defecto, intermedia si hay algo de Balmer
        if H_avg > 1.0:
            clasificacion_gruesa = 'intermedia'
        else:
            clasificacion_gruesa = 'tardia'

    justificacion.append(f"Clasificación gruesa: {clasificacion_gruesa}")

    # ============================================================
    # PASO 2: ESTRELLAS O, B y A (buscar He I)
    # ============================================================
    if clasificacion_gruesa == 'temprana':
        if He_I_present:
            lineas_usadas.extend(['He I 4026', 'He I 4471', 'He I 4922', 'He I 5016'])
            justificacion.append("He I presente → OB")

            # ============================================================
            # ESTRELLAS O: Buscar He II FUERTE (4200, 4542, 4686)
            # ============================================================
            if He_II_strong:
                spectral_type = 'O'
                lineas_usadas.extend(['He II 4200', 'He II 4542', 'He II 4686'])
                justificacion.append(f"He II fuerte (4686={He_II_4686:.2f}, 4542={He_II_4542:.2f}) → tipo O")

                # Subtipo O: comparar He I 4471 vs He II 4542
                He_II_ref = He_II_4542 if He_II_4542 > 0.02 else He_II_4686

                if He_I_4471 < He_II_ref:
                    # O temprana (< O7)
                    ratio = He_I_4471 / (He_II_ref + 0.01)
                    if ratio < 0.5:
                        subtype = 'O5'
                        justificacion.append(f"He I 4471 ({He_I_4471:.2f}) << He II 4542 ({He_II_ref:.2f}) → O temprana")
                    else:
                        subtype = 'O6-O7'
                        justificacion.append(f"He I 4471 < He II 4542 → O6-O7")
                elif He_I_4387 >= He_II_ref:
                    # O tardía (> O9)
                    if Si_III_4553 > 0.05 and abs(Si_III_4553 - He_II_ref) < He_II_ref * 0.5:
                        subtype = 'O9.5'
                        lineas_usadas.append('Si III 4553')
                        justificacion.append("Si III 4553 ≈ He II 4542 → O9.5-B0")
                    else:
                        subtype = 'O9'
                        justificacion.append(f"He I 4387 ({He_I_4387:.2f}) >= He II 4542 → O tardía")
                else:
                    subtype = 'O8'
                    justificacion.append("He I ≈ He II → O7-O9")

            # ============================================================
            # ESTRELLAS B: He I presente, He II ausente
            # ============================================================
            else:
                spectral_type = 'B'
                justificacion.append("He I presente, He II ausente → tipo B")

                # Comparar He I 4471 vs Mg II 4481
                lineas_usadas.extend(['He I 4471', 'Mg II 4481'])

                if Mg_II_4481 < 0.05 or He_I_4471 > Mg_II_4481 * 2:
                    # B TEMPRANAS: He I >> Mg II
                    justificacion.append(f"He I 4471 ({He_I_4471:.2f}) >> Mg II 4481 ({Mg_II_4481:.2f}) → B temprana")
                    lineas_usadas.extend(['Si IV 4089', 'Si III 4553', 'Si II 4128'])

                    # Criterios Si
                    if Si_IV_4089 > 0.05 and Si_III_4553 > 0.05:
                        if abs(Si_IV_4089 - Si_III_4553) < max(Si_IV_4089, Si_III_4553) * 0.4:
                            subtype = 'B0'
                            justificacion.append("Si IV ≈ Si III → B0")
                            # Confirmación: He II 4686 solo en B0
                            if He_II_4686 > 0.02:
                                justificacion.append("He II 4686 confirma B0")
                                confianza = min(confianza, 95)
                        elif Si_III_4553 > Si_IV_4089:
                            subtype = 'B1-B2'
                            justificacion.append("Si III > Si IV → B1-B2")
                        else:
                            subtype = 'B0.5'
                    elif Si_III_4553 > Si_II_4128:
                        if Si_III_4553 >= Si_II_4128 * 1.5:
                            subtype = 'B2'
                            justificacion.append("Si III dominante → B2")
                        else:
                            subtype = 'B3'
                            justificacion.append("Si III ≥ Si II → B3")
                    elif Si_II_4128 > Si_III_4553 * 1.5:
                        subtype = 'B5'
                        justificacion.append("Si III << Si II → B4-B5")
                    else:
                        subtype = 'B3'
                        confianza = min(confianza, 70)
                        advertencias.append("Líneas de Si débiles, subtipo incierto")

                    # Confirmaciones O II, C II
                    if O_II_4591 > 0.05:
                        justificacion.append("O II 4591 confirma B temprana")
                    if C_II_4267 > 0.05:
                        lineas_usadas.append('C II 4267')

                else:
                    # B TARDÍAS: Mg II comparable o > He I
                    ratio_Mg_He = Mg_II_4481 / (He_I_4471 + 0.01)
                    justificacion.append(f"Mg II/He I = {ratio_Mg_He:.2f} → B tardía")

                    if ratio_Mg_He < 1.5:
                        subtype = 'B6-B7'
                        justificacion.append("Mg II ≈ He I → B6-B7")
                    elif ratio_Mg_He < 3.0:
                        subtype = 'B8'
                        justificacion.append("Mg II 2-3× He I → B8")
                    else:
                        subtype = 'B9'
                        justificacion.append("Mg II >> He I → B9-A0")

                    # Confirmación Ti II 4468
                    if Ti_II_4468 > 0.05:
                        lineas_usadas.append('Ti II 4468')
                        if abs(Ti_II_4468 - He_I_4471) < max(Ti_II_4468, He_I_4471) * 0.3:
                            justificacion.append("Ti II 4468 ≈ He I → B9.5")
                            subtype = 'B9.5'

        else:
            # Sin He I → tipo A
            spectral_type = 'A'
            justificacion.append("Sin He I → tipo A")
            confianza = min(confianza, 80)
            advertencias.append("Clasificación A por ausencia de He I")
            subtype = 'A5'

    # ============================================================
    # ESTRELLAS INTERMEDIAS (PASO 3)
    # ============================================================
    elif clasificacion_gruesa == 'intermedia':
        # He I presente → B tardía
        if He_I_present:
            spectral_type = 'B'
            subtype = 'B8-B9'
            lineas_usadas.extend(['He I 4471', 'Mg II 4481'])
            justificacion.append("He I presente en espectro Balmer → B tardía")

        # CASO 1: Cobertura azul disponible (Ca II K y Hε)
        elif covers_blue and H_epsilon > 0.1 and Ca_II_K > 0.05:
            ratio_Ca_H = Ca_II_K / (H_epsilon + 0.01)
            lineas_usadas.extend(['Ca II K 3933', 'H epsilon 3970'])

            if ratio_Ca_H < 0.3:
                spectral_type = 'A'
                subtype = 'A0-A2'
                justificacion.append(f"Ca II K << Hε (ratio={ratio_Ca_H:.2f}) → A temprana")
            elif ratio_Ca_H < 0.6:
                spectral_type = 'A'
                subtype = 'A5'
                justificacion.append(f"Ca II K ≈ ½ Hε (ratio={ratio_Ca_H:.2f}) → A intermedia")
            elif ratio_Ca_H < 1.2:
                spectral_type = 'F'
                subtype = 'F0'
                justificacion.append(f"Ca II K ≈ Hε (ratio={ratio_Ca_H:.2f}) → F0")
            else:
                spectral_type = 'F'
                subtype = 'F5'
                justificacion.append(f"Ca II K > Hε → F tardía")

        # CASO 2: Sin cobertura azul → usar criterios alternativos
        else:
            lineas_usadas.extend(['H delta 4102', 'H gamma 4341', 'H beta 4861'])
            lineas_usadas.extend(['Mg II 4481', 'Fe I 4383', 'Ca I 4227'])

            # CRITERIO ALTERNATIVO para A vs F:
            # - Tipo A: Balmer MUY fuerte (H_avg > 5), metales débiles
            # - Tipo F: Balmer moderado (2 < H_avg < 5), metales visibles

            # Mg II 4481 aparece en A tardía y F (ausente en A temprana)
            has_Mg_II = Mg_II_4481 > 0.10
            # Metales significativos
            has_metals = (Fe_I_4383 > 0.15 or Fe_I_4144 > 0.15 or Ca_I_4227 > 0.15)

            if H_avg > 6.0 and not has_metals:
                # Balmer muy fuerte, metales ausentes → A temprana
                spectral_type = 'A'
                subtype = 'A0-A2'
                justificacion.append(f"Balmer muy fuerte (H_avg={H_avg:.2f}), sin metales → A temprana")

            elif H_avg > 4.0 and (not has_metals or metales_avg < 0.2):
                # Balmer fuerte, metales débiles → A
                spectral_type = 'A'
                if has_Mg_II:
                    subtype = 'A5-A7'
                    justificacion.append(f"Balmer fuerte (H_avg={H_avg:.2f}), Mg II visible → A intermedia-tardía")
                else:
                    subtype = 'A2-A5'
                    justificacion.append(f"Balmer fuerte (H_avg={H_avg:.2f}), sin Mg II → A temprana-intermedia")

            elif H_avg > 2.0 and has_metals:
                # Balmer moderado, metales visibles → F
                spectral_type = 'F'
                ratio_H_metal = H_avg / (metales_avg + 0.01)
                if ratio_H_metal > 3.0:
                    subtype = 'F0-F2'
                    justificacion.append(f"Balmer > metales (ratio={ratio_H_metal:.1f}) → F temprana")
                elif ratio_H_metal > 1.5:
                    subtype = 'F5'
                    justificacion.append(f"Balmer ≈ metales (ratio={ratio_H_metal:.1f}) → F intermedia")
                else:
                    subtype = 'F8'
                    justificacion.append(f"Metales comparables a Balmer → F tardía")

            elif H_avg > 2.0:
                # Balmer moderado sin metales claros → A tardía
                spectral_type = 'A'
                subtype = 'A7-A9'
                justificacion.append(f"Balmer moderado (H_avg={H_avg:.2f}) → A tardía")

            else:
                # Balmer débil → probablemente F o más tardía
                spectral_type = 'F'
                subtype = 'F5-F8'
                justificacion.append("Balmer débil en espectro intermedio → F")

            if not covers_blue:
                advertencias.append("Sin cobertura de Ca II K/Hε, usando criterios alternativos")
                confianza = min(confianza, 75)

    # ============================================================
    # ESTRELLAS TARDÍAS (PASO 4)
    # ============================================================
    else:  # clasificacion_gruesa == 'tardia'
        lineas_usadas.extend(['H delta 4102', 'Fe I 4046', 'Fe I 4144', 'Ca I 4227'])

        # Balmer aún visible → F
        if H_avg > 2.0 and H_avg > metales_avg:
            # PASO 11: ESTRELLAS F
            spectral_type = 'F'
            lineas_usadas.extend(['H gamma 4341', 'Fe I 4384'])

            ratio_H_metal = H_avg / (metales_avg + 0.01)
            if ratio_H_metal > 4:
                subtype = 'F0-F2'
                justificacion.append(f"HI >> metales (ratio={ratio_H_metal:.1f}) → F temprana")
            elif ratio_H_metal > 1.5:
                subtype = 'F5'
                justificacion.append(f"HI > metales → F intermedia")
            else:
                subtype = 'F8-G0'
                justificacion.append(f"HI ≈ metales (ratio={ratio_H_metal:.1f}) → F9-G0")

        # Ca I muy intensa + bandas TiO → K tardía o M
        elif Ca_I_4227 > 0.5 and (Ca_I_4227 > H_avg or H_avg < 1.0):
            # PASO 13: K tardías y M - Evaluar TiO
            if wavelengths is not None and flux_normalized is not None:
                tio_strength, n_bands, tio_depth = evaluate_tio_bands(
                    wavelengths, flux_normalized, measurements)
                asymm, flux_l, flux_r, has_tio = detect_FeI_4957_asymmetry(
                    wavelengths, flux_normalized)

                lineas_usadas.extend(['TiO 4762', 'TiO 4955', 'TiO 5167', 'Fe I 4957'])

                if tio_strength in ['moderado', 'fuerte', 'muy_fuerte'] or has_tio:
                    spectral_type = 'M'
                    justificacion.append(f"Bandas TiO detectadas ({tio_strength})")

                    # Criterio "dientes de sierra": TiO muy_fuerte en las 3 bandas Y
                    # Fe I 4957 ha desaparecido por completo (profundidad < 5%)
                    fe_4957_depth = measurements.get('Fe_I_4957', {}).get('depth', 0.0)
                    TiO_4762_depth = measurements.get('TiO_4762', {}).get('depth', 0.0)
                    TiO_4955_depth = measurements.get('TiO_4955', {}).get('depth', 0.0)
                    TiO_5167_depth = measurements.get('TiO_5167', {}).get('depth', 0.0)
                    todas_bandas_tio = (TiO_4762_depth > 0.30 and
                                        TiO_4955_depth > 0.30 and
                                        TiO_5167_depth > 0.30)

                    if tio_strength == 'muy_fuerte' and todas_bandas_tio and fe_4957_depth < 0.05:
                        subtype = 'M6-M7'
                        justificacion.append(
                            f"Espectro dominado por TiO en 3 bandas (profundidades: "
                            f"4762={TiO_4762_depth:.2f}, 4955={TiO_4955_depth:.2f}, "
                            f"5167={TiO_5167_depth:.2f}), Fe I 4957 desaparece → "
                            f"'dientes de sierra' M6-M7"
                        )
                    elif tio_strength == 'muy_fuerte' or asymm < 0.2:
                        subtype = 'M5-M6'
                        justificacion.append("TiO domina, Fe I 4957 desaparece → M tardía")
                    elif asymm < 0.5:
                        subtype = 'M2-M4'
                        justificacion.append(f"Fe I 4957 asimétrica (asymm={asymm:.2f}) → M0-M4")
                    else:
                        subtype = 'M0-M2'
                        justificacion.append("TiO moderado → M temprana")

                elif tio_strength == 'debil':
                    spectral_type = 'K'
                    subtype = 'K5'
                    justificacion.append("TiO débil, Fe I 4957 simétrica → K5")
                else:
                    spectral_type = 'K'
                    subtype = 'K3-K5'
                    justificacion.append("Sin TiO, Ca I fuerte → K intermedia")
            else:
                # Sin datos de espectro completo
                if TiO_4955 > 0.3:
                    spectral_type = 'M'
                    subtype = 'M0-M2'
                    lineas_usadas.append('TiO 4955')
                else:
                    spectral_type = 'K'
                    subtype = 'K5'
                advertencias.append("Sin datos de espectro para evaluar TiO")
                confianza = min(confianza, 60)

        # PASO 12: G o K temprana
        else:
            lineas_usadas.append('Cr I 4254')

            # Comparar HI 4102 con metales
            H_vs_Ca = H_delta / (Ca_I_4227 + 0.01)
            H_vs_Fe4046 = H_delta / (Fe_I_4046 + 0.01)
            H_vs_Fe4144 = H_delta / (Fe_I_4144 + 0.01)
            Cr_vs_Fe = Cr_I_4254 / (Fe_I_4260 + 0.01)

            if H_delta > Ca_I_4227 and H_delta > Fe_I_4046 and H_delta > Fe_I_4144:
                spectral_type = 'G'
                subtype = 'G0'
                justificacion.append("HI > metales → G0")
            elif abs(H_delta - Ca_I_4227) < max(H_delta, Ca_I_4227) * 0.3:
                spectral_type = 'G'
                # Refinar subtipo G con Cr I 4254 / Fe I 4260
                # Cr < Fe → G más temprana; Cr > Fe → G tardía tendiendo a K
                if Cr_I_4254 > 0.05 and Fe_I_4260 > 0.05:
                    if Cr_vs_Fe < 0.90:
                        subtype = 'G2-G5'
                        justificacion.append(
                            f"HI ≈ Ca I, Cr I < Fe I (ratio={Cr_vs_Fe:.2f}) → G2-G5")
                    elif Cr_vs_Fe < 1.10:
                        subtype = 'G5'
                        justificacion.append(
                            f"HI ≈ Ca I, Cr I ≈ Fe I (ratio={Cr_vs_Fe:.2f}) → G5")
                    else:
                        subtype = 'G5-G8'
                        justificacion.append(
                            f"HI ≈ Ca I, Cr I > Fe I (ratio={Cr_vs_Fe:.2f}) → G5-G8")
                    if 'Cr I 4254' not in lineas_usadas:
                        lineas_usadas.append('Cr I 4254')
                    lineas_usadas.append('Fe I 4260')
                else:
                    subtype = 'G5'
                    justificacion.append("HI ≈ metales → G5")
            elif H_delta < Ca_I_4227 and H_delta < Fe_I_4046:
                # Usar Cr/Fe para distinguir G tardía vs K
                if Cr_vs_Fe < 1.0:
                    spectral_type = 'G'
                    subtype = 'G8'
                    justificacion.append(f"Cr I < Fe I (ratio={Cr_vs_Fe:.2f}) → G tardía")
                else:
                    spectral_type = 'K'
                    subtype = 'K0'
                    justificacion.append(f"Cr I > Fe I (ratio={Cr_vs_Fe:.2f}) → K")

                # Ca I >> Fe I → K7+
                if Ca_I_4227 > Fe_I_4046 * 2:
                    subtype = 'K5-K7'
                    justificacion.append("Ca I 4226 >> Fe I 4046 → K tardía")
            else:
                spectral_type = 'G'
                subtype = 'G5'
                confianza = min(confianza, 70)
                advertencias.append("Clasificación G/K ambigua")

    # ============================================================
    # CLASE DE LUMINOSIDAD MK
    # ============================================================
    # Se calcula siempre a partir de las mediciones ya disponibles.
    # Indicadores usados según tipo:
    #   O/B : He II en emisión/absorción, O II, N III
    #   A/F : Sr II 4077 / Fe I 4071, Balmer width
    #   G/K : Ca I 4227 / Fe I 4383, Ba II 4554 / Fe I 4260
    #   M   : TiO / CaH ratio
    luminosity_class = estimate_luminosity_class(measurements, spectral_type)
    mk_full = combine_spectral_and_luminosity(subtype if subtype else spectral_type,
                                              luminosity_class)

    # Nombre descriptivo de la clase de luminosidad
    _lum_names = {
        'Ia':  'Supergigante muy luminosa',
        'Ib':  'Supergigante',
        'II':  'Gigante brillante',
        'III': 'Gigante',
        'IV':  'Subgigante',
        'V':   'Secuencia principal (enana)',
    }
    lum_name = _lum_names.get(luminosity_class, luminosity_class)

    # ============================================================
    # CONSTRUIR DIAGNÓSTICOS
    # ============================================================
    diagnostics = {
        # Salidas requeridas
        'lineas_usadas': lineas_usadas,
        'justificacion': ' | '.join(justificacion),
        'advertencias': advertencias,
        'confianza': confianza,

        # Clasificación espectral
        'clasificacion_gruesa': clasificacion_gruesa,
        'spectral_type': spectral_type,
        'subtype': subtype,

        # ── CLASE DE LUMINOSIDAD MK ────────────────────────────────
        'luminosity_class': luminosity_class,   # 'Ia', 'Ib', 'II', 'III', 'IV', 'V'
        'mk_full': mk_full,                      # ej. 'G2V', 'B1Ib', 'K3III'
        'lum_name': lum_name,                    # nombre descriptivo en español

        # Mediciones clave
        'has_He_II': He_II_strong,
        'has_He_II_weak': He_II_weak,
        'has_He_I': He_I_present,
        'He_I_lines_detected': He_I_lines_detected,
        'balmer_muy_fuerte': balmer_muy_fuerte,
        'He_II_4686': He_II_4686,
        'He_II_4542': He_II_4542,
        'He_I_4471': He_I_4471,
        'He_I_4387': He_I_4387,
        'Mg_II_4481': Mg_II_4481,
        'Si_IV_4089': Si_IV_4089,
        'Si_III_4553': Si_III_4553,
        'Si_II_4128': Si_II_4128,
        'H_avg': H_avg,
        'H_delta': H_delta,
        'H_gamma': H_gamma,
        'H_epsilon': H_epsilon,
        'Ca_II_K': Ca_II_K,
        'Ca_I_4227': Ca_I_4227,
        'Fe_I_4046': Fe_I_4046,
        'Fe_I_4144': Fe_I_4144,
        'Fe_I_4260': Fe_I_4260,
        'Fe_I_4957': Fe_I_4957,
        'Cr_I_4254': Cr_I_4254,
        'TiO_4955': TiO_4955,
        'wave_min': wave_min,
        'wave_max': wave_max,
        'covers_blue': covers_blue,
    }

    return spectral_type, subtype, diagnostics


# ============================================================================
# CORRECCIÓN 5: Clasificación basada en anchos equivalentes (intensidades)
# ============================================================================

def classify_star_corrected(measurements, wavelengths=None, flux_normalized=None, use_decision_tree=True):
    """
    Clasifica una estrella siguiendo el esquema de clasificación espectral estándar.

    NUEVO: Por defecto usa el árbol de decisión de 13 pasos (use_decision_tree=True)
    que sigue el esquema clásico de clasificación espectral.

    Parameters
    ----------
    measurements : dict
        Diccionario de mediciones de líneas (de measure_diagnostic_lines)
    wavelengths : array, optional
        Longitudes de onda (para detección mejorada de tipo M)
    flux_normalized : array, optional
        Flujo normalizado (para detección mejorada de tipo M)
    use_decision_tree : bool, default True
        Si True, usa el nuevo árbol de decisión de 13 pasos
        Si False, usa la lógica antigua (más compleja pero menos estructurada)

    Returns
    -------
    spectral_type : str
        Tipo espectral estimado
    subtype : str
        Subtipo numérico estimado
    diagnostics : dict
        Información diagnóstica usada en la clasificación
    """
    # ============================================================
    # NUEVO: Usar árbol de decisión por defecto
    # ============================================================
    if use_decision_tree:
        return classify_star_decision_tree(measurements, wavelengths, flux_normalized)

    # ============================================================
    # LÓGICA ANTIGUA (solo si use_decision_tree=False)
    # ============================================================
    # PASO 1: Extraer todas las mediciones necesarias
    # ============================================================

    # He II (diagnóstico de tipo O)
    He_II_4686 = measurements.get('He_II_4686', {}).get('ew', 0.0)
    He_II_4542 = measurements.get('He_II_4542', {}).get('ew', 0.0)
    He_II_4200 = measurements.get('He_II_4200', {}).get('ew', 0.0)

    # He I (diagnóstico de OB)
    He_I_4471 = measurements.get('He_I_4471', {}).get('ew', 0.0)
    He_I_4387 = measurements.get('He_I_4387', {}).get('ew', 0.0)
    He_I_4026 = measurements.get('He_I_4026', {}).get('ew', 0.0)
    He_I_4922 = measurements.get('He_I_4922', {}).get('ew', 0.0)
    He_I_5016 = measurements.get('He_I_5016', {}).get('ew', 0.0)

    # Hidrógeno (Balmer)
    H_epsilon = measurements.get('H_epsilon', {}).get('ew', 0.0)  # 3970 - para tipo A
    H_delta = measurements.get('H_delta', {}).get('ew', 0.0)      # 4101 - para tipo G-K
    H_gamma = measurements.get('H_gamma', {}).get('ew', 0.0)      # 4340 - para tipo F
    H_beta = measurements.get('H_beta', {}).get('ew', 0.0)        # 4861
    H_avg = (H_beta + H_gamma + H_delta) / 3.0

    # Nitrógeno (O tempranas)
    N_V_4604 = measurements.get('N_V_4604', {}).get('ew', 0.0)
    N_V_4620 = measurements.get('N_V_4620', {}).get('ew', 0.0)
    N_IV_4058 = measurements.get('N_IV_4058', {}).get('ew', 0.0)
    N_III_4634 = measurements.get('N_III_4634', {}).get('ew', 0.0)
    N_III_4641 = measurements.get('N_III_4641', {}).get('ew', 0.0)

    # Carbono (O tempranas)
    C_IV_5801 = measurements.get('C_IV_5801', {}).get('ew', 0.0)
    C_IV_5812 = measurements.get('C_IV_5812', {}).get('ew', 0.0)
    C_IV_4658 = measurements.get('C_IV_4658', {}).get('ew', 0.0)

    # Oxígeno ionizado (O tempranas)
    O_V_5114 = measurements.get('O_V_5114', {}).get('ew', 0.0)
    O_III_5592 = measurements.get('O_III_5592', {}).get('ew', 0.0)

    # Silicio (B tempranas)
    Si_III_4553 = measurements.get('Si_III_4553', {}).get('ew', 0.0)
    Si_IV_4089 = measurements.get('Si_IV_4089', {}).get('ew', 0.0)
    Si_II_4128 = measurements.get('Si_II_4128', {}).get('ew', 0.0)

    # Magnesio (B tardías)
    Mg_II_4481 = measurements.get('Mg_II_4481', {}).get('ew', 0.0)

    # Oxígeno (indicador de luminosidad en B)
    O_II_4591 = measurements.get('O_II_4591', {}).get('ew', 0.0)
    O_II_4596 = measurements.get('O_II_4596', {}).get('ew', 0.0)

    # Calcio (A y tardías)
    Ca_II_K = measurements.get('Ca_II_K', {}).get('ew', 0.0)       # 3933
    Ca_II_H = measurements.get('Ca_II_H', {}).get('ew', 0.0)       # 3968 (≈ H epsilon)
    Ca_I_4227 = measurements.get('Ca_I_4227', {}).get('ew', 0.0)   # 4227 - para F-G-K

    # Hierro (tipos F-G-K-M)
    Fe_I_4046 = measurements.get('Fe_I_4046', {}).get('ew', 0.0)
    Fe_I_4144 = measurements.get('Fe_I_4144', {}).get('ew', 0.0)
    Fe_I_4260 = measurements.get('Fe_I_4260', {}).get('ew', 0.0)
    Fe_I_4325 = measurements.get('Fe_I_4325', {}).get('ew', 0.0)
    Fe_I_4383 = measurements.get('Fe_I_4383', {}).get('ew', 0.0)
    Fe_I_4957 = measurements.get('Fe_I_4957', {}).get('ew', 0.0)

    # Cromo (distinguir G vs K)
    Cr_I_4254 = measurements.get('Cr_I_4254', {}).get('ew', 0.0)

    # Titanio/TiO (tipos K-M)
    TiO_4955 = measurements.get('TiO_4955', {}).get('ew', 0.0)

    # ============================================================
    # PASO 2: Detectar presencia de He I y He II
    # ============================================================

    # Umbral de detección general: 0.05 Å (ruido típico)
    DETECTION_THRESHOLD = 0.05

    # Umbral para He II: CRÍTICO para detectar O8-O9
    # O tempranas (O3-O7): He II fuerte (0.3-1.5 Å)
    # O tardías (O8-O9): He II muy débil (0.01-0.12 Å) pero PRESENTE
    # BAJADO A 0.01 Å para evitar confusión O8 → A0 (casi sin umbral)
    HE_II_THRESHOLD = 0.01  # Reducido de 0.15 → 0.08 → 0.05 → 0.03 → 0.01 (MÍNIMO para O8-O9)

    # Contar líneas de He II por encima del umbral
    He_II_lines_detected = sum([
        He_II_4686 > HE_II_THRESHOLD,
        He_II_4542 > HE_II_THRESHOLD,
        He_II_4200 > HE_II_THRESHOLD
    ])

    # CORRECCIÓN CRÍTICA: He II tiene PRIORIDAD ABSOLUTA
    # Si He II está presente (incluso débil), ES tipo O
    # Los filtros solo se usan para casos ambiguos con características tardías

    # Calcular He II promedio para determinar confianza
    He_II_avg = (He_II_4686 + He_II_4542 + He_II_4200) / 3.0

    # Calcular He I promedio para comparar (O tardías tienen He I > He II)
    He_I_avg = (He_I_4471 + He_I_4026 + He_I_4922 + He_I_5016) / 4.0

    # Verificar características de tipos tardíos (solo como advertencia)
    has_late_type_features = (
        Ca_II_K > 2.0 or           # Ca II K fuerte → tipo A-K
        Ca_I_4227 > 0.5 or         # Ca I presente → tipo F-K
        Fe_I_4046 > 0.3 or         # Fe I presente → tipo F-M
        Fe_I_4144 > 0.3            # Fe I presente → tipo G-M
    )

    # LÓGICA MEJORADA: Verificar características tardías SIEMPRE
    # Si hay evidencia fuerte de tipo tardío (F-G-K), rechazar He II incluso si aparece
    # Esto previene falsos positivos por ruido o normalización defectuosa

    # Características MUY FUERTES de tipos tardíos (incompatibles con tipo O/B)
    # IMPORTANTE: No depender solo de Ca II K (puede estar fuera del rango espectral)
    has_very_strong_late_features = (
        Ca_II_K > 3.0 or           # Ca II K muy fuerte → tipo F-K
        (Ca_I_4227 > 0.8 and Fe_I_4046 > 0.5) or  # Ca I + Fe I fuerte → tipo F-K
        Fe_I_4144 > 0.8 or         # Fe I 4144 muy fuerte → tipo G-K
        (H_avg < 3.0 and (Fe_I_4046 > 0.3 or Ca_I_4227 > 0.5))  # H débil + metales → tipo F-K
    )

    if has_very_strong_late_features:
        # Características tardías MUY fuertes → NO puede ser tipo O
        # Incluso si detectó He II, son falsos positivos
        has_He_II = False
    elif He_II_lines_detected >= 2:
        # 2+ líneas de He II detectadas → PROBABLEMENTE tipo O
        # Permitir características tardías moderadas (pueden ser ruido)
        has_He_II = True  # Confianza alta con 2+ líneas
    elif He_II_lines_detected == 1 and He_II_avg > 0.10:
        # 1 línea pero significativa (> 0.10 Å) → probablemente tipo O
        # Reducido de 0.3 a 0.10 para incluir O8-O9
        has_He_II = not has_late_type_features
    elif He_II_lines_detected >= 1 and He_II_avg > HE_II_THRESHOLD:
        # ✅ CRÍTICO para O8-O9: Cualquier He II detectable (> 0.03 Å) → Tipo O
        # RELAJADO: Ignorar características tardías si He I también fuerte
        # O8-O9 tiene He I fuerte, puede parecer tipo B/A pero He II confirma tipo O
        has_strong_He_I = He_I_avg > 0.3  # He I fuerte indica OB, no A
        if has_strong_He_I:
            has_He_II = True  # He II + He I fuerte → definitivamente tipo O
        else:
            has_He_II = not has_late_type_features
    else:
        # He II ausente → NO es tipo O
        has_He_II = False

    # Presencia de He I (al menos 1 línea detectada)
    # He I en estrellas B tempranas es fuerte (0.3-1.5 Å)
    # He I en O8-O9 puede ser moderado (0.10-0.30 Å)
    # CRÍTICO: Usar umbral moderado (0.10 Å) para detectar O8-O9 y B
    HE_I_THRESHOLD = 0.10  # Reducido de 0.15 para detectar O tardías

    He_I_lines_detected = sum([
        He_I_4471 > HE_I_THRESHOLD,
        He_I_4026 > HE_I_THRESHOLD,
        He_I_4922 > HE_I_THRESHOLD,
        He_I_5016 > HE_I_THRESHOLD
    ])

    # VERIFICACIÓN CRUZADA CON BALMER (para evitar confusión B tardía ↔ A0)
    # Si H es extremadamente fuerte (H avg > 4.0 Å), es probablemente tipo A, no B
    # En tipo A0, H beta puede alcanzar 5-6 Å (máximo), mientras que en B es < 3 Å
    has_strong_balmer = (H_avg > 4.0 or H_beta > 5.0)

    # LÓGICA MEJORADA: Verificar características tardías SIEMPRE para He I también
    # Prevenir falsos positivos de He I en estrellas F-G-K

    if has_very_strong_late_features:
        # Características tardías MUY fuertes → NO puede ser tipo B
        # Incluso si detectó He I, son falsos positivos
        has_He_I = False
    elif He_I_lines_detected >= 2 and not has_strong_balmer:
        # 2+ líneas de He I, Balmer no domina, sin características tardías fuertes
        # Verificar características tardías moderadas
        has_He_I = not has_late_type_features
    elif He_I_lines_detected == 1:
        # 1 línea He I → aplicar múltiples filtros
        # Rechazar si: (1) características tardías O (2) Balmer muy fuerte
        has_late_features = (Ca_II_K > 3.0 and Fe_I_4144 > 0.5)
        has_He_I = not (has_late_features or has_strong_balmer)
    else:
        has_He_I = False

    # ============================================================
    # PASO 2.5: LÓGICA ESPECIAL PARA O TARDÍAS / B TEMPRANAS (O8.5-B0)
    # ============================================================
    # PROBLEMA: O8.5-B0 tienen He II tan débil que puede no detectarse
    # SOLUCIÓN: Si He I presente Y He II tiene trazas → forzar tipo O tardío

    # Detectar He I presente (más permisivo que has_He_I)
    He_I_any = (He_I_4471 > 0.05 or He_I_4026 > 0.05 or
                He_I_4922 > 0.05 or He_I_5016 > 0.05)

    # Detectar trazas de He II (más permisivo que has_He_II)
    He_II_traces = (He_II_4686 > 0.02 or He_II_4542 > 0.02 or He_II_4200 > 0.02)

    # Si He I presente Y He II trazas Y NO características tardías fuertes
    # → Forzar detección como tipo O tardío (O8.5-O9.5)
    if He_I_any and He_II_traces and not has_very_strong_late_features:
        has_He_II = True  # ✅ Forzar tipo O aunque He II sea muy débil
        # No modificar has_He_I - se usa para subtipos

    # ============================================================
    # PASO 3: CLASIFICACIÓN PRINCIPAL
    # ============================================================

    # covers_blue: determina si el espectro cubre Ca II K para criterio canónico A/F
    wave_min_leg = float(wavelengths[0]) if wavelengths is not None and len(wavelengths) > 0 else 9999.0
    covers_blue_leg = wave_min_leg <= 3950

    diagnostics = {
        # ── Campos requeridos por la interfaz web (compatibilidad con árbol de decisión) ──
        'lineas_usadas': [],        # Se llenará al final del clasificador legacy
        'justificacion': '',        # Se llenará al final del clasificador legacy
        'advertencias': [],         # Se llenará al final del clasificador legacy
        'covers_blue': covers_blue_leg,
        'clasificacion_gruesa': '',  # Se llenará al final

        # Helio (tipos O-B)
        'has_He_II': has_He_II,
        'has_He_I': has_He_I,
        'He_II_4686': He_II_4686,
        'He_II_4542': He_II_4542,
        'He_I_4471': He_I_4471,
        'He_I_4387': He_I_4387,
        # Promedios para compatibilidad con scripts
        'He_II': (He_II_4686 + He_II_4542 + He_II_4200) / 3.0,
        'He_I': (He_I_4471 + He_I_4026 + He_I_4922 + He_I_5016) / 4.0,

        # Hidrógeno (todas las series)
        'H_avg': H_avg,
        'H_epsilon': H_epsilon,
        'H_delta': H_delta,
        'H_gamma': H_gamma,
        'H_beta': H_beta,

        # Calcio (tipos A-K)
        'Ca_II_K': Ca_II_K,
        'Ca_II_H': Ca_II_H,
        'Ca_I_4227': Ca_I_4227,

        # Hierro (tipos F-M)
        'Fe_I_4046': Fe_I_4046,
        'Fe_I_4144': Fe_I_4144,
        'Fe_I_4260': Fe_I_4260,
        'Fe_I_4383': Fe_I_4383,
        'Fe_I_4957': Fe_I_4957,

        # Otros metales (tipos B-K)
        'Mg_II_4481': Mg_II_4481,
        'Cr_I_4254': Cr_I_4254,

        # Oxígeno (indicador de luminosidad en B)
        'O_II_4591': O_II_4591,
        'O_II_4596': O_II_4596,
        'O_II_avg': (O_II_4591 + O_II_4596) / 2.0,

        # Bandas moleculares (tipos M)
        'TiO_4955': TiO_4955,

        # Ratios diagnósticos clave
        'ratio_Ca_H_epsilon': Ca_II_K / (H_epsilon + 0.01) if H_epsilon > 0 else 0,
        'ratio_H_gamma_Ca_I': H_gamma / (Ca_I_4227 + 0.01) if Ca_I_4227 > 0 else 0,
        'ratio_H_delta_Ca_I': H_delta / (Ca_I_4227 + 0.01) if Ca_I_4227 > 0 else 0,
        'ratio_Cr_Fe': Cr_I_4254 / (Fe_I_4260 + 0.01) if Fe_I_4260 > 0 else 0,
        'ratio_TiO_Fe': TiO_4955 / (Fe_I_4957 + 0.01) if Fe_I_4957 > 0 else 0,

        # Ratios para clasificación de luminosidad en B
        'ratio_Mg_He_I': Mg_II_4481 / (He_I_4471 + 0.01),
        'ratio_Si_II_III': Si_II_4128 / (Si_III_4553 + 0.01),
        'ratio_Si_III_II': Si_III_4553 / (Si_II_4128 + 0.01),  # Para subtipos B
        'ratio_Si_IV_III': Si_IV_4089 / (Si_III_4553 + 0.01),  # Para B0-B1
    }

    # ------------------------------------------------------------
    # TIPO O: Si se ven líneas de He II
    # ------------------------------------------------------------
    if has_He_II:
        spectral_type = 'O'

        # Comparar He I 4471 vs He II 4542 para subtipos
        # (Usar He II 4686 si 4542 no está disponible)
        He_II_ref = He_II_4542 if He_II_4542 > DETECTION_THRESHOLD else He_II_4686

        # Calcular ratio He I / He II para clasificación precisa
        ratio_He_I_He_II = He_I_4471 / He_II_ref if He_II_ref > 0.01 else 0

        # O tempranas (O3-O6): He I << He II
        # CORREGIDO: Extender rango hasta He I < 75% He II para incluir O6pe
        if ratio_He_I_He_II < 0.75:
            # ============================================================
            # CLASIFICACIÓN MEJORADA CON CRITERIOS ADICIONALES
            # ============================================================
            # Teoría espectroscópica (Walborn 1971, Gray & Corbally 2009):
            # - O3: N V muy fuerte (> 0.5 Å), He II/He I >> 1
            # - O4: N V fuerte (0.20-0.5 Å), He II λ4541 / He I λ4471 ≫ 1 (criterio clásico)
            # - O5: N V moderado (0.11-0.20 Å), He I/He II < 0.65
            # - O6: N V débil (< 0.11 Å) O He I/He II > 0.55 (muy variable, incluye O6e, O6pe)

            # Calcular criterios adicionales para O tempranas (T > 43-45 kK)
            N_V_avg = (N_V_4604 + N_V_4620) / 2.0  # Promedio de doblete N V
            N_III_avg = (N_III_4634 + N_III_4641) / 2.0  # N III (fuerte en O6-O7, débil en O3-O5)
            C_IV_avg = (C_IV_5801 + C_IV_5812) / 2.0  # C IV fuerte en O tempranas

            # Ratio He II / He I (criterio clásico para O4)
            ratio_He_II_He_I = He_II_ref / He_I_4471 if He_I_4471 > 0.01 else 100.0

            # Indicadores de O muy tempranas (O3-O4)
            has_O_V = O_V_5114 > 0.05  # O V presente → T > 43 kK
            has_strong_C_IV = C_IV_avg > 0.15 or C_IV_4658 > 0.10  # C IV fuerte
            has_weak_N_III = N_III_avg < 0.10  # N III débil (fuerte en O6-O7)

            # ============================================================
            # CLASIFICACIÓN CON CRITERIOS MÚLTIPLES
            # ============================================================

            if N_V_4604 > 0.5 or (N_V_avg > 0.4 and has_O_V):
                # O3: N V muy fuerte + O V presente (T ≥ 45 kK)
                subtype = 'O3'

            elif N_V_4604 > 0.15 or N_V_avg > 0.12 or (ratio_He_II_He_I > 5.0 and N_V_avg > 0.08 and has_weak_N_III):
                # O4: CRITERIOS MEJORADOS Y RELAJADOS
                # - Criterio primario: N V > 0.15 Å (bajado de 0.20)
                # - Criterio secundario: N V promedio > 0.12 Å
                # - Criterio alternativo: He II/He I > 5 (bajado de 10) + N V > 0.08 + N III débil
                # JUSTIFICACIÓN: En espectros reales, N V puede ser moderado pero ratio He II/He I muy alto
                subtype = 'O4'

            elif ratio_He_I_He_II < 0.70:
                # O5: CRITERIO PRINCIPAL SIMPLIFICADO
                # He I/He II < 0.70 → O5 (He I débil, característico de O5)
                # JUSTIFICACIÓN: Ratio He I/He II es más robusto que N V
                # O5e puede tener N V muy variable, pero He I/He II es consistente
                subtype = 'O5'

            elif ratio_He_I_He_II < 0.85:
                # O6: He I/He II 0.70-0.85
                # Incluye O6, O6e, O6pe con rangos variables
                # O6pe puede llegar hasta He I/He II ~ 0.80-0.85
                # N III empieza a ser visible en O6-O7
                subtype = 'O6'

        # O7: He I ≈ He II
        # CORREGIDO: Rango ajustado para no confundir con O6pe
        # O6pe puede tener He I/He II hasta ~0.85
        elif 0.85 <= ratio_He_I_He_II <= 1.3:
            subtype = 'O7'

        # O tardías (O8-B0): He I > He II
        else:
            # Comparar He I 4387 vs He II 4542
            if He_I_4387 < He_II_ref:
                subtype = 'O8'
            elif abs(He_I_4387 - He_II_ref) < 0.2:
                subtype = 'O9'
            elif Si_III_4553 > DETECTION_THRESHOLD:
                if He_II_ref > Si_III_4553:
                    subtype = 'O9.5'
                else:
                    subtype = 'O9.7-B0'
            else:
                subtype = 'O9'

        # ============================================================
        # DETERMINAR CLASE DE LUMINOSIDAD para tipo O
        # ============================================================
        # Criterios principales:
        # - Si IV: Más fuerte en supergigantes (Ia/Ib)
        # - O II: Más fuerte en gigantes y supergigantes
        # - NUEVO: He II en emisión → Supergigante (viento estelar)
        # - N IV en emisión → Supergigante
        # - V (enanas): Si IV débil, O II débil, He II en absorción profunda
        # - III (gigantes): Si IV moderado, O II moderado
        # - Ib-Ia (supergigantes): Si IV fuerte, O II fuerte, He II puede estar en emisión

        # Calcular indicadores
        O_II_avg = (O_II_4591 + O_II_4596) / 2.0
        has_Si_IV = Si_IV_4089 > 0.10

        # NUEVO: Detectar emisión en He II (indicador de viento estelar en supergigantes)
        # Emisión se detecta por profundidad negativa (flux > continuo)
        # O por EW negativo
        He_II_4686_depth = measurements.get('He_II_4686', {}).get('depth', 0.0)
        He_II_4542_depth = measurements.get('He_II_4542', {}).get('depth', 0.0)
        He_II_4686_ew = measurements.get('He_II_4686', {}).get('ew', 0.0)
        He_II_4542_ew = measurements.get('He_II_4542', {}).get('ew', 0.0)

        # Emisión: depth negativo O EW negativo (relajado a -0.01)
        He_II_in_emission = (He_II_4686_depth < -0.01 or He_II_4542_depth < -0.01 or
                            He_II_4686_ew < -0.05 or He_II_4542_ew < -0.05)

        # Detectar N IV en emisión (frecuente en O tempranas supergigantes)
        N_IV_depth = measurements.get('N_IV_4058', {}).get('depth', 0.0)
        N_IV_ew = measurements.get('N_IV_4058', {}).get('ew', 0.0)
        N_IV_in_emission = N_IV_depth < -0.01 or N_IV_ew < -0.05

        # Clasificar luminosidad (umbrales corregidos según datos empíricos)
        # Basado en Sota et al. (2011, 2014) - GOSSS survey

        if He_II_in_emission or N_IV_in_emission:
            # CRITERIO PRIORITARIO: Emisión en He II o N IV → Supergigante (viento estelar)
            if O_II_avg > 0.60 or Si_IV_4089 > 0.40:
                luminosity_class = 'Ia'  # Supergigante muy luminosa
            elif O_II_avg > 0.40 or Si_IV_4089 > 0.25:
                luminosity_class = 'Ib'  # Supergigante brillante
            else:
                luminosity_class = 'Ib-II'  # Supergigante/Gigante brillante

        elif O_II_avg > 0.70 and Si_IV_4089 > 0.50:
            # O II muy fuerte Y Si IV muy fuerte → Supergigante luminosa
            luminosity_class = 'Ia'
        elif O_II_avg > 0.50 and Si_IV_4089 > 0.35:
            # O II fuerte Y Si IV fuerte → Supergigante brillante
            luminosity_class = 'Ib'
        elif (O_II_avg > 0.40 and Si_IV_4089 > 0.25) or O_II_avg > 0.60:
            # O II moderado-fuerte Y Si IV moderado-fuerte → Gigante brillante
            luminosity_class = 'II-III'
        elif O_II_avg > 0.25 or (Si_IV_4089 > 0.15 and O_II_avg > 0.20):
            # O II moderado O (Si IV moderado Y O II presente) → Gigante
            luminosity_class = 'III'
        elif O_II_avg > 0.15 and Si_IV_4089 > 0.10:
            # O II débil-moderado Y Si IV débil-moderado → Subgigante
            luminosity_class = 'IV'
        else:
            # O II débil Y Si IV débil/ausente Y He II en absorción → Secuencia principal
            luminosity_class = 'V'

        # Añadir sufijo "e" si hay emisión en He II o Balmer
        # Emisión indica viento estelar, común en O tempranas y supergigantes
        emission_suffix = ''
        if He_II_in_emission or N_IV_in_emission:
            emission_suffix = 'e'

        # También verificar emisión en Balmer (menos común pero posible)
        H_beta_depth = measurements.get('H_beta', {}).get('depth', 0.0)
        if H_beta_depth < -0.02:  # Hβ en emisión
            emission_suffix = 'e'

        # Combinar subtipo + emisión + luminosidad
        subtype = subtype + emission_suffix + luminosity_class

    # ------------------------------------------------------------
    # TIPO B: He I presente, He II ausente
    # ------------------------------------------------------------
    elif has_He_I:
        spectral_type = 'B'

        # Comparar He I 4471 vs Mg II 4481
        if Mg_II_4481 < DETECTION_THRESHOLD or He_I_4471 > Mg_II_4481 * 2:
            # B TEMPRANAS: He I >> Mg II
            # Subtipos según Si III, Si IV, Si II (usando RATIOS)

            # Calcular ratios diagnósticos
            ratio_Si_III_II = Si_III_4553 / (Si_II_4128 + 0.01)
            ratio_Si_IV_III = Si_IV_4089 / (Si_III_4553 + 0.01)

            # B0: He II presente y Si IV fuerte
            if He_II_4542 > DETECTION_THRESHOLD and He_II_4542 >= Si_III_4553:
                subtype = 'B0'

            # B0.5: Si IV domina sobre Si III
            elif ratio_Si_IV_III > 1.0:
                subtype = 'B0.5'

            # ============================================================
            # AJUSTE CRÍTICO: B1 vs B2 vs B3
            # ============================================================
            # B1: Si IV presente (> 0.1 Å) Y Si III >= Si II
            # En B1 real, Si IV aún es visible aunque débil
            elif Si_IV_4089 > 0.1 and Si_III_4553 >= Si_II_4128:
                subtype = 'B1'

            # B1 alternativo: Si III >> Si II (ratio > 1.8) aunque Si IV sea débil
            elif ratio_Si_III_II > 1.8:
                subtype = 'B1'

            # B2: Si III claramente > Si II (ratio > 1.2) pero Si IV ausente
            elif ratio_Si_III_II > 1.2 and Si_IV_4089 < 0.1:
                subtype = 'B2'

            # B2-B3 intermedio: Si III ligeramente > Si II
            elif ratio_Si_III_II > 1.0:
                if Si_III_4553 > 0.25:
                    subtype = 'B2'
                else:
                    subtype = 'B3'

            # B3: Si III ≈ Si II (ratio entre 0.7 y 1.0)
            elif ratio_Si_III_II >= 0.7:
                subtype = 'B3'

            # B4: Si II > Si III (ratio entre 0.3 y 0.7)
            elif ratio_Si_III_II >= 0.3:
                subtype = 'B4'

            # B5: Si II >> Si III (ratio < 0.3)
            elif Si_III_4553 > DETECTION_THRESHOLD or Si_II_4128 > DETECTION_THRESHOLD:
                subtype = 'B5'

            # Caso por defecto (ambas líneas muy débiles)
            else:
                subtype = 'B3'

            # ============================================================
            # DETERMINAR CLASE DE LUMINOSIDAD para B tempranas (B0-B5)
            # ============================================================
            # Criterios MEJORADOS para distinguir IV (subgigante) de V (secuencia principal)
            # Las subgigantes tienen características MUY claras de baja gravedad
            # No debemos clasificar como IV por indicadores débiles

            # Calcular indicadores de luminosidad
            O_II_avg = (measurements.get('O_II_4591', {}).get('ew', 0.0) +
                       measurements.get('O_II_4596', {}).get('ew', 0.0)) / 2.0

            ratio_Mg_He_I = Mg_II_4481 / (He_I_4471 + 0.01)
            ratio_Si_II_III = Si_II_4128 / (Si_III_4553 + 0.01)

            # Contar indicadores de baja gravedad (subgigante/gigante)
            low_gravity_indicators = 0
            strong_indicators = 0  # Indicadores fuertes

            # Indicador 1: O II presente (más fuerte en bajas gravedades)
            if O_II_avg > 0.3:
                low_gravity_indicators += 1
                strong_indicators += 1
            elif O_II_avg > 0.15:
                low_gravity_indicators += 1

            # Indicador 2: Mg II relativamente fuerte
            # AJUSTADO: Más estricto, debe ser > 0.4 para contar como fuerte
            if ratio_Mg_He_I > 0.5:
                low_gravity_indicators += 1
                if ratio_Mg_He_I > 0.8:
                    strong_indicators += 1

            # Indicador 3: Si II comparable o mayor que Si III (ionización reducida)
            # AJUSTADO: Más estricto, debe ser > 0.9 para contar
            if ratio_Si_II_III > 1.0:
                low_gravity_indicators += 1
                strong_indicators += 1
            elif ratio_Si_II_III > 0.85:
                low_gravity_indicators += 1

            # Indicador 4: H débil para el tipo espectral
            # AJUSTADO: Más estricto
            if H_avg < 1.5 and He_I_4471 > 0.5:
                low_gravity_indicators += 1
                if H_avg < 1.0:
                    strong_indicators += 1

            # Clasificar luminosidad basado en indicadores
            # CRITERIOS MÁS ESTRICTOS:
            # - III (gigante): 3+ indicadores, al menos 2 fuertes
            # - IV (subgigante): 3+ indicadores O 2 indicadores fuertes
            # - V (secuencia principal): todo lo demás

            if low_gravity_indicators >= 3 and strong_indicators >= 2:
                # Gigante: múltiples indicadores fuertes
                subtype += ' III'
            elif low_gravity_indicators >= 3 or strong_indicators >= 2:
                # Subgigante: indicadores claros pero no tan extremos
                subtype += ' IV'
            else:
                # Secuencia principal: 0-2 indicadores débiles
                subtype += ' V'

        else:
            # B TARDÍAS: Mg II comparable o mayor que He I
            ratio_Mg_He = Mg_II_4481 / (He_I_4471 + 0.01)

            if ratio_Mg_He < 1.5:
                subtype = 'B6-B7'
            elif ratio_Mg_He < 3.0:
                subtype = 'B8'
            else:
                subtype = 'B9-A0'

            # Agregar clase V por defecto para B tardías (simplificado)
            subtype += ' V'

    # ------------------------------------------------------------
    # TIPOS A y más tardíos: Sin He
    # ------------------------------------------------------------
    else:
        # ============================================================
        # VERIFICACIÓN CRÍTICA: ¿Es realmente sin He o es O8-O9/B9?
        # ============================================================
        # PROBLEMA: O8-O9 puede tener He II muy débil (< 0.01 Å, no detectado)
        # pero He I fuerte (0.3-1.0 Å). Si los filtros rechazan He, cae aquí.
        # SOLUCIÓN: Si He I medido es fuerte + sin metales → es B9/O9.7, no A

        # Verificar He I real (medido, ignorando filtros)
        He_I_total = He_I_4471 + He_I_4026 + He_I_4922 + He_I_5016
        has_strong_He_I_measured = He_I_total > 0.5  # He I fuerte en mediciones

        # Sin metales fuertes (no es tipo F/G/K)
        no_strong_metals = (Fe_I_4046 < 0.3 and Fe_I_4144 < 0.3 and Ca_I_4227 < 0.3)

        # Si He I fuerte medido + sin metales → es tipo B tardío (B9/O9.7), no A
        if has_strong_He_I_measured and no_strong_metals:
            spectral_type = 'B'
            subtype = 'B9-B9.5'  # Muy tardío, cerca de O9.7-B0
            # Nota: Probablemente es O9.7 pero He II no detectado
            # Retornar temprano para evitar sobrescritura
            diagnostics['clasificacion'] = 'B tardía por He I fuerte sin metales'
            return spectral_type, subtype, diagnostics

        # ========================================
        # IMPORTANTE: El orden de verificación es CRÍTICO
        # 1. PRIMERO: Verificar Ca II K / H epsilon (distingue A vs F)
        # 2. DESPUÉS: Verificar intensidad de Balmer (subtipos)
        # ========================================

        # ========================================
        # VERIFICACIÓN CRÍTICA: Rango espectral disponible
        # ========================================
        # Determinar si el espectro cubre las líneas diagnósticas clave
        # Ca II K (3933.7 Å) y H epsilon (3970.1 Å) son cruciales para A/F/G/K
        # Elodie típicamente cubre 4000-5690 Å → NO incluye estas líneas

        wave_min = float(wavelengths[0]) if wavelengths is not None and len(wavelengths) > 0 else 3900.0
        wave_max = float(wavelengths[-1]) if wavelengths is not None and len(wavelengths) > 0 else 7000.0

        # Margen de 10 Å para considerar que una línea está "cubierta"
        covers_Ca_II_K = bool(wave_min <= 3923.7)  # Ca II K a 3933.7 Å - 10 Å margen
        covers_H_epsilon = bool(wave_min <= 3960.1)  # H epsilon a 3970.1 Å - 10 Å margen
        covers_diagnostic_blue = covers_Ca_II_K and covers_H_epsilon

        # Guardar en diagnósticos (convertir a tipos Python nativos para JSON)
        diagnostics['wave_range'] = f"{wave_min:.1f}-{wave_max:.1f}"
        diagnostics['covers_Ca_II_K'] = bool(covers_Ca_II_K)
        diagnostics['covers_H_epsilon'] = bool(covers_H_epsilon)

        # Calcular ratio SOLO si las líneas están cubiertas
        if covers_diagnostic_blue and H_epsilon > 0:
            ratio_Ca_H_epsilon = Ca_II_K / (H_epsilon + 0.01)
        else:
            # Si no cubre las líneas, usar valor neutro que NO active clasificación A
            ratio_Ca_H_epsilon = 0.75  # Valor neutro entre A (<0.5) y F (>0.5)

        # Verificar si tenemos Ca II K disponible (definir ANTES de usarlo)
        # CORREGIDO: Solo marcar como disponible si el espectro cubre esa región
        has_Ca_II_K = covers_Ca_II_K and Ca_II_K > 0.05

        # Verificar cuántas líneas de Balmer se distinguen bien
        balmer_detectadas = sum([
            H_beta > 0.5,      # H beta 4861
            H_gamma > 0.5,     # H gamma 4340
            H_delta > 0.5,     # H delta 4101
            H_epsilon > 0.5    # H epsilon 3970
        ])

        # ========================================
        # PRE-CALCULAR CRITERIOS DE CLASIFICACIÓN
        # ========================================
        # RATIOS diagnósticos (calculados SIEMPRE para usar en subtipos)
        ratio_Fe_I_H_beta = Fe_I_4046 / (H_beta + 0.01)
        ratio_Ca_I_H_gamma = Ca_I_4227 / (H_gamma + 0.01)

        # CRITERIO 3: Distinguir A de F cuando no hay Ca II K
        # En tipo A: Balmer domina (H > metales × 2)
        # En tipo F: metales comparables a Balmer (metales > H/2)
        metales_promedio = (Fe_I_4046 + Fe_I_4383 + Ca_I_4227) / 3.0
        balmer_domina_metales = (H_avg > 2.0 and metales_promedio < H_avg / 2.0)

        # Criterios para tipo A:
        # 1. Balmer fuerte (>= 2.4 Å) sin He I
        # 2. O Balmer moderado (> 2.0 Å) Y Ca II K débil (ratio Ca/H < 0.6)
        # 3. O Balmer moderado (> 2.0 Å) sin Ca II K PERO Balmer >> metales (A tardía)
        es_tipo_A_por_balmer = (H_avg >= 2.4 and not has_He_I) or \
                               (H_avg > 2.0 and not has_He_I and
                                has_Ca_II_K and ratio_Ca_H_epsilon < 0.6) or \
                               (not has_He_I and not has_Ca_II_K and balmer_domina_metales)

        # Calcular es_tipo_F (necesario para elif posterior)
        if has_Ca_II_K:
            # Usar criterios basados en Ca II K
            es_tipo_F = (balmer_detectadas >= 3 and ratio_Ca_H_epsilon >= 0.5) or \
                        (Ca_II_K > 2.0 and 1.0 < H_avg < 5.0 and ratio_Ca_H_epsilon >= 0.3)
        else:
            # CRITERIO ALTERNATIVO sin Ca II K (espectro no cubre 3933 Å)
            # MEJORA 2026-01-18: Usar ratios en lugar de valores absolutos
            # Más robusto para diferentes S/N y resoluciones espectrales
            # (ratios ya calculados arriba en PRE-CALCULAR)

            sin_helio = (not has_He_II and not has_He_I)
            H_moderado = (1.0 < H_avg < 5.0)

            # Criterios mejorados basados en ratios
            # A5-A7: Fe I/H < 0.15, Ca I/H < 0.20 (Balmer domina)
            # F0-F2: 0.15 < Fe I/H < 0.30, 0.20 < Ca I/H < 0.40
            # F5-F8: Fe I/H > 0.30, Ca I/H > 0.40
            # G0+: Fe I/H > 0.50, Ca I/H > 0.80, H débil

            tiene_metales_F = (ratio_Fe_I_H_beta > 0.15 or ratio_Ca_I_H_gamma > 0.20)
            metales_muy_fuertes_G = (ratio_Fe_I_H_beta > 0.50 and ratio_Ca_I_H_gamma > 0.80)

            # Excluir casos que deberían ser tipo G (ratios muy altos + H débil)
            H_muy_debil_para_F = (H_avg < 2.0 and metales_muy_fuertes_G)

            # Excluir casos que son tipo A (Balmer domina)
            es_tipo_A = (H_avg > 4.5 and ratio_Fe_I_H_beta < 0.15 and ratio_Ca_I_H_gamma < 0.20)

            es_tipo_F = H_moderado and tiene_metales_F and sin_helio and not H_muy_debil_para_F and not es_tipo_A

        # ========================================
        # FILTRO CRÍTICO: Detectar tipo K/G por líneas metálicas
        # ========================================
        # CORRECCIÓN 2026-01-19: Si hay MUCHAS líneas metálicas fuertes,
        # NO puede ser tipo A (donde Balmer domina y metales son débiles)
        # Tipo K tiene muchas líneas metálicas no resueltas/mezcladas

        # Contar líneas metálicas detectadas con EW significativo
        n_lineas_metalicas = sum([
            Fe_I_4046 > 0.15,
            Fe_I_4144 > 0.15,
            Fe_I_4260 > 0.15,
            Fe_I_4325 > 0.15,
            Fe_I_4383 > 0.15,
            Fe_I_4957 > 0.15,
            Ca_I_4227 > 0.15,
            Cr_I_4254 > 0.10,
        ])

        # Suma total de EW de metales (indicador de tipo tardío)
        suma_metales = Fe_I_4046 + Fe_I_4144 + Fe_I_4260 + Fe_I_4383 + Ca_I_4227 + Cr_I_4254

        # Criterio: Si hay muchas líneas metálicas Y Balmer no es claramente dominante
        # → Es tipo G/K, NO tipo A
        tiene_muchas_lineas_metalicas = (n_lineas_metalicas >= 4 or suma_metales > 1.5)
        balmer_no_domina = (H_avg < 4.0 or metales_promedio > H_avg * 0.3)

        es_tipo_tardio_por_metales = tiene_muchas_lineas_metalicas and balmer_no_domina

        # Guardar diagnóstico (convertir a tipos nativos de Python para JSON)
        diagnostics['n_lineas_metalicas'] = int(n_lineas_metalicas)
        diagnostics['suma_metales'] = round(float(suma_metales), 3)
        diagnostics['es_tipo_tardio_por_metales'] = bool(es_tipo_tardio_por_metales)

        # ========================================
        # TIPO A: Ca II K ≪ H epsilon
        # ========================================
        # CRITERIO PRINCIPAL: Ca II K es mucho más débil que H epsilon
        # En A0: ratio < 0.2 (Ca II K casi ausente)
        # En F0: ratio ~ 0.5-1.0 (Ca II K comparable a H epsilon)
        # CORRECCIÓN: NO aplicar si hay muchas líneas metálicas (tipo G/K)

        # NOTA: Solo usar ratio_Ca_H_epsilon si el espectro cubre esas líneas
        if covers_diagnostic_blue and ratio_Ca_H_epsilon < 0.5 and not es_tipo_tardio_por_metales:
            # Ca II K ≪ H epsilon → TIPO A (solo si tenemos datos de estas líneas)
            spectral_type = 'A'

            # Subtipos basados en Ca II K / H epsilon
            if ratio_Ca_H_epsilon < 0.2:
                subtype = 'A0-A2'  # A temprana: Ca II K muy débil
            elif ratio_Ca_H_epsilon < 0.35:
                subtype = 'A3-A5'  # A intermedia
            else:
                subtype = 'A7-A9'  # A tardía: Ca II K ~ 1/2 × H epsilon

        # ========================================
        # RAMA ESPECIAL: Tipo tardío detectado por líneas metálicas
        # ========================================
        # CORRECCIÓN 2026-01-19: Si tiene muchas líneas metálicas → tipo G/K
        # Esto evita clasificar K como A cuando Ca II K está fuera del rango

        elif es_tipo_tardio_por_metales:
            # Tiene muchas líneas metálicas → tipo G, K o M
            # CORRECCIÓN: Verificar PRIMERO si hay bandas TiO (tipo M)

            # También verificar TiO_4955 directamente desde measurements
            TiO_4955_ew = measurements.get('TiO_4955', {}).get('ew', 0.0)
            TiO_4955_depth = measurements.get('TiO_4955', {}).get('depth', 0.0)

            # Evaluar bandas TiO si tenemos wavelengths y flux_normalized
            if wavelengths is not None and flux_normalized is not None:
                tio_strength, n_bands, tio_avg_depth = evaluate_tio_bands(
                    wavelengths, flux_normalized, measurements
                )
            else:
                # Sin datos de espectro, usar solo measurements
                tio_strength = 'ausente'
                n_bands = 0
                tio_avg_depth = TiO_4955_depth

            # Criterio para tipo M: bandas TiO detectadas
            es_tipo_M = (tio_strength in ['moderado', 'fuerte', 'muy_fuerte'] or
                        n_bands >= 2 or
                        tio_avg_depth > 0.10 or
                        TiO_4955_ew > 0.5 or
                        TiO_4955_depth > 0.08)

            if es_tipo_M:
                # ES TIPO M
                spectral_type = 'M'

                # Criterio M6-M7 "dientes de sierra": 3 bandas TiO fuertes Y Fe I 4957 desaparece
                fe_4957_depth_leg = measurements.get('Fe_I_4957', {}).get('depth', 0.0)
                TiO_4762_depth_leg = measurements.get('TiO_4762', {}).get('depth', 0.0)
                TiO_4955_depth_leg = measurements.get('TiO_4955', {}).get('depth', 0.0)
                TiO_5167_depth_leg = measurements.get('TiO_5167', {}).get('depth', 0.0)
                todas_tio_leg = (TiO_4762_depth_leg > 0.30 and
                                 TiO_4955_depth_leg > 0.30 and
                                 TiO_5167_depth_leg > 0.30)

                if tio_strength == 'muy_fuerte' and todas_tio_leg and fe_4957_depth_leg < 0.05:
                    subtype = 'M6-M7'
                elif tio_strength == 'muy_fuerte' or tio_avg_depth > 0.4:
                    subtype = 'M5-M6'
                elif tio_strength == 'fuerte' or tio_avg_depth > 0.25:
                    subtype = 'M2-M4'
                elif tio_strength == 'moderado' or tio_avg_depth > 0.12:
                    subtype = 'M0-M2'
                else:
                    subtype = 'M0-M2'

                diagnostics['TiO_strength'] = tio_strength
                diagnostics['TiO_n_bands'] = int(n_bands)
                diagnostics['TiO_avg_depth'] = float(tio_avg_depth)

            else:
                # Sin TiO → tipo G o K
                ratio_Ca_I_H = Ca_I_4227 / (H_avg + 0.01)
                Cr_vs_Fe_leg = Cr_I_4254 / (Fe_I_4260 + 0.01) \
                    if (Fe_I_4260 > 0.05 and Cr_I_4254 > 0.05) else None

                if ratio_Ca_I_H > 1.5 or Ca_I_4227 > 1.0 or suma_metales > 2.5:
                    # Metales muy dominantes → tipo K
                    spectral_type = 'K'

                    if Fe_I_4144 > 1.2 or Ca_I_4227 > 2.0:
                        subtype = 'K5-K7'  # K tardía
                    elif Fe_I_4046 > 0.8 or Ca_I_4227 > 1.2:
                        subtype = 'K3-K5'  # K intermedia
                    elif Fe_I_4144 > 0.5:
                        subtype = 'K2-K4'
                    else:
                        subtype = 'K0-K3'  # K temprana

                elif 0.8 < ratio_Ca_I_H <= 1.5 and Cr_vs_Fe_leg is not None:
                    # Zona ambigua G/K: usar Cr I 4254 / Fe I 4260 como árbitro
                    # Cr < Fe → G tardía; Cr > Fe → K temprana (Gray & Corbally)
                    lineas_usadas.extend(['Cr I 4254', 'Fe I 4260'])
                    if Cr_vs_Fe_leg < 1.0:
                        spectral_type = 'G'
                        subtype = 'G5-G8'
                        justificacion.append(
                            f"Zona G/K: Cr I < Fe I (ratio={Cr_vs_Fe_leg:.2f}) → G tardía")
                    else:
                        spectral_type = 'K'
                        subtype = 'K0-K2'
                        justificacion.append(
                            f"Zona G/K: Cr I > Fe I (ratio={Cr_vs_Fe_leg:.2f}) → K temprana")

                else:
                    # Metales moderados → tipo G
                    spectral_type = 'G'

                    if Fe_I_4144 > 1.0 or Ca_I_4227 > 1.0:
                        subtype = 'G5-G8'  # G tardía
                    elif Fe_I_4046 > 0.8 or Fe_I_4144 > 0.7:
                        subtype = 'G3-G5'
                    else:
                        subtype = 'G0-G3'

                    # Refinar G con Cr/Fe incluso dentro de G claro
                    if Cr_vs_Fe_leg is not None:
                        lineas_usadas.extend(['Cr I 4254', 'Fe I 4260'])
                        if Cr_vs_Fe_leg > 1.10 and subtype in ('G0-G3', 'G3-G5'):
                            subtype = 'G3-G5'  # Cr elevado → G más tardía
                            justificacion.append(
                                f"Cr I > Fe I (ratio={Cr_vs_Fe_leg:.2f}) → G tardía")
                        elif Cr_vs_Fe_leg < 0.80 and subtype == 'G5-G8':
                            subtype = 'G3-G5'   # Cr bajo → G más temprana
                            justificacion.append(
                                f"Cr I < Fe I (ratio={Cr_vs_Fe_leg:.2f}) → G intermedia")

        # ========================================
        # TIPO A (criterio adicional): Balmer FUERTE
        # ========================================
        # CORRECCIÓN 2024-12-14: Si Balmer es fuerte Y/O Ca II K débil en comparación,
        # es tipo A independientemente de Ca II K disponible
        # Esto previene que A5 se clasifique erróneamente como F o G
        # CORRECCIÓN 2026-01-19: Excluir si tiene muchas líneas metálicas

        elif es_tipo_A_por_balmer and not es_tipo_tardio_por_metales:
            spectral_type = 'A'

            # Subtipos según intensidad de Balmer
            if H_avg > 8.0:
                subtype = 'A0-A2'  # Balmer en máximo absoluto
            elif H_avg > 6.0:
                subtype = 'A2-A5'
            elif H_avg > 4.5:
                subtype = 'A5-A7'
            elif H_avg > 3.0:
                subtype = 'A7-F0'  # Transición a F
            else:  # 2.0 < H_avg <= 3.0
                subtype = 'A5-F0'  # A tardía con H moderado

        # ========================================
        # TIPO F: Ca II K comparable a H epsilon Y Balmer visible
        # ========================================
        # Usa el valor es_tipo_F pre-calculado arriba (líneas 992-1012)

        elif es_tipo_F:
            spectral_type = 'F'

            # MEJORA 2026-01-18: Subtipos basados en ratios (más preciso)
            # Usar ratio_Fe_I_H_beta y H_avg para subtipo

            if H_avg > 5.0:  # Balmer muy fuerte
                if ratio_Fe_I_H_beta < 0.20:
                    subtype = 'F0-F2'  # F temprana: H domina, metales débiles
                else:
                    subtype = 'F2-F5'  # F intermedia
            elif H_avg > 3.0:  # Balmer moderado
                if ratio_Fe_I_H_beta < 0.35:
                    subtype = 'F5-F7'  # F intermedia-tardía
                else:
                    subtype = 'F7-F8'  # F tardía
            else:  # Balmer débil pero aún visible
                if ratio_Fe_I_H_beta > 0.50:
                    subtype = 'F8-G0'  # Transición a G
                else:
                    subtype = 'F8'  # F tardía

        # =====================================================
        # CRITERIO 2: Si Ca I 4226.7 es mucho más intensa que las demás
        #             Y se ven bandas de TiO → TIPO K tardío o M
        # =====================================================
        elif Ca_I_4227 > 1.0 and Ca_I_4227 > H_avg * 2.0:
            # Ca I 4227 mucho más intensa que H (ratio > 2.0)

            # ============================================================
            # DETECCIÓN MEJORADA DE TIPO M (según esquema estándar)
            # ============================================================
            # Si tenemos wavelengths y flux, usar detección avanzada
            if wavelengths is not None and flux_normalized is not None:
                # Evaluar asimetría de Fe I 4957 (diagnóstico crítico)
                asymmetry_ratio, flux_left, flux_right, has_tio_band = \
                    detect_FeI_4957_asymmetry(wavelengths, flux_normalized)

                # Evaluar bandas TiO (4762, 4955, 5167)
                tio_strength, n_bands, tio_avg_depth = \
                    evaluate_tio_bands(wavelengths, flux_normalized, measurements)

                # Guardar diagnósticos adicionales
                diagnostics['FeI_4957_asymmetry'] = asymmetry_ratio
                diagnostics['FeI_4957_has_TiO'] = has_tio_band
                diagnostics['TiO_strength'] = tio_strength
                diagnostics['TiO_n_bands'] = int(n_bands)
                diagnostics['TiO_avg_depth'] = float(tio_avg_depth)

                # Clasificación basada en esquema estándar
                if tio_strength in ['moderado', 'fuerte', 'muy_fuerte'] or has_tio_band:
                    # ES TIPO M
                    spectral_type = 'M'

                    # Criterio M6-M7 "dientes de sierra": 3 bandas TiO fuertes Y Fe I desaparece
                    fe_4957_depth_leg2 = measurements.get('Fe_I_4957', {}).get('depth', 0.0)
                    TiO_4762_d2 = measurements.get('TiO_4762', {}).get('depth', 0.0)
                    TiO_4955_d2 = measurements.get('TiO_4955', {}).get('depth', 0.0)
                    TiO_5167_d2 = measurements.get('TiO_5167', {}).get('depth', 0.0)
                    todas_tio_2 = (TiO_4762_d2 > 0.30 and
                                   TiO_4955_d2 > 0.30 and
                                   TiO_5167_d2 > 0.30)

                    if tio_strength == 'muy_fuerte' and todas_tio_2 and fe_4957_depth_leg2 < 0.05:
                        subtype = 'M6-M7'
                    elif tio_strength == 'muy_fuerte':
                        # TiO domina el espectro
                        subtype = 'M5-M6'
                    elif tio_avg_depth > 0.35:
                        # TiO fuerte, Fe I puede estar desapareciendo
                        if asymmetry_ratio < 0.2:
                            subtype = 'M4'  # Fe I casi desaparece
                        else:
                            subtype = 'M2-M4'
                    elif tio_avg_depth > 0.15 or n_bands >= 2:
                        # TiO moderado, Fe I asimétrica
                        if asymmetry_ratio > 0.5:
                            subtype = 'M0'  # Asimetría clara (~50%)
                        else:
                            subtype = 'M0-M2'
                    else:
                        # TiO débil pero presente
                        subtype = 'M0'

                elif tio_strength == 'debil':
                    # TiO apenas visible → K tardía
                    spectral_type = 'K'
                    subtype = 'K5-K7'
                else:
                    # Sin TiO → K intermedia
                    spectral_type = 'K'
                    if Fe_I_4144 > 1.0:
                        subtype = 'K3-K5'
                    elif Fe_I_4046 > 0.8:
                        subtype = 'K2-K4'
                    else:
                        subtype = 'K0-K3'

            else:
                # FALLBACK: Detección básica sin wavelengths
                # (mantiene compatibilidad con código antiguo)
                tiene_TiO = TiO_4955 > DETECTION_THRESHOLD

                if tiene_TiO:
                    if TiO_4955 < 0.3:
                        spectral_type = 'K'
                        subtype = 'K5-K7'
                    elif TiO_4955 < 1.0:
                        spectral_type = 'M'
                        subtype = 'M0-M2'
                    elif TiO_4955 < 2.0:
                        spectral_type = 'M'
                        subtype = 'M2-M4'
                    elif TiO_4955 < 3.0:
                        spectral_type = 'M'
                        subtype = 'M4-M6'
                    else:
                        # EW muy grande → espectro dominado por TiO (dientes de sierra)
                        spectral_type = 'M'
                        subtype = 'M6-M7'
                else:
                    spectral_type = 'K'
                    if Fe_I_4144 > 1.0:
                        subtype = 'K3-K5'
                    elif Fe_I_4046 > 0.8:
                        subtype = 'K2-K4'
                    else:
                        subtype = 'K0-K3'

        # =====================================================
        # CRITERIO ADICIONAL: Tipo G cuando metales dominan sobre H
        # (incluso sin Ca II K disponible)
        # =====================================================
        elif H_avg < 2.0 and (Fe_I_4046 > 0.5 or Fe_I_4144 > 0.5 or Ca_I_4227 > 0.5):
            # H débil + metales fuertes → tipo G o K
            # CORRECCIÓN 2024-12-14: Distinguir G vs K por ratio Ca I / H

            # Calcular ratio Ca I / H
            ratio_Ca_I_H = Ca_I_4227 / (H_avg + 0.01)

            # VERIFICACIÓN 1: Ca II K débil → tipo A (no G)
            if has_Ca_II_K and Ca_II_K < 2.0:
                # Ca II K débil → NO es tipo G, es tipo A
                spectral_type = 'A'
                subtype = 'A5-A7'  # A tardía con posible problema de medición

            # VERIFICACIÓN 2: Ca I >> H → tipo K (no G)
            elif ratio_Ca_I_H > 2.5 or Ca_I_4227 > 1.5:
                # Ca I domina sobre H → tipo K
                spectral_type = 'K'

                # Subtipos K según intensidad de metales
                if Fe_I_4144 > 1.2 or Ca_I_4227 > 2.0:
                    subtype = 'K5-K7'  # K tardía: metales muy fuertes
                elif Fe_I_4046 > 0.8 or Ca_I_4227 > 1.2:
                    subtype = 'K3-K5'  # K intermedia
                else:
                    subtype = 'K0-K3'  # K temprana

            else:
                # Ca I comparable a H → tipo G
                spectral_type = 'G'

                # Subtipos basados en intensidad de metales vs H
                if Fe_I_4144 > 1.0 or Ca_I_4227 > 1.0:
                    subtype = 'G5-G8'  # G tardía: metales muy fuertes
                elif Fe_I_4046 > 0.8 or Fe_I_4144 > 0.7:
                    subtype = 'G3-G5'  # G intermedia
                else:
                    subtype = 'G0-G3'  # G temprana

        # =====================================================
        # CRITERIO 3: Si NO se destacan ni H I ni Ca I 4226
        #             → TIPO G o K temprana
        # =====================================================
        elif H_avg < 3.0 and Ca_I_4227 < 1.0:
            # Ni H ni Ca I se destacan (ambos moderados/débiles)

            # Distinguir G vs K temprana por ratio H / Ca I
            if H_avg > 1.0 and H_avg >= Ca_I_4227:
                # H todavía comparable o mayor que Ca I → G
                spectral_type = 'G'

                if H_delta > 1.5:
                    subtype = 'G0-G2'  # G temprana: H delta todavía fuerte
                elif H_delta > 0.8:
                    subtype = 'G2-G5'  # G intermedia
                else:
                    subtype = 'G5-G8'  # G tardía

            elif Ca_I_4227 > H_avg and Ca_I_4227 > 0.3:
                # Ca I empieza a dominar sobre H → K temprana
                spectral_type = 'K'

                if H_delta > 0.5:
                    subtype = 'K0-K2'  # K temprana: H todavía visible
                else:
                    subtype = 'K2-K4'  # K intermedia

            else:
                # Caso ambiguo: valores muy bajos en ambos
                # Usar Ca II K como criterio de desempate
                if Ca_II_K > 3.0:
                    spectral_type = 'K'
                    subtype = 'K0-K3'
                else:
                    spectral_type = 'G'
                    subtype = 'G5-K0'

        # =====================================================
        # TRANSICIÓN A-F: Ca II K ~ H epsilon
        # =====================================================
        elif ratio_Ca_H_epsilon >= 0.5 and ratio_Ca_H_epsilon < 1.5:
            # Ca II K comparable a H epsilon → A tardía o F temprana

            if balmer_detectadas >= 2:  # Balmer aún visible
                spectral_type = 'F'
                subtype = 'F0-F2'
            else:
                spectral_type = 'A'
                subtype = 'A8-F0'

        # =====================================================
        # CASO POR DEFECTO: Clasificación por características generales
        # =====================================================
        else:
            # Si llegamos aquí, usar criterios generales

            if Ca_II_K > 3.0:
                # Ca II K muy fuerte → tipo tardío
                if H_avg > 2.0:
                    spectral_type = 'G'
                    subtype = 'G0-G5'
                else:
                    spectral_type = 'K'
                    subtype = 'K0-K5'

            elif H_avg > 5.0:
                # H muy fuerte, Ca débil → F
                spectral_type = 'F'
                subtype = 'F0-F5'

            elif TiO_4955 > DETECTION_THRESHOLD:
                # TiO presente → M (usar detección mejorada si es posible)
                if wavelengths is not None and flux_normalized is not None:
                    asymmetry_ratio, _, _, has_tio_band = \
                        detect_FeI_4957_asymmetry(wavelengths, flux_normalized)
                    tio_strength, n_bands, tio_avg_depth = \
                        evaluate_tio_bands(wavelengths, flux_normalized, measurements)

                    spectral_type = 'M'
                    if tio_avg_depth > 0.3:
                        subtype = 'M2-M5'
                    elif asymmetry_ratio > 0.5:
                        subtype = 'M0'
                    else:
                        subtype = 'M0-M2'

                    # Guardar diagnósticos
                    diagnostics['FeI_4957_asymmetry'] = asymmetry_ratio
                    diagnostics['TiO_strength'] = tio_strength
                    diagnostics['TiO_n_bands'] = n_bands
                    diagnostics['TiO_avg_depth'] = tio_avg_depth
                else:
                    # Fallback básico
                    spectral_type = 'M'
                    subtype = 'M0-M3'

            else:
                # Sin características claras
                spectral_type = 'G'
                subtype = 'desconocido'

    # ── Rellenar campos de interfaz web en el clasificador legacy ──────────
    # Construir lineas_usadas según el tipo resultante
    _lineas = []
    _just = []
    _adv = []

    if spectral_type == 'O':
        _lineas = ['He II 4200', 'He II 4542', 'He II 4686', 'He I 4471', 'He I 4387']
        _just = [
            f'He II detectado (4686={He_II_4686:.2f}, 4542={He_II_4542:.2f}, 4200={He_II_4200:.2f}) → Tipo O',
            f'Subtipo por ratio He I/He II: {subtype}'
        ]
    elif spectral_type == 'B':
        _lineas = ['He I 4471', 'He I 4026', 'He I 4922', 'He I 5016', 'Mg II 4481',
                   'Si IV 4089', 'Si III 4553', 'Si II 4128']
        _just = [
            f'He I presente ({He_I_4471:.2f} Å), He II ausente → Tipo B',
            f'Subtipo por Si/Mg: {subtype}'
        ]
    elif spectral_type == 'A':
        _lineas = ['Ca II K 3933', 'H epsilon 3970', 'H beta 4861', 'H gamma 4341']
        if covers_blue_leg:
            ratio = diagnostics.get('ratio_Ca_H_epsilon', 0)
            _just = [
                f'Balmer domina, He ausente → Tipo A',
                f'Ca II K / Hε = {ratio:.2f} → {subtype}'
            ]
        else:
            _just = [f'Balmer domina, sin Ca II K en rango → Tipo A ({subtype})']
            _adv = ['Sin cobertura de Ca II K / Hε, usando criterios alternativos']
    elif spectral_type == 'F':
        _lineas = ['Ca II K 3933', 'H epsilon 3970', 'H beta 4861', 'Fe I 4383', 'Mg II 4481']
        if covers_blue_leg:
            ratio = diagnostics.get('ratio_Ca_H_epsilon', 0)
            _just = [f'Ca II K / Hε = {ratio:.2f} → F ({subtype})']
        else:
            _just = [f'Balmer moderado, metales visibles → F ({subtype})']
            _adv = ['Sin cobertura de Ca II K / Hε, usando Balmer + metales']
    elif spectral_type == 'G':
        cr_fe = diagnostics.get('ratio_Cr_Fe', 0)
        _lineas = ['H delta 4102', 'Ca I 4227', 'Fe I 4046', 'Fe I 4144', 'Cr I 4254', 'Fe I 4260']
        _just = [
            f'Balmer débil, metales moderados → G/K',
            f'Cr I / Fe I = {cr_fe:.2f} → G ({subtype})'
        ]
    elif spectral_type == 'K':
        cr_fe = diagnostics.get('ratio_Cr_Fe', 0)
        _lineas = ['Ca II K 3933', 'Ca I 4227', 'Fe I 4046', 'Fe I 4144', 'Cr I 4254', 'Fe I 4260']
        _just = [
            f'Metales dominan sobre Balmer → G/K',
            f'Cr I / Fe I = {cr_fe:.2f} → K ({subtype})'
        ]
    elif spectral_type == 'M':
        tio_ratio = diagnostics.get('ratio_TiO_Fe', 0)
        _lineas = ['TiO 4762', 'TiO 4955', 'TiO 5167', 'Fe I 4957', 'Ca I 4227']
        _just = [
            f'Bandas TiO detectadas (ratio TiO/Fe = {tio_ratio:.2f}) → M',
            f'Subtipo por intensidad TiO y asimetría Fe I 4957: {subtype}'
        ]
        if 'M6' in subtype or 'M7' in subtype:
            _just.append("Espectro dominado por TiO en 3 bandas, Fe I 4957 desaparece → 'dientes de sierra' M6-M7")

    if not covers_blue_leg and spectral_type in ['A', 'F', 'G', 'K']:
        _adv.append('Sin cobertura de Ca II K / Hε (λ < 3950 Å): clasificación con incertidumbre aumentada')

    diagnostics['lineas_usadas'] = _lineas
    diagnostics['justificacion'] = ' | '.join(_just)
    diagnostics['advertencias'] = _adv
    diagnostics['clasificacion_gruesa'] = (
        'temprana' if spectral_type in ['O', 'B'] else
        'intermedia' if spectral_type in ['A', 'F'] else
        'tardia'
    )

    # ============================================================
    # CLASE DE LUMINOSIDAD MK
    # ============================================================
    # Usa las funciones importadas al inicio del módulo (sin try/from src).
    # Si el módulo no está disponible, los fallbacks ya están definidos arriba.
    luminosity_class = estimate_luminosity_class(measurements, spectral_type)
    mk_full = combine_spectral_and_luminosity(
        subtype if subtype else spectral_type, luminosity_class
    )

    _lum_names = {
        'Ia':  'Supergigante muy luminosa',
        'Ib':  'Supergigante',
        'II':  'Gigante brillante',
        'III': 'Gigante',
        'IV':  'Subgigante',
        'V':   'Secuencia principal (enana)',
    }

    diagnostics['luminosity_class'] = luminosity_class
    diagnostics['mk_full']          = mk_full
    diagnostics['lum_name']         = _lum_names.get(luminosity_class, luminosity_class)

    return spectral_type, subtype, diagnostics


# ============================================================================
# Función principal de procesamiento FITS
# ============================================================================

def process_fits_corrected(file_path):
    """
    Procesa un archivo FITS con normalización y mediciones corregidas.

    Parameters
    ----------
    file_path : str
        Ruta al archivo FITS

    Returns
    -------
    wavelengths : array
        Longitudes de onda en Å
    flux_normalized : array
        Flujo normalizado al continuo
    continuum : array
        Continuo estimado
    measurements : dict
        Mediciones de líneas espectrales
    objeto : str
        Nombre del objeto
    tipo_fits : str
        Tipo espectral del header FITS
    """
    # Abrir archivo FITS
    hdul = fits.open(file_path)
    data = hdul[0].data
    header = hdul[0].header
    hdul.close()

    # Extraer información del header
    crval1 = header['CRVAL1']
    crpix1 = header['CRPIX1']
    cdelt1 = header['CDELT1']
    objeto = header.get('OBJECT', 'Desconocido')
    tipo_fits = header.get('SPTYPE', 'Desconocido')
    tipo_fits = tipo_fits.strip().upper().replace("/", "")
    tipo_fits = tipo_fits[:5] if len(tipo_fits) > 5 else tipo_fits

    # Calcular eje de longitud de onda
    num_pixels = len(data)
    pixels = np.arange(1, num_pixels + 1)
    wavelengths = crval1 + (pixels - crpix1) * cdelt1

    # CORRECCIÓN: Normalizar al continuo (no min-max) — algoritmo de dos pasos
    flux_normalized, continuum = normalize_to_continuum(wavelengths, data)

    # Medir líneas diagnóstico con banderas de calidad y ventanas adaptativas
    measurements = measure_diagnostic_lines(wavelengths, flux_normalized)

    return wavelengths, flux_normalized, continuum, measurements, objeto, tipo_fits


def process_spectrum_with_diagnostics(wavelengths, flux, output_dir=".",
                                      object_name="espectro",
                                      spectral_type="?", subtype="?",
                                      write_diag_file=True,
                                      min_snr=SNR_MINIMUM,
                                      min_resolution=RESOLUTION_MINIMUM):
    """
    Procesa un espectro completo con diagnósticos avanzados.

    Flujo completo:
    1. Normalización al continuo (dos pasos)
    2. Medición de líneas con ventanas adaptativas y banderas de calidad
    3. Estimación de SNR
    4. Verificación de velocidad radial
    5. (Opcional) Escritura del archivo de diagnóstico

    Parameters
    ----------
    wavelengths : array
        Longitudes de onda en Å
    flux : array
        Flujo observado (cualquier unidad)
    output_dir : str
        Directorio de salida para el archivo de diagnóstico
    object_name : str
        Nombre del objeto (para el nombre del archivo)
    spectral_type : str
        Tipo espectral (se actualiza después de la clasificación)
    subtype : str
        Subtipo espectral
    write_diag_file : bool
        Si True, escribe el archivo line_measurements.txt
    min_snr : float
        SNR mínimo recomendado (advertencia si se supera por debajo)
    min_resolution : float
        Resolución mínima recomendada (solo informativo)

    Returns
    -------
    flux_normalized : array
    continuum : array
    measurements : dict
    snr : float
    rv_info : tuple (mean_shift, rv_estimate, rv_warning)
    diag_file : str or None — ruta al archivo de diagnóstico
    """
    # 1. Normalización
    flux_normalized, continuum = normalize_to_continuum(wavelengths, flux)

    # 2. Medición de líneas
    measurements = measure_diagnostic_lines(wavelengths, flux_normalized)

    # 3. SNR
    snr = compute_snr(wavelengths, flux_normalized)
    if snr < min_snr and min_snr > 0:
        print(f"  ADVERTENCIA SNR: SNR estimado = {snr:.1f} < {min_snr} (mínimo recomendado)")

    # 4. Verificación de velocidad radial
    rv_info = check_radial_velocity(measurements)
    _, _, rv_warning = rv_info
    if rv_warning:
        print(f"  ADVERTENCIA VR: {rv_warning}")

    # 5. Archivo de diagnóstico
    diag_file = None
    if write_diag_file:
        import os
        safe_name = object_name.replace(" ", "_").replace("/", "_")
        diag_path = os.path.join(output_dir, f"{safe_name}_line_measurements.txt")
        diag_file = write_diagnostic_file(
            measurements, output_path=diag_path,
            spectral_type=spectral_type, subtype=subtype,
            snr=snr, rv_info=rv_info
        )

    return flux_normalized, continuum, measurements, snr, rv_info, diag_file


# ============================================================================
# Función de visualización actualizada
# ============================================================================

def plot_spectrum_corrected(wavelengths, flux_normalized, measurements,
                           spectral_type, subtype, objeto, tipo_fits,
                           save_path=None):
    """
    Grafica el espectro con la clasificación corregida y zoom en líneas diagnóstico.

    Parameters
    ----------
    wavelengths : array
        Longitudes de onda
    flux_normalized : array
        Flujo normalizado al continuo
    measurements : dict
        Mediciones de líneas
    spectral_type : str
        Tipo espectral clasificado
    subtype : str
        Subtipo clasificado
    objeto : str
        Nombre del objeto
    tipo_fits : str
        Tipo espectral del FITS
    save_path : str, optional
        Ruta para guardar la figura
    """
    # Definir líneas diagnóstico según el tipo espectral
    diagnostic_regions = _get_diagnostic_regions(spectral_type, measurements, wavelengths)

    # Determinar número de paneles
    n_zoom_panels = len(diagnostic_regions)
    n_rows = 2 + n_zoom_panels  # Espectro completo + barras + zooms

    # Calcular altura dinámica para el panel de barras según número de líneas
    n_lines_detected = sum(1 for data in measurements.values() if data['ew'] > 0.05)
    bar_panel_height = max(2.5, min(6, n_lines_detected * 0.4))  # Entre 2.5 y 6

    # Crear figura con grid dinámico
    fig = plt.figure(figsize=(18, 5 * n_rows))  # Más ancho y alto
    gs = fig.add_gridspec(n_rows, 1, height_ratios=[2.5, bar_panel_height] + [1.8] * n_zoom_panels)

    # ========================================
    # Panel 1: Espectro completo
    # ========================================
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(wavelengths, flux_normalized, 'k-', linewidth=0.5, alpha=0.7, label='Espectro normalizado')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Continuo')

    # Marcar líneas con EW significativo
    for line_name, data in measurements.items():
        if data['ew'] > 0.05:  # Marcar líneas detectadas
            color = _get_line_color(line_name)
            ax1.axvline(x=data['wavelength'], color=color, linestyle=':', alpha=0.4, linewidth=1)
            ax1.text(data['wavelength'], 1.05, line_name.replace('_', ' '), rotation=90,
                    fontsize=7, verticalalignment='bottom', color=color, alpha=0.8)

    ax1.set_ylabel('Flujo Normalizado', fontsize=11)
    ax1.set_title(f'{objeto} - Original: {tipo_fits} | Clasificado: {spectral_type} {subtype}',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 1.3])

    # ========================================
    # Panel 2: Anchos equivalentes (barras)
    # ========================================
    ax2 = fig.add_subplot(gs[1])
    line_names = []
    ews = []
    colors = []

    for line_name, data in measurements.items():
        if data['ew'] > 0.05:  # Solo líneas detectadas
            line_names.append(line_name.replace('_', ' '))
            ews.append(data['ew'])
            colors.append(_get_line_color(line_name))

    if line_names:
        # Barras más gruesas y legibles
        bar_height = 0.7  # Altura de cada barra
        bars = ax2.barh(line_names, ews, color=colors, alpha=0.75,
                       edgecolor='black', linewidth=0.8, height=bar_height)
        ax2.set_xlabel('Ancho Equivalente (Å)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Línea Espectral', fontsize=13, fontweight='bold')
        ax2.set_title('Intensidades de Líneas Diagnóstico (EW)', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.4, axis='x')
        ax2.tick_params(axis='y', labelsize=11)  # Etiquetas de líneas más grandes
        ax2.tick_params(axis='x', labelsize=10)

        # Añadir valores en las barras con mejor formato
        max_ew = max(ews) if ews else 1
        for bar, ew in zip(bars, ews):
            # Posicionar texto dentro o fuera de la barra según su longitud
            if ew > max_ew * 0.3:
                ax2.text(ew - 0.08, bar.get_y() + bar.get_height()/2,
                        f'{ew:.2f} Å', va='center', ha='right', fontsize=10,
                        fontweight='bold', color='white')
            else:
                ax2.text(ew + 0.05, bar.get_y() + bar.get_height()/2,
                        f'{ew:.2f} Å', va='center', ha='left', fontsize=10, fontweight='bold')

    # ========================================
    # Paneles 3+: Zoom en regiones diagnóstico
    # ========================================
    for i, region in enumerate(diagnostic_regions):
        ax_zoom = fig.add_subplot(gs[2 + i])

        # Filtrar datos en la región
        mask = (wavelengths >= region['wave_min']) & (wavelengths <= region['wave_max'])
        wave_region = wavelengths[mask]
        flux_region = flux_normalized[mask]

        if len(wave_region) == 0:
            ax_zoom.text(0.5, 0.5, 'Región no cubierta por el espectro',
                        ha='center', va='center', transform=ax_zoom.transAxes)
            ax_zoom.set_title(region['title'], fontsize=11, fontweight='bold')
            continue

        # Plotear espectro en la región
        ax_zoom.plot(wave_region, flux_region, 'k-', linewidth=1.5, label='Espectro')
        ax_zoom.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # Marcar líneas en esta región
        for line_info in region['lines']:
            line_name = line_info['name']
            line_wave = line_info['wavelength']

            if line_name in measurements:
                ew = measurements[line_name]['ew']
                depth = measurements[line_name]['depth']
                color = _get_line_color(line_name)

                # Línea vertical en el centro
                ax_zoom.axvline(line_wave, color=color, linestyle='-', linewidth=2, alpha=0.7)

                # Región de integración (estimada)
                if ew > 0.05:
                    width = ew * 2
                    ax_zoom.axvspan(line_wave - width/2, line_wave + width/2,
                                   alpha=0.2, color=color)

                # Etiqueta con información
                label_text = f'{line_name.replace("_", " ")}\n'
                label_text += f'EW={ew:.2f}Å'
                if depth > 0.05:
                    label_text += f'\nProf={depth:.2f}'

                ax_zoom.text(line_wave, 1.15, label_text,
                           ha='center', va='bottom', fontsize=9,
                           color=color, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   edgecolor=color, alpha=0.7))

        ax_zoom.set_xlabel('Longitud de Onda (Å)', fontsize=10)
        ax_zoom.set_ylabel('Flujo Normalizado', fontsize=10)
        ax_zoom.set_title(region['title'], fontsize=11, fontweight='bold', color='darkblue')
        ax_zoom.grid(alpha=0.3)
        ax_zoom.set_ylim([0, 1.3])
        ax_zoom.set_xlim([region['wave_min'], region['wave_max']])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def _get_line_color(line_name):
    """Asigna color a una línea según su tipo"""
    if 'He_II' in line_name:
        return 'purple'
    elif 'He_I' in line_name:
        return 'blue'
    elif 'H_' in line_name:
        return 'cyan'
    elif 'Ca_II' in line_name:
        return 'orange'
    elif 'Si_' in line_name:
        return 'green'
    elif 'Mg_' in line_name:
        return 'brown'
    elif 'N_' in line_name or 'O_' in line_name:
        return 'red'
    elif 'Fe_' in line_name:
        return 'darkgray'
    else:
        return 'black'


def _get_diagnostic_regions(spectral_type, measurements, wavelengths):
    """
    Define regiones de interés para zoom según el tipo espectral.
    SOLO incluye regiones y líneas que están DENTRO del rango espectral.

    Returns
    -------
    list of dict
        Cada diccionario contiene: 'title', 'wave_min', 'wave_max', 'lines'
    """
    wave_min, wave_max = float(wavelengths[0]), float(wavelengths[-1])
    regions = []

    def line_in_range(wavelength, margin=10):
        """Verifica si una línea está dentro del rango espectral con margen."""
        return (wave_min - margin) <= wavelength <= (wave_max + margin)

    def filter_lines_in_range(lines):
        """Filtra solo las líneas que están dentro del rango espectral."""
        return [l for l in lines if line_in_range(l['wavelength'])]

    def region_has_valid_lines(region):
        """Verifica si una región tiene al menos una línea válida."""
        valid_lines = filter_lines_in_range(region['lines'])
        return len(valid_lines) > 0

    def adjust_region_to_spectrum(region):
        """Ajusta los límites de la región al rango del espectro."""
        region['wave_min'] = max(region['wave_min'], wave_min)
        region['wave_max'] = min(region['wave_max'], wave_max)
        region['lines'] = filter_lines_in_range(region['lines'])
        return region

    # ============================================================
    # TIPO O: Enfocarse en He II y He I
    # ============================================================
    if spectral_type == 'O':
        # Región 1: He II 4686 (línea principal de tipo O)
        if 4650 <= wave_max and 4720 >= wave_min:
            regions.append({
                'title': 'He II 4686 Å - Diagnóstico Principal Tipo O',
                'wave_min': 4650,
                'wave_max': 4720,
                'lines': [
                    {'name': 'He_II_4686', 'wavelength': 4685.7}
                ]
            })

        # Región 2: He II 4200 Å — tercera línea diagnóstico O (ANTES faltaba)
        if 4170 <= wave_max and 4230 >= wave_min:
            regions.append({
                'title': 'He II 4200 Å - Tercer Diagnóstico Tipo O',
                'wave_min': 4170,
                'wave_max': 4230,
                'lines': [
                    {'name': 'He_II_4200', 'wavelength': 4199.8}
                ]
            })

        # Región 3: He II 4542 vs He I 4471 (discrimina subtipos O)
        if 4400 <= wave_max and 4600 >= wave_min:
            regions.append({
                'title': 'He I 4471 vs He II 4542 - Subtipos O (ratio → O5/O7/O9)',
                'wave_min': 4400,
                'wave_max': 4600,
                'lines': [
                    {'name': 'He_I_4471', 'wavelength': 4471.5},
                    {'name': 'He_II_4542', 'wavelength': 4541.6},
                    {'name': 'Si_III_4553', 'wavelength': 4552.7}
                ]
            })

        # Región 4: He I 4387 + N V 4604 (O tempranas vs tardías)
        if 4350 <= wave_max and 4650 >= wave_min:
            regions.append({
                'title': 'He I 4387 + N V 4604 - O Tempranas vs Tardías',
                'wave_min': 4350,
                'wave_max': 4650,
                'lines': filter_lines_in_range([
                    {'name': 'He_I_4387', 'wavelength': 4387.9},
                    {'name': 'N_V_4604', 'wavelength': 4604.0}
                ])
            })

    # ============================================================
    # TIPO B: Enfocarse en He I, Si, Mg
    # ============================================================
    elif spectral_type == 'B':
        # Región 1: He I 4471 vs Mg II 4481 (discrimina B temprana/tardía)
        if 4400 <= wave_max and 4560 >= wave_min:
            regions.append({
                'title': 'He I 4471 vs Mg II 4481 - B Temprana/Tardía',
                'wave_min': 4430,
                'wave_max': 4530,
                'lines': [
                    {'name': 'He_I_4471', 'wavelength': 4471.5},
                    {'name': 'Mg_II_4481', 'wavelength': 4481.2}
                ]
            })

        # Región 2: Si III/Si IV (B tempranas)
        if 4050 <= wave_max and 4600 >= wave_min:
            regions.append({
                'title': 'Si IV 4089 vs Si III 4553 - Subtipos B Tempranas',
                'wave_min': 4050,
                'wave_max': 4600,
                'lines': [
                    {'name': 'Si_IV_4089', 'wavelength': 4088.9},
                    {'name': 'Si_II_4128', 'wavelength': 4128.1},
                    {'name': 'Si_III_4553', 'wavelength': 4552.7}
                ]
            })

    # ============================================================
    # TIPO A: Enfocarse en Balmer
    # ============================================================
    elif spectral_type == 'A':
        # Región 1: H beta (máxima en A0)
        if 4820 <= wave_max and 4900 >= wave_min:
            regions.append({
                'title': 'H beta 4861 Å - Máxima en A0',
                'wave_min': 4820,
                'wave_max': 4900,
                'lines': [
                    {'name': 'H_beta', 'wavelength': 4861.3}
                ]
            })

        # Región 2: H gamma + Ca II K (SOLO si Ca II K está en el rango)
        if line_in_range(3933.7):  # Ca II K está en el rango
            regions.append({
                'title': 'Ca II K 3933 vs H gamma 4340 - A Temprana/Tardía',
                'wave_min': max(3900, wave_min),
                'wave_max': min(4380, wave_max),
                'lines': filter_lines_in_range([
                    {'name': 'Ca_II_K', 'wavelength': 3933.7},
                    {'name': 'Ca_II_H', 'wavelength': 3968.5},
                    {'name': 'H_gamma', 'wavelength': 4340.5}
                ])
            })
        # Alternativa: H delta + H gamma si no hay cobertura de Ca II K
        elif line_in_range(4101.7) and line_in_range(4340.5):
            regions.append({
                'title': 'H delta 4102 + H gamma 4340 - Serie Balmer',
                'wave_min': max(4050, wave_min),
                'wave_max': min(4400, wave_max),
                'lines': [
                    {'name': 'H_delta', 'wavelength': 4101.7},
                    {'name': 'H_gamma', 'wavelength': 4340.5}
                ]
            })

    # ============================================================
    # TIPOS F, G, K: Enfocarse en Ca II K y metales
    # ============================================================
    elif spectral_type in ['F', 'G', 'K']:
        # Región 1: Ca II K/H (SOLO si están en el rango espectral)
        if line_in_range(3933.7):  # Ca II K está en el rango
            regions.append({
                'title': 'Ca II K/H - Líneas de Metales',
                'wave_min': max(3900, wave_min),
                'wave_max': min(4020, wave_max),
                'lines': filter_lines_in_range([
                    {'name': 'Ca_II_K', 'wavelength': 3933.7},
                    {'name': 'Ca_II_H', 'wavelength': 3968.5}
                ])
            })
        # Alternativa: Fe I y Mg II si no hay cobertura de Ca II K
        elif line_in_range(4481.2) or line_in_range(4383.5):
            lines_alt = filter_lines_in_range([
                {'name': 'Mg_II_4481', 'wavelength': 4481.2},
                {'name': 'Fe_I_4383', 'wavelength': 4383.5},
                {'name': 'Fe_I_4271', 'wavelength': 4271.8}
            ])
            if lines_alt:
                regions.append({
                    'title': 'Metales (Fe I, Mg II) - Alternativa a Ca II K',
                    'wave_min': max(4250, wave_min),
                    'wave_max': min(4520, wave_max),
                    'lines': lines_alt
                })

        # Región 2: H delta (para ver fuerza de Balmer en F, G, K)
        if line_in_range(4101.7):
            regions.append({
                'title': 'H delta 4102 Å - Diagnóstico Balmer',
                'wave_min': max(4060, wave_min),
                'wave_max': min(4150, wave_max),
                'lines': [
                    {'name': 'H_delta', 'wavelength': 4101.7}
                ]
            })

        # Región 3: H beta (para F)
        if spectral_type == 'F' and line_in_range(4861.3):
            regions.append({
                'title': 'H beta - Todavía Visible en F',
                'wave_min': max(4820, wave_min),
                'wave_max': min(4900, wave_max),
                'lines': [
                    {'name': 'H_beta', 'wavelength': 4861.3}
                ]
            })

        # Región 4: Cr I 4254 vs Fe I 4260 — criterio clave G vs K (ANTES faltaba)
        if spectral_type in ['G', 'K'] and (line_in_range(4254.3) or line_in_range(4260.5)):
            regions.append({
                'title': 'Cr I 4254 vs Fe I 4260 - Criterio Decisivo G/K\n'
                         '(Cr I < Fe I → G tardía  |  Cr I > Fe I → K)',
                'wave_min': max(4230, wave_min),
                'wave_max': min(4290, wave_max),
                'lines': filter_lines_in_range([
                    {'name': 'Cr_I_4254', 'wavelength': 4254.3},
                    {'name': 'Fe_I_4260', 'wavelength': 4260.5}
                ])
            })

        # Región 5: Fe I 4957 y Cr I (para G, K)
        if spectral_type in ['G', 'K'] and line_in_range(4957.6):
            regions.append({
                'title': 'Fe I 4957 + Cr I - Metales Tardíos',
                'wave_min': max(4900, wave_min),
                'wave_max': min(5050, wave_max),
                'lines': filter_lines_in_range([
                    {'name': 'Fe_I_4957', 'wavelength': 4957.6},
                    {'name': 'Cr_I_5206', 'wavelength': 5206.0}
                ])
            })

    # Si no hay regiones específicas, mostrar regiones genéricas
    if len(regions) == 0:
        # Región genérica de Balmer (solo líneas que estén en el rango)
        balmer_lines = filter_lines_in_range([
            {'name': 'H_delta', 'wavelength': 4101.7},
            {'name': 'H_gamma', 'wavelength': 4340.5},
            {'name': 'H_beta', 'wavelength': 4861.3}
        ])
        if balmer_lines:
            # Calcular wave_min y wave_max basados en las líneas disponibles
            line_wavelengths = [l['wavelength'] for l in balmer_lines]
            region_min = max(min(line_wavelengths) - 50, wave_min)
            region_max = min(max(line_wavelengths) + 50, wave_max)
            regions.append({
                'title': 'Serie de Balmer - Líneas Disponibles',
                'wave_min': region_min,
                'wave_max': region_max,
                'lines': balmer_lines
            })

    return regions


# ============================================================================
# ARCHIVO DIAGNÓSTICO DE MEDICIONES (Mejora 7)
# ============================================================================

def write_diagnostic_file(measurements, output_path="line_measurements.txt",
                           spectral_type="?", subtype="?",
                           snr=None, rv_info=None):
    """
    Genera un archivo de texto con todas las mediciones de líneas.

    Formato de columnas:
        line_name  lambda_lab  lambda_meas  window_width  EqW  quality_flag

    Este archivo es esencial para la reproducibilidad científica y la
    depuración de mediciones erróneas. Ejemplo de fila:
        HeI4471   4471.48   4471.32   12.0   0.82   OK

    Parameters
    ----------
    measurements : dict
        Diccionario de mediciones de measure_diagnostic_lines()
    output_path : str
        Ruta del archivo de salida (default: 'line_measurements.txt')
    spectral_type : str
        Tipo espectral clasificado (para el encabezado)
    subtype : str
        Subtipo espectral (para el encabezado)
    snr : float or None
        SNR estimado del espectro
    rv_info : tuple or None
        (mean_shift, rv_est, rv_warning) de check_radial_velocity()

    Returns
    -------
    output_path : str
        Ruta del archivo creado
    """
    lines_out = []

    # ── Encabezado ──────────────────────────────────────────────────────
    lines_out.append("# ============================================================")
    lines_out.append("#  SpectroClass — Archivo de Diagnóstico de Mediciones")
    lines_out.append("# ============================================================")
    lines_out.append(f"#  Tipo espectral clasificado : {spectral_type} {subtype}".rstrip())
    if snr is not None:
        quality_snr = "BUENO" if snr >= SNR_MINIMUM else "BAJO"
        lines_out.append(f"#  SNR estimado               : {snr:.1f}  [{quality_snr}; mínimo recomendado: {SNR_MINIMUM}]")
    if rv_info is not None:
        mean_shift, rv_est, rv_warn = rv_info
        lines_out.append(f"#  Desplazamiento medio (Δλ)  : {mean_shift:+.3f} Å  "
                         f"(VR ≈ {rv_est:+.0f} km/s)")
        if rv_warn:
            lines_out.append(f"#  ADVERTENCIA VR             : {rv_warn}")
    lines_out.append("#")
    lines_out.append("#  Nota: Las mediciones con quality_flag != 'OK' deben usarse")
    lines_out.append("#  con precaución en la clasificación.")
    lines_out.append("# ============================================================")
    lines_out.append("")

    # ── Cabecera de columnas ─────────────────────────────────────────────
    col_fmt = "{:<22} {:>10} {:>12} {:>12} {:>8} {:<20}"
    lines_out.append(col_fmt.format(
        "line_name", "lambda_lab", "lambda_meas", "window_width", "EqW", "quality_flag"
    ))
    lines_out.append("-" * 88)

    # ── Filas de datos — ordenadas por longitud de onda de laboratorio ───
    sorted_lines = sorted(
        measurements.items(),
        key=lambda x: x[1].get('wavelength', 0.0) if isinstance(x[1], dict) else 0.0
    )

    for line_name, data in sorted_lines:
        if not isinstance(data, dict):
            continue

        lambda_lab   = data.get('wavelength', 0.0)
        lambda_meas  = data.get('lambda_measured', 0.0)
        win_width    = data.get('window_width', 12.0)
        ew           = data.get('ew', 0.0)
        qflag        = data.get('quality_flag', 'UNKNOWN')

        # Mostrar lambda_meas solo si fue detectada
        lm_str = f"{lambda_meas:.2f}" if lambda_meas > 0 else "---"

        lines_out.append(col_fmt.format(
            line_name, f"{lambda_lab:.2f}", lm_str,
            f"{win_width:.1f}", f"{ew:.3f}", qflag
        ))

    # ── Resumen estadístico ──────────────────────────────────────────────
    n_ok = sum(1 for d in measurements.values()
               if isinstance(d, dict) and d.get('quality_flag') == 'OK')
    n_total = sum(1 for d in measurements.values() if isinstance(d, dict))
    n_out_of_range = sum(1 for d in measurements.values()
                         if isinstance(d, dict) and d.get('quality_flag') == 'OUT_OF_RANGE')

    lines_out.append("")
    lines_out.append(f"# Resumen: {n_ok} líneas OK / {n_total - n_out_of_range} en rango / {n_total} total")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines_out) + '\n')

    return output_path


# ============================================================================
# VERIFICACIÓN DE VELOCIDAD RADIAL (Mejora 8)
# ============================================================================

def check_radial_velocity(measurements, rv_tolerance=5.0):
    """
    Estima el desplazamiento medio de longitud de onda de las líneas detectadas.

    Si el desplazamiento es significativamente diferente de cero, emite una
    advertencia indicando que el espectro puede no estar corregido por velocidad
    radial. Se asume VR = 0 km/s a menos que el usuario indique lo contrario.

    Parameters
    ----------
    measurements : dict
        Diccionario de mediciones de measure_diagnostic_lines()
    rv_tolerance : float
        Desplazamiento máximo esperado en Å sin advertencia (default: 5.0 Å)
        Equivale a ~330 km/s a 4500 Å (amplio para cubrir estrellas rápidas)

    Returns
    -------
    mean_shift : float
        Desplazamiento medio (λ_obs − λ_lab) en Å.
        Positivo → espectro desplazado al rojo (VR > 0)
    rv_estimate : float
        Velocidad radial estimada en km/s (usando λ_ref = 4500 Å)
    rv_warning : str or None
        Mensaje de advertencia si |mean_shift| > rv_tolerance, o None.
    """
    shifts = []

    for line_name, data in measurements.items():
        if not isinstance(data, dict):
            continue
        if data.get('quality_flag') != 'OK':
            continue

        lambda_lab  = data.get('wavelength', 0.0)
        lambda_meas = data.get('lambda_measured', 0.0)

        if lambda_lab > 0 and lambda_meas > 0:
            shifts.append(lambda_meas - lambda_lab)

    if len(shifts) < 3:
        return 0.0, 0.0, None

    shifts_arr = np.array(shifts)
    mean_shift = float(np.mean(shifts_arr))
    std_shift  = float(np.std(shifts_arr))

    # Eliminación de outliers robusta (2σ) para estimación más limpia
    if std_shift > 0 and len(shifts_arr) >= 5:
        inlier = np.abs(shifts_arr - mean_shift) < 2.0 * std_shift
        if np.sum(inlier) >= 3:
            mean_shift = float(np.mean(shifts_arr[inlier]))

    # Conversión a velocidad radial: VR = c × Δλ / λ_ref
    c_kms = 299792.458      # km/s
    lambda_ref = 4500.0     # Å (longitud de onda de referencia media)
    rv_estimate = (mean_shift / lambda_ref) * c_kms

    rv_warning = None
    if abs(mean_shift) > rv_tolerance:
        rv_warning = (
            f"Desplazamiento sistemático de {mean_shift:+.2f} Å "
            f"(VR ≈ {rv_estimate:+.0f} km/s) detectado en {len(shifts)} líneas. "
            f"El espectro puede NO estar corregido por velocidad radial. "
            f"(Se asume VR = 0 km/s a menos que se especifique)"
        )

    return mean_shift, rv_estimate, rv_warning


# ============================================================================
# Programa principal
# ============================================================================

def main():
    """
    Procesa archivos FITS con el método corregido.
    """
    ######################################################################### Configuración
    input_dir = "e:/ASTROFISICA TODO/crreccion espestrocopia/elodie/TIPOO/"
    output_dir = "C:/Users/Roberto/Desktop/crreccion espestrocopia/output_corregido/"
    os.makedirs(output_dir, exist_ok=True)
    #############################################################################
    results = []

    # Procesar archivos FITS
    all_files = [f for f in os.listdir(input_dir) if f.endswith(".fits")]
    #############################################33
    selected_files = all_files[:10]  # Procesar primeros 10 para prueba 
    #############################################
    for file_name in selected_files:
        file_path = os.path.join(input_dir, file_name)

        try:
            # Procesar con método corregido
            wavelengths, flux_normalized, continuum, measurements, objeto, tipo_fits = \
                process_fits_corrected(file_path)

            # Clasificar
            spectral_type, subtype, diagnostics = classify_star_corrected(measurements)

            # Graficar
            save_path = os.path.join(output_dir, f"{file_name}.png")
            plot_spectrum_corrected(
                wavelengths, flux_normalized, measurements,
                spectral_type, subtype, objeto, tipo_fits, save_path
            )

            # Almacenar resultados
            results.append({
                'Archivo': file_name,
                'Objeto': objeto,
                'Tipo FITS': tipo_fits,
                'Tipo Clasificado': spectral_type,
                'Subtipo': subtype,
                'He II EW': diagnostics['He_II'],
                'He I EW': diagnostics['He_I'],
                'H avg EW': diagnostics['H_avg'],
                'Ca II K EW': diagnostics['Ca_II_K'],
            })

            print(f"Procesado: {file_name} -> {spectral_type} {subtype}")

        except Exception as e:
            print(f"Error procesando {file_name}: {e}")

    # Guardar resultados
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, "clasificacion_corregida.csv"), index=False)

    print(f"\nProcesamiento completado. Resultados en: {output_dir}")
    print(f"Total procesados: {len(results)}")


if __name__ == "__main__":
    main()
