# SISTEMA AUTOMATIZADO DE CLASIFICACIÓN ESPECTRAL ESTELAR
## Implementación de Métodos Computacionales para la Determinación de Tipos Espectrales

---

**Informe Técnico-Científico**

**Versión:** 3.1
**Fecha:** Marzo 2026

---

## RESUMEN

El presente trabajo describe el diseño, implementación y validación de un sistema automatizado para la clasificación espectral de estrellas basado en el análisis de líneas de absorción. El sistema implementa múltiples metodologías de clasificación: un clasificador físico basado en reglas espectroscópicas fundamentales (árbol de 26 nodos), un árbol de decisión entrenado mediante aprendizaje automático, comparación por plantillas (template matching), y clasificadores neuronales (K-Nearest Neighbors y Redes Neuronales Convolucionales). La integración de estos métodos mediante un sistema de votación ponderada permite obtener clasificaciones robustas con métricas de confianza cuantificables. El sistema ha sido validado con el catálogo ELODIE, alcanzando precisiones superiores al 84% en la clasificación de tipos espectrales principales (O, B, A, F, G, K, M). La versión 3.1 incorpora: 85 líneas diagnóstico con sistema de banderas de calidad (quality_flag), ventanas de integración adaptativas por FWHM para líneas metálicas, rechazo de rayos cósmicos en dos pasos relativo al continuo local, ajuste de continuo con spline suavizado (s < len(puntos)), archivo diagnóstico de mediciones (line_measurements.txt), verificación de velocidad radial, parámetros de SNR y resolución, y 6 razones adicionales para estrellas de tipo O-B temprano (HeI4471_HeII4541, NIII4640_HeII4686, SiIV4089_HeI4026, NV_NIII, OIII5592_HeI5876, SiIV4116_SiIII4553).

**Palabras clave:** Clasificación espectral, espectroscopía estelar, aprendizaje automático, anchos equivalentes, serie de Balmer, secuencia principal.

---

## ÍNDICE

1. [Introducción](#1-introducción)
2. [Marco Teórico](#2-marco-teórico)
3. [Metodología](#3-metodología)
4. [Implementación Computacional](#4-implementación-computacional)
5. [Validación y Resultados](#5-validación-y-resultados)
6. [Discusión](#6-discusión)
7. [Conclusiones](#7-conclusiones)
8. [Referencias](#8-referencias)
9. [Anexos](#9-anexos)

---

## 1. INTRODUCCIÓN

### 1.1 Contexto y Motivación

La clasificación espectral estelar constituye uno de los pilares fundamentales de la astrofísica moderna. Desde los trabajos pioneros de Angelo Secchi en el siglo XIX hasta el sistema de clasificación de Harvard-Draper, la categorización de estrellas según sus características espectrales ha permitido comprender la física estelar, la evolución química del universo y la estructura de nuestra galaxia.

El sistema de clasificación espectral MK (Morgan-Keenan), establecido en 1943, clasifica las estrellas en siete tipos espectrales principales: O, B, A, F, G, K, M, ordenados por temperatura efectiva decreciente. Cada tipo se subdivide en subtipos numéricos (0-9) y se complementa con clases de luminosidad (I-V) que indican el estado evolutivo de la estrella.

### 1.2 Problemática

La clasificación espectral tradicional requiere experiencia considerable y es inherentemente subjetiva. Con el advenimiento de surveys espectroscópicos masivos como SDSS, LAMOST y Gaia, que generan millones de espectros, se hace imperativa la automatización del proceso de clasificación. Sin embargo, los métodos automatizados enfrentan desafíos significativos:

1. **Variabilidad instrumental:** Diferentes espectrógrafos producen datos con distintas resoluciones y rangos espectrales.
2. **Ruido y artefactos:** Rayos cósmicos, defectos del CCD y ruido fotónico contaminan las observaciones.
3. **Efectos físicos:** Rotación estelar, velocidad radial y peculiaridades químicas modifican los perfiles espectrales.
4. **Ambigüedad intrínseca:** Los límites entre tipos espectrales son continuos, no discretos.

### 1.3 Objetivos

**Objetivo General:**
Desarrollar un sistema computacional robusto para la clasificación automatizada de espectros estelares que integre métodos físicos y de aprendizaje automático.

**Objetivos Específicos:**
1. Implementar normalización espectral al continuo con rechazo de rayos cósmicos.
2. Desarrollar medición automatizada de anchos equivalentes para líneas diagnóstico.
3. Crear un clasificador basado en criterios espectroscópicos físicos.
4. Entrenar modelos de aprendizaje automático (árboles de decisión, KNN, CNN).
5. Integrar múltiples clasificadores mediante votación ponderada.
6. Validar el sistema con catálogos de referencia.

---

## 2. MARCO TEÓRICO

### 2.1 Fundamentos de Espectroscopía Estelar

#### 2.1.1 Formación del Espectro Estelar

El espectro de una estrella resulta de la interacción entre la radiación del continuo, generada en las capas profundas de la fotosfera, y la absorción selectiva producida por átomos y iones en las capas superiores. La intensidad específica emergente I_λ se describe mediante la ecuación de transferencia radiativa:

```
dI_λ/dτ_λ = I_λ - S_λ
```

donde τ_λ es la profundidad óptica y S_λ es la función fuente. En condiciones de equilibrio termodinámico local (LTE), la función fuente equivale a la función de Planck B_λ(T).

#### 2.1.2 Perfiles de Líneas de Absorción

Las líneas de absorción presentan perfiles característicos determinados por múltiples mecanismos de ensanchamiento:

1. **Ensanchamiento natural:** Debido a la incertidumbre cuántica en la energía de los niveles (perfil Lorentziano).

2. **Ensanchamiento Doppler térmico:** Por el movimiento térmico de los átomos absorbentes:
   ```
   Δλ_D = (λ_0/c) * sqrt(2kT/m)
   ```

3. **Ensanchamiento por presión (Stark, Van der Waals):** Por perturbaciones de partículas vecinas.

4. **Ensanchamiento rotacional:** En estrellas de rotación rápida, el desplazamiento Doppler diferencial ensancha las líneas.

El perfil resultante es una convolución de estos efectos, típicamente aproximado por un perfil de Voigt.

#### 2.1.3 Ancho Equivalente

El ancho equivalente (EW) es una medida de la intensidad integrada de una línea de absorción, definido como:

```
EW = ∫ (1 - F_λ/F_c) dλ
```

donde F_λ es el flujo observado y F_c es el flujo del continuo. El EW tiene unidades de longitud (típicamente Angstroms) y representa el ancho de un rectángulo de altura igual al continuo que tiene la misma área que la línea de absorción.

**Importancia del EW:**
- Es independiente de la resolución espectral (a diferencia del FWHM).
- Mide la intensidad total de la línea, relacionada con la abundancia del elemento y las condiciones físicas.
- Es el parámetro fundamental para la clasificación espectral cuantitativa.

### 2.2 Sistema de Clasificación Espectral MK

#### 2.2.1 Secuencia de Tipos Espectrales

| Tipo | T_eff (K) | Color | Características Principales |
|------|-----------|-------|----------------------------|
| O | >30,000 | Azul | He II fuerte, He I presente, H débil |
| B | 10,000-30,000 | Azul-blanco | He I máximo en B2, H creciente |
| A | 7,500-10,000 | Blanco | H máximo (Balmer), metales débiles |
| F | 6,000-7,500 | Amarillo-blanco | H decreciente, Ca II K creciente |
| G | 5,200-6,000 | Amarillo | Ca II H y K fuertes, banda G (CH) |
| K | 3,700-5,200 | Naranja | Metales dominantes, TiO incipiente |
| M | <3,700 | Rojo | Bandas moleculares TiO, VO |

#### 2.2.2 Criterios de Clasificación

**Tipo O:**
- Presencia de He II (λ4686, λ4541) indica T > 30,000 K
- Ratio He II/He I discrimina subtipos O3-O9
- Líneas de N III, C III en emisión (estrellas Of)

**Tipo B:**
- He I presente (λ4471, λ4026, λ4922)
- Ausencia de He II (excepto B0-B0.5)
- Serie de Balmer creciente hacia B tardío
- Líneas de Si (Si IV → Si III → Si II) para subtipos

**Tipo A:**
- Máximo de líneas de Balmer (Hβ, Hγ, Hδ hasta ~12 Å)
- Ca II K débil pero creciente
- Aparición de líneas metálicas (Fe I, Mg II)

**Tipo F:**
- Balmer decreciente, Ca II K prominente
- Banda G (CH, λ4300) visible
- Líneas metálicas (Fe I, Ca I) fortaleciendo

**Tipos G, K, M:**
- Dominio de líneas metálicas
- Ca II H y K muy intensos
- Bandas moleculares en K tardío y M (TiO, VO, H2O)

#### 2.2.3 Clases de Luminosidad

| Clase | Denominación | Características |
|-------|--------------|-----------------|
| Ia, Ib | Supergigantes | Líneas estrechas, ratios de ionización alterados |
| II | Gigantes brillantes | Intermedio |
| III | Gigantes | Líneas moderadamente estrechas |
| IV | Subgigantes | Intermedio |
| V | Enanas (secuencia principal) | Líneas más anchas (mayor presión) |

### 2.3 Líneas Espectrales Diagnóstico

#### 2.3.1 Serie de Balmer del Hidrógeno

| Línea | λ (Å) | Transición | Uso Diagnóstico |
|-------|-------|------------|-----------------|
| Hα | 6562.8 | 3→2 | Cromosfera, emisión en Be |
| Hβ | 4861.3 | 4→2 | Clasificación primaria A-F |
| Hγ | 4340.5 | 5→2 | Clasificación, luminosidad |
| Hδ | 4101.7 | 6→2 | Clasificación |
| Hε | 3970.1 | 7→2 | Contaminación con Ca II H |

#### 2.3.2 Líneas de Helio

| Línea | λ (Å) | Especie | Rango Espectral |
|-------|-------|---------|-----------------|
| He II 4686 | 4685.7 | He+ | O (T > 30,000 K) |
| He II 4541 | 4541.6 | He+ | O |
| He I 4471 | 4471.5 | He | O tardío - B |
| He I 4026 | 4026.2 | He | B |
| He I 4922 | 4921.9 | He | B |

#### 2.3.3 Líneas Metálicas

| Línea | λ (Å) | Uso |
|-------|-------|-----|
| Ca II K | 3933.7 | F-G-K diagnóstico primario |
| Ca II H | 3968.5 | F-G-K (contaminado con Hε) |
| Ca I 4227 | 4226.7 | G-K-M |
| Mg II 4481 | 4481.2 | B tardío - A |
| Fe I 4046 | 4045.8 | F-G-K-M |
| Fe I 4383 | 4383.5 | G-K-M |

#### 2.3.4 Líneas de Silicio (Subtipos B)

| Línea | λ (Å) | Subtipos |
|-------|-------|----------|
| Si IV 4089 | 4088.9 | B0-B1 |
| Si III 4553 | 4552.6 | B1-B3 |
| Si II 4128 | 4128.1 | B5-B9 |

---

## 3. METODOLOGÍA

### 3.1 Preprocesamiento Espectral

#### 3.1.1 Rechazo de Rayos Cósmicos

Los rayos cósmicos producen picos agudos en los espectros CCD que pueden contaminar severamente la normalización y medición de líneas. Se implementa un algoritmo de sigma-clipping iterativo:

```python
def sigma_clip(flux, sigma=3.0, max_iter=5):
    """
    Rechaza outliers mediante sigma-clipping iterativo.

    Algoritmo:
    1. Calcular mediana y desviación estándar robusta (MAD)
    2. Identificar puntos que excedan sigma * MAD
    3. Reemplazar outliers por interpolación local
    4. Iterar hasta convergencia
    """
    for iteration in range(max_iter):
        median = np.median(flux)
        mad = np.median(np.abs(flux - median))
        std_robust = 1.4826 * mad  # Factor para distribución normal

        mask = np.abs(flux - median) > sigma * std_robust
        if not np.any(mask):
            break

        # Interpolación lineal para outliers
        flux[mask] = np.interp(
            np.where(mask)[0],
            np.where(~mask)[0],
            flux[~mask]
        )

    return flux
```

#### 3.1.2 Normalización al Continuo

La normalización correcta es **crítica** para la medición de anchos equivalentes. Se implementa ajuste del continuo mediante:

1. **Selección de ventanas de continuo:** Regiones libres de líneas de absorción.
2. **Ajuste polinomial/spline:** Interpolación del continuo entre ventanas.
3. **División:** F_norm = F_obs / F_continuo

```python
def normalize_to_continuum(wavelengths, flux, poly_order=5):
    """
    Normaliza el espectro al continuo = 1.0

    Método:
    1. Sigma-clip para rechazar líneas y rayos cósmicos
    2. Seleccionar puntos del continuo (percentil alto)
    3. Ajustar spline cúbico
    4. Dividir espectro por continuo ajustado
    """
    # Rechazar outliers
    flux_clean = sigma_clip(flux.copy())

    # Seleccionar puntos de continuo (percentil 85-95)
    rolling_max = scipy.ndimage.maximum_filter1d(flux_clean, size=50)
    continuum_points = flux_clean > 0.9 * rolling_max

    # Ajuste spline
    spline = scipy.interpolate.UnivariateSpline(
        wavelengths[continuum_points],
        flux_clean[continuum_points],
        s=len(wavelengths) * 0.01
    )
    continuum = spline(wavelengths)

    # Normalizar
    flux_normalized = flux / continuum

    return flux_normalized, continuum
```

**Nota sobre errores comunes:**

La normalización `(flux - min) / (max - min)` es **incorrecta** para espectroscopía porque:
- El mínimo se convierte en 0, perdiendo información del perfil
- Un rayo cósmico en el máximo distorsiona todo el espectro
- Impide el cálculo de anchos equivalentes (que requieren continuo = 1)

### 3.2 Medición de Anchos Equivalentes

#### 3.2.1 Algoritmo de Medición

```python
def measure_equivalent_width(wavelengths, flux_norm, line_center,
                             window=10.0, tolerance=2.0):
    """
    Mide el ancho equivalente de una línea de absorción.

    Parámetros:
        wavelengths: Array de longitudes de onda
        flux_norm: Flujo normalizado (continuo = 1.0)
        line_center: Longitud de onda central teórica
        window: Semi-ancho de ventana de integración (Å)
        tolerance: Tolerancia para identificación de línea (Å)

    Retorna:
        ew: Ancho equivalente (Å)
        detected: Booleano indicando detección válida
        diagnostics: Diccionario con información adicional
    """
    # 1. Extraer región de la línea
    mask = (wavelengths > line_center - window) & \
           (wavelengths < line_center + window)

    if np.sum(mask) < 10:
        return 0.0, False, {'error': 'Insufficient data'}

    wave_region = wavelengths[mask]
    flux_region = flux_norm[mask]

    # 2. Estimar continuo local
    continuum_local = np.percentile(flux_region, 90)

    # 3. Detectar mínimo de absorción
    min_idx = np.argmin(flux_region)
    min_wave = wave_region[min_idx]
    min_flux = flux_region[min_idx]

    # 4. Verificar posición (dentro de tolerancia)
    if abs(min_wave - line_center) > tolerance:
        return 0.0, False, {'error': 'Line not at expected position'}

    # 5. Verificar profundidad mínima (3%)
    depth = 1.0 - min_flux / continuum_local
    if depth < 0.03:
        return 0.0, False, {'error': 'Line too shallow'}

    # 6. Calcular EW por integración trapezoidal
    # EW = ∫(1 - F/Fc)dλ
    integrand = 1.0 - flux_region / continuum_local
    ew = np.trapz(integrand, wave_region)

    # Solo contar absorción (EW > 0 para absorción)
    if ew < 0:
        ew = 0.0

    return ew, True, {
        'depth': depth,
        'center_observed': min_wave,
        'continuum_local': continuum_local
    }
```

#### 3.2.2 Validación de Detecciones

Se implementan cinco criterios de validación:

1. **Datos suficientes:** Mínimo 10 puntos en la ventana
2. **Continuo local robusto:** Estimación por percentil alto
3. **Posición verificada:** Centro observado dentro de ±2 Å del teórico
4. **Profundidad mínima:** Al menos 3% bajo el continuo
5. **Detección de blends:** Análisis de asimetría del perfil

### 3.3 Clasificador Físico

#### 3.3.1 Árbol de Decisión Espectroscópico

El clasificador físico implementa un árbol de decisión basado en criterios espectroscópicos fundamentales:

```
                    ┌─────────────────┐
                    │ He II > 0.30 Å? │
                    └────────┬────────┘
                      Sí     │      No
                    ┌────────┴────────┐
                    │                 │
                 Tipo O          ┌────┴────┐
                             │He I > 0.15Å│
                             └────┬───────┘
                               Sí │     No
                             ┌────┴────┐
                             │         │
                          Tipo B   ┌───┴───┐
                                   │H > 4Å?│
                                   └───┬───┘
                                    Sí │    No
                                  ┌────┴────┐
                                  │         │
                           ┌──────┴──┐  ┌───┴────┐
                           │H > 8Å?  │  │Ca K>2Å?│
                           └────┬────┘  └───┬────┘
                             Sí │ No      Sí │  No
                               A   F       G-K   M
```

#### 3.3.2 Criterios Detallados por Tipo

**Tipo O (T_eff > 30,000 K):**
```python
if He_II_4686 > 0.30:  # Å
    tipo = 'O'
    # Subtipos por ratio He II/He I
    ratio = He_II_4686 / (He_I_4471 + 0.01)
    if ratio > 3.0:
        subtipo = 'O3-O5'
    elif ratio > 1.5:
        subtipo = 'O6-O7'
    else:
        subtipo = 'O8-O9'
```

**Tipo B (T_eff 10,000-30,000 K):**
```python
if He_I_4471 > 0.15 and He_II_4686 < 0.30:
    tipo = 'B'
    # Subtipos por líneas de Si
    if Si_IV_4089 > 0.10:
        subtipo = 'B0-B1'
    elif Si_III_4553 / (Si_II_4128 + 0.01) > 1.2:
        subtipo = 'B2-B3'
    else:
        subtipo = 'B5-B9'
```

**Validación cruzada con Balmer:**
```python
# Evitar confusión B/A para estrellas A tempranas
H_avg = (H_beta + H_gamma + H_delta) / 3.0
if H_avg > 4.0:  # Balmer muy fuerte
    ratio_He_H = He_I_4471 / (H_avg + 0.01)
    if ratio_He_H < 0.15:  # He I < 15% de H
        # Probablemente A0, no B tardío
        He_I_detectado = False
```

### 3.4 Clasificadores de Aprendizaje Automático

#### 3.4.1 Árbol de Decisión

Se entrena un clasificador RandomForest con las siguientes características (features):

**Features de entrada (24 dimensiones):**
```python
features = [
    # Anchos equivalentes (17)
    'He_II_4686', 'He_I_4471', 'H_beta', 'H_gamma', 'H_delta',
    'H_epsilon', 'Si_IV_4089', 'Si_III_4553', 'Si_II_4128',
    'Mg_II_4481', 'Ca_II_K', 'Ca_II_H', 'Ca_I_4227',
    'Fe_I_4046', 'Fe_I_4144', 'Fe_I_4383', 'Fe_I_4957',

    # Ratios diagnósticos (6)
    'ratio_He_I_He_II',      # He I / He II
    'ratio_Si_III_Si_II',    # Si III / Si II
    'ratio_Ca_K_H_epsilon',  # Ca II K / Hε
    'ratio_Mg_He_I',         # Mg II / He I
    'ratio_Fe_H',            # Fe I avg / H avg
    'ratio_Ca_I_H_gamma',    # Ca I / Hγ

    # Feature derivada (1)
    'H_avg'                  # Promedio Balmer
]
```

**Hiperparámetros optimizados:**
```python
classifier = RandomForestClassifier(
    n_estimators=100,      # Número de árboles
    max_depth=9,           # Profundidad máxima
    min_samples_split=5,   # Mínimo para dividir nodo
    min_samples_leaf=2,    # Mínimo en hojas
    class_weight='balanced' # Balanceo de clases
)
```

#### 3.4.2 K-Nearest Neighbors (KNN)

El clasificador KNN opera en el espacio de features normalizadas:

```python
# Preprocesamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Modelo
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',    # Ponderación por distancia
    metric='euclidean'     # Métrica de distancia
)
```

**Ventajas:**
- No paramétrico, adaptable a distribuciones complejas
- Interpretable (los k vecinos más cercanos)
- Robusto a outliers con ponderación por distancia

#### 3.4.3 Red Neuronal Convolucional (CNN 1D)

Para clasificación directa desde el espectro (sin extracción manual de features):

```python
model = Sequential([
    # Capa convolucional 1
    Conv1D(32, kernel_size=5, activation='relu',
           input_shape=(spectrum_length, 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # Capa convolucional 2
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # Capas densas
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

**Arquitectura justificada:**
- Capas convolucionales detectan patrones locales (líneas de absorción)
- Pooling reduce dimensionalidad preservando features importantes
- Dropout previene sobreajuste
- Softmax proporciona probabilidades por clase

### 3.5 Sistema de Votación Multi-Método

#### 3.5.1 Votación Ponderada

Los cuatro clasificadores se combinan mediante votación ponderada:

```python
# Pesos por defecto (ajustados empíricamente)
weights = {
    'physical': 0.10,        # Clasificador físico
    'decision_tree': 0.40,   # Árbol de decisión (ML)
    'template_matching': 0.10, # Comparación con plantillas
    'neural': 0.40           # KNN o CNN
}

# Votación
votes = defaultdict(float)
for method, result in results.items():
    tipo = result['tipo']
    confianza = result['confianza'] / 100.0
    peso = weights[method]
    votes[tipo] += peso * confianza

# Tipo final
tipo_final = max(votes, key=votes.get)
```

#### 3.5.2 Cálculo de Confianza Global

```python
def calculate_global_confidence(votes, results):
    """
    Calcula confianza basada en:
    1. Consenso entre métodos
    2. Confianza individual de cada método
    3. Margen sobre la segunda alternativa
    """
    sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)

    top_vote = sorted_votes[0][1]
    second_vote = sorted_votes[1][1] if len(sorted_votes) > 1 else 0

    # Factor de consenso
    ratio = top_vote / (second_vote + 0.01)

    if ratio > 2.0:  # Consenso fuerte
        confidence = min(95.0, base_confidence * 1.1)
    else:
        confidence = base_confidence * 0.85

    return confidence
```

### 3.6 Template Matching

#### 3.6.1 Plantillas de Referencia

Se definen plantillas con valores típicos de EW para cada tipo espectral:

```python
templates = {
    'O': {
        'He_II_4686': 1.0, 'He_I_4471': 0.5, 'H_beta': 2.0,
        'Si_IV_4089': 0.3, 'Ca_II_K': 0.0
    },
    'B': {
        'He_II_4686': 0.0, 'He_I_4471': 1.0, 'H_beta': 3.0,
        'Mg_II_4481': 0.5, 'Ca_II_K': 0.1
    },
    'A': {
        'He_I_4471': 0.0, 'H_beta': 9.0, 'H_gamma': 8.0,
        'Ca_II_K': 2.0
    },
    # ... etc.
}
```

#### 3.6.2 Métrica Chi-Cuadrado

```python
def calculate_chi2(measurements, template):
    """
    Chi-cuadrado reducido para comparación con plantilla.
    """
    chi2 = 0.0
    n_lines = 0

    for line, template_ew in template.items():
        measured_ew = measurements.get(line, {}).get('ew', 0.0)
        error = max(template_ew * 0.1, 0.05)  # 10% o 0.05 Å mínimo

        chi2 += ((measured_ew - template_ew) / error) ** 2
        n_lines += 1

    return chi2 / n_lines  # Reducido
```

---

## 4. IMPLEMENTACIÓN COMPUTACIONAL

### 4.1 Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTERFAZ WEB (Flask)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Análisis │  │  Lotes   │  │Herramien.│  │Redes Neural. │   │
│  │ Espectro │  │          │  │          │  │              │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘   │
└───────┼─────────────┼─────────────┼───────────────┼───────────┘
        │             │             │               │
        └─────────────┴─────────────┴───────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  SpectralValidator │
                    │  (Votación Multi- │
                    │      Método)       │
                    └─────────┬─────────┘
                              │
        ┌───────────┬─────────┼─────────┬───────────┐
        │           │         │         │           │
   ┌────┴────┐ ┌────┴────┐ ┌──┴───┐ ┌───┴───────┐ │
   │ Físico  │ │ Árbol   │ │Templa│ │  Neural   │ │
   │         │ │Decisión │ │  te  │ │ (KNN/CNN) │ │
   └────┬────┘ └────┬────┘ └──┬───┘ └───┬───────┘ │
        │           │         │         │          │
        └───────────┴─────────┴─────────┴──────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Preprocesamiento │
                    │  - Normalización  │
                    │  - Medición EW    │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Carga de Datos   │
                    │  (.txt, .fits)    │
                    └───────────────────┘
```

### 4.2 Módulos Principales

#### 4.2.1 `spectral_classification_corrected.py`

Módulo central que implementa:

- `normalize_to_continuum()`: Normalización espectral al continuo (spline cúbico)
- `measure_equivalent_width()`: Medición de EW con 5 validaciones
- `LINE_WINDOWS`: Diccionario de ventanas de integración por línea (6–30 Å)
- `measure_diagnostic_lines()`: Medición de **85 líneas** diagnóstico
- `compute_spectral_ratios()`: Calcula **21 razones** diagnóstico
- `classify_star_corrected()`: Clasificador físico con árbol de 26 nodos
- `detect_FeI_4957_asymmetry()`: Detección de asimetría por TiO en tipos M
- `evaluate_tio_bands()`: Evaluación de profundidad de bandas TiO
- `plot_spectrum_corrected()`: Visualización con líneas diagnóstico

#### 4.2.2 `spectral_validation.py`

Sistema de votación multi-método:

- `SpectralValidator`: Clase principal
- `DecisionTreeClassifier`: Wrapper para sklearn
- `TemplateMatchingClassifier`: Comparación con plantillas
- Integración con clasificadores neuronales

#### 4.2.3 `neural_classifiers.py`

Clasificadores de aprendizaje profundo:

- `KNNClassifier`: K-Nearest Neighbors
- `CNNClassifier`: Red Convolucional (TensorFlow)
- `NeuralClassifierManager`: Gestión de modelos

#### 4.2.4 `webapp/app.py`

Servidor web Flask con endpoints:

- `POST /upload`: Análisis de espectro individual (devuelve tipo + luminosidad + ratios)
- `POST /api/luminosity`: Estimación de clase de luminosidad MK independiente
- `POST /batch_process`: Procesamiento por lotes
- `POST /train_neural`: Entrenamiento de modelos
- `GET /neural_metrics`: Métricas de modelos
- `GET /health`: Estado del servidor (reporta `lines_configured: 85`)

#### 4.2.5 `src/luminosity_classification.py`

Módulo de clasificación de clase de luminosidad MK (I–V):

- `estimate_luminosity_class(measurements, spectral_type)`: Estima clase (I–V) mediante
  indicadores sensibles a gravedad superficial (Sr II, Ba II, CaI/FeI, TiO/CaH)
- `combine_spectral_and_luminosity(tipo, subtipo, lum_class)`: Construye el tipo MK completo
  (ej. `"G2V"`, `"K3III"`)
- Los resultados se almacenan en `diagnostics['luminosity_class']` y `diagnostics['mk_full']`
  durante la clasificación física principal

### 4.3 Base de Datos de Líneas Espectrales (85 líneas)

```python
SPECTRAL_LINES = {
    # Serie de Balmer (5 líneas)
    'H_alpha': 6562.8,  'H_beta': 4861.3,  'H_gamma': 4340.5,
    'H_delta': 4101.7,  'H_epsilon': 3970.1,

    # Helio (He I: 6 líneas; He II: 4 líneas)
    'He_II_4686': 4685.7,  'He_II_4541': 4541.6,
    'He_II_4200': 4199.8,  'He_II_5411': 5411.5,
    'He_I_4471': 4471.5,   'He_I_4026': 4026.2,
    'He_I_4388': 4387.9,   'He_I_4922': 4921.9,
    'He_I_5016': 5015.7,   'He_I_5876': 5875.7,

    # N, C, O ionizados (11 líneas) — tipos O, B
    'N_V_4604': 4603.7,    'N_V_4620': 4619.9,
    'N_IV_4058': 4057.8,   'N_III_4634': 4634.1,   'N_III_4641': 4640.6,
    'C_III_4650': 4647.4,  'C_IV_4658': 4658.3,    'C_II_4267': 4267.0,
    'O_II_4415': 4414.9,   'O_II_4417': 4416.9,
    'O_II_4591': 4590.9,   'O_II_4596': 4595.9,    'O_V_5114': 5114.1,

    # Silicio (8 líneas) — diagnóstico subtipos B
    'Si_IV_4089': 4088.9,  'Si_IV_4116': 4116.1,
    'Si_III_4553': 4552.6, 'Si_III_4568': 4567.8,  'Si_III_4575': 4574.8,
    'Si_II_4128': 4128.1,  'Si_II_4131': 4130.9,
    'Si_II_5041': 5041.1,  'Si_II_5056': 5056.3,

    # Calcio (4 líneas)
    'Ca_II_K': 3933.7,  'Ca_II_H': 3968.5,
    'Ca_I_4227': 4226.7, 'Ca_I_4455': 4454.8,
    'Ca_I_6122': 6122.2, 'Ca_I_6162': 6162.2,

    # Magnesio (3 líneas)
    'Mg_II_4481': 4481.2,
    'Mg_I_4703': 4702.9,  'Mg_I_5183': 5183.6,

    # Hierro (11 líneas)
    'Fe_I_4046': 4045.8,  'Fe_I_4071': 4071.7,  'Fe_I_4144': 4143.9,
    'Fe_I_4260': 4260.5,  'Fe_I_4271': 4271.2,  'Fe_I_4326': 4325.8,
    'Fe_I_4383': 4383.5,  'Fe_I_4957': 4957.6,
    'Fe_I_5270': 5269.5,  'Fe_I_5328': 5328.0,  'Fe_I_6495': 6495.0,
    'Fe_II_4173': 4173.5, 'Fe_II_4233': 4233.2,

    # Cromo, estroncio, bario, sodio
    'Cr_I_4254': 4254.3,
    'Sr_II_4077': 4077.7, 'Sr_II_4216': 4215.5,
    'Ba_II_4554': 4554.0,
    'Na_I_D': 5895.9,

    # Titanio, aluminio
    'Ti_II_4468': 4467.8,  'Al_III_4480': 4479.9,

    # CH G-band
    'CH_Gband': 4300.0,

    # Bandas TiO clásicas (3) + extendidas (3)
    'TiO_4762': 4761.0,  'TiO_4955': 4955.0,  'TiO_5167': 5167.0,
    'TiO_5448': 5448.0,  'TiO_6158': 6158.0,  'TiO_6651': 6651.0,

    # Bandas VO y CaH (tipos M tardíos)
    'VO_7434': 7434.0,   'VO_7865': 7865.0,
    'CaH_6382': 6382.0,  'CaH_6750': 6750.0,
    # ... 85 líneas en total
}
```

---

## 5. VALIDACIÓN Y RESULTADOS

### 5.1 Conjunto de Datos

#### 5.1.1 Catálogo ELODIE

Se utilizó el catálogo ELODIE del Observatorio de Haute-Provence:

| Parámetro | Valor |
|-----------|-------|
| Total de espectros | 856 |
| Rango espectral | 3900-6800 Å |
| Resolución | R ~ 42,000 |
| S/N típico | > 100 |

**Distribución por tipo espectral:**

| Tipo | Cantidad | Porcentaje |
|------|----------|------------|
| O | 15 | 1.8% |
| B | 64 | 7.5% |
| A | 139 | 16.2% |
| F | 352 | 41.1% |
| G | 1 | 0.1% |
| K | 257 | 30.0% |
| M | 27 | 3.2% |

#### 5.1.2 División Train/Test

- **Entrenamiento:** 80% (684 espectros)
- **Test:** 20% (171 espectros)
- **Estratificación:** Proporcional por tipo
- **Validación cruzada:** 5-fold

### 5.2 Métricas de Rendimiento

#### 5.2.1 Árbol de Decisión

```
Accuracy global: 84.2%
Accuracy por validación cruzada: 82.5% ± 3.1%

              precision    recall  f1-score   support
           A       0.82      0.86      0.84        28
           B       0.75      0.69      0.72        13
           F       0.89      0.90      0.89        70
           K       0.82      0.80      0.81        51
           M       0.83      0.91      0.87         6
           O       1.00      0.67      0.80         3

    accuracy                           0.84       171
   macro avg       0.85      0.81      0.82       171
weighted avg       0.84      0.84      0.84       171
```

#### 5.2.2 KNN (k=5)

```
Accuracy global: 81.3%
Accuracy por validación cruzada: 79.8% ± 2.8%
```

#### 5.2.3 Sistema Multi-Método (Votación)

```
Accuracy global: 86.5%
Mejora sobre mejor método individual: +2.3%
```

### 5.3 Matriz de Confusión

```
Predicho →    O    B    A    F    G    K    M
Real ↓
    O         2    1    0    0    0    0    0
    B         0    9    3    1    0    0    0
    A         0    2   24    2    0    0    0
    F         0    0    2   63    0    5    0
    K         0    0    0    5    0   46    0
    M         0    0    0    0    0    1    5
```

**Observaciones:**
- Confusión B↔A: Esperada por gradiente continuo en transición
- Confusión F↔K: Debido a solapamiento de características metálicas
- Tipo O: Bajo recall por muestra pequeña (3 espectros test)

### 5.4 Casos de Estudio

#### 5.4.1 Caso 1: HD138749 (B6Vnn)

**Problema inicial:** Clasificado erróneamente como B0.5

**Diagnóstico:**
- He I 4471: 0.18 Å (débil)
- Detección de "línea" en posición incorrecta

**Solución implementada:**
- Validación de posición (±2 Å del centro teórico)
- Verificación de profundidad mínima (3%)

**Resultado:** Clasificación correcta como B6V

#### 5.4.2 Caso 2: HD071557 (A0)

**Problema inicial:** Clasificado como B6-B7

**Diagnóstico:**
- H promedio: 5.2 Å (muy fuerte para B)
- He I débil pero detectado (ruido)

**Solución implementada:**
- Validación cruzada con Balmer
- Si H_avg > 4.0 Å y He_I/H < 0.15, descartar He I

**Resultado:** Clasificación correcta como A0

### 5.5 Análisis de Errores

#### 5.5.1 Fuentes de Error Sistemático

1. **Rango espectral limitado:**
   - Ca II K (3933 Å) fuera de rango en algunos espectros
   - Impacta clasificación F-G-K

2. **Desbalance de clases:**
   - Tipo O: solo 15 muestras (1.8%)
   - Tipo G: solo 1 muestra (0.1%)

3. **Velocidad radial no corregida:**
   - Desplazamiento de líneas hasta ±2 Å
   - Tolerancia de identificación debe ser ≥2 Å

#### 5.5.2 Incertidumbre en Mediciones

```
Error típico en EW: σ_EW ≈ 0.05-0.1 Å
Fuentes:
- Ruido fotónico: ~0.03 Å
- Continuo local: ~0.05 Å
- Blends no resueltos: ~0.02-0.1 Å
```

---

## 6. DISCUSIÓN

### 6.1 Comparación con Trabajos Previos

| Sistema | Accuracy | Método | Referencia |
|---------|----------|--------|------------|
| Este trabajo | 86.5% | Multi-método | - |
| MKCLASS | ~85% | Reglas | Gray & Corbally (2014) |
| LAMOST Pipeline | 83% | Template | Luo et al. (2015) |
| StarNet | 89% | CNN | Fabbro et al. (2018) |

### 6.2 Ventajas del Enfoque Multi-Método

1. **Robustez:** Errores de un método compensados por otros
2. **Interpretabilidad:** Clasificador físico proporciona justificación
3. **Flexibilidad:** Pesos ajustables según calidad de datos
4. **Cuantificación de incertidumbre:** Consenso indica confianza

### 6.3 Limitaciones

1. **Dependencia del rango espectral:** Requiere 3900–5000 Å mínimo; luminosidad M requiere hasta 7900 Å
2. **Estrellas peculiares:** No maneja Am, Ap, Be, emisión
3. **Clase de luminosidad:** Implementada con indicadores de razones; estimación aproximada (±1 subclase)
4. **Metalicidad:** No considera [Fe/H] explícitamente

### 6.4 Mejoras Futuras

1. **Corrección de velocidad radial automática**
2. **Detección de estrellas peculiares**
3. **Estimación de parámetros físicos (T_eff, log g, [Fe/H])**
4. **Extensión a infrarrojo cercano**
5. **Incertidumbres bayesianas**

---

## 7. CONCLUSIONES

Se ha desarrollado un sistema automatizado de clasificación espectral estelar que integra múltiples metodologías computacionales. Las principales contribuciones son:

1. **Implementación correcta de normalización espectral** con rechazo de rayos cósmicos y ajuste de continuo, fundamental para la medición precisa de anchos equivalentes.

2. **Medición automatizada de 85 líneas diagnóstico** con ventanas de integración adaptativas por línea (LINE_WINDOWS, 6–30 Å), validación de posición, profundidad y detección de blends.

3. **21 razones diagnóstico** (compute_spectral_ratios) que cubren toda la secuencia OBAFGKM, incluyendo indicadores de ionización (He I/He II, Si III/Si II), de temperatura (CaII K/Hε, CrI/FeI) y de luminosidad (Sr II/Fe I, Ba II/Fe I, TiO/CaH).

4. **Clasificador físico basado en 26 nodos** que proporciona clasificaciones interpretables y fundamentadas en la física de atmósferas estelares, con 8 nodos nuevos para tipos O tempranos, B con líneas de O II, F/G con banda CH y Mg b, K con Na I, y M tardías con VO y CaH.

5. **Módulo de clasificación de luminosidad MK** (luminosity_classification.py) que produce tipos completos (ej. G2V, K3III) visibles en ambas pestañas de la interfaz web.

6. **Integración de aprendizaje automático** (árboles de decisión, KNN, CNN) que mejora la precisión mediante el aprendizaje de patrones en grandes conjuntos de datos.

7. **Sistema de votación ponderada** que combina las fortalezas de cada método, alcanzando una precisión del 86.5% en el catálogo ELODIE.

8. **Interfaz web interactiva** que facilita el uso del sistema sin requerir conocimientos de programación, con árbol interactivo de 26 nodos, exportación a CSV/TXT/PDF, y visualización de luminosidad en tiempo real.

El sistema representa una herramienta útil para la clasificación automatizada de espectros estelares, con aplicaciones en surveys espectroscópicos, caracterización de candidatos exoplanetarios, y estudios de poblaciones estelares.

---

## 8. REFERENCIAS

1. Bailer-Jones, C. A. L. (2002). "Automated stellar classification for large surveys: a review." *Astronomy & Astrophysics*, 379(3), 1046-1059.

2. Fabbro, S., et al. (2018). "An application of deep learning in the analysis of stellar spectra." *Monthly Notices of the Royal Astronomical Society*, 475(3), 2978-2993.

3. Gray, R. O., & Corbally, C. J. (2009). *Stellar Spectral Classification*. Princeton University Press.

4. Gray, R. O., & Corbally, C. J. (2014). "An expert computer program for classifying stars on the MK spectral classification system." *The Astronomical Journal*, 147(4), 80.

5. Kurucz, R. L. (1992). "Atomic and molecular data for opacity calculations." *Revista Mexicana de Astronomía y Astrofísica*, 23, 45.

6. Morgan, W. W., Keenan, P. C., & Kellman, E. (1943). *An Atlas of Stellar Spectra*. University of Chicago Press.

7. Moultaka, J., et al. (2004). "The ELODIE archive." *Publications of the Astronomical Society of the Pacific*, 116(821), 693.

8. Sánchez-Blázquez, P., et al. (2006). "Medium-resolution Isaac Newton Telescope library of empirical spectra." *Monthly Notices of the Royal Astronomical Society*, 371(2), 703-718.

9. Kirkpatrick, J. D., Reid, I. N., & Liebert, J. (1999). "Dwarfs Cooler than M: The Definition of Spectral Type L Using Discoveries from the 2 Micron All-Sky Survey (2MASS)." *The Astrophysical Journal*, 519(2), 802-833.

10. Sota, A., Maíz Apellániz, J., Morrell, N. I., et al. (2014). "The Galactic O-Star Spectroscopic Survey (GOSSS). II. Bright Southern Stars." *The Astrophysical Journal Supplement Series*, 211(1), 10.

---

## 9. ANEXOS

### Anexo A: Estructura de Archivos del Sistema

```
crreccion espestrocopiv3 -final/
├── src/
│   ├── spectral_classification_corrected.py  # Motor: 85 líneas, LINE_WINDOWS, 21 ratios
│   ├── luminosity_classification.py          # Clasificación de luminosidad MK
│   ├── spectral_validation.py                # Votación multi-método
│   ├── neural_classifiers.py                 # KNN/CNN
│   └── train_neural_models.py                # Entrenamiento
│
├── webapp/
│   ├── app.py                                # Servidor Flask + /api/luminosity
│   ├── templates/
│   │   └── index.html                        # Interfaz: Clasificación, Árbol (26 nodos), Ayuda
│   └── static/
│       ├── script.js                         # Árbol 26 nodos + UI luminosidad
│       └── style.css                         # Estilos badges MK, luminosidad
│
├── models/                                   # Modelos entrenados
│   ├── decision_tree.pkl
│   ├── knn_model.pkl
│   └── cnn_model.h5
│
├── data/
│   ├── elodie/                               # Catálogo de entrenamiento (856 espectros)
│   └── espectros/                            # Espectros de referencia
│
├── docs/                                     # Documentación técnica
├── requirements.txt                          # Dependencias básicas
├── instalar.bat                              # Instalación Windows
└── iniciar.bat                               # Inicio rápido
```

### Anexo B: Formato de Archivos de Entrada

**Archivo .txt (dos columnas, CSV):**
```
Longitud de onda (Å),espectro
4000.0,0.9829112
4000.05,0.83460677
4000.10,1.1787636
...
```

**Archivo .fits:**
- Header WCS: CRVAL1, CDELT1, CRPIX1
- Extensión de datos con flujo

### Anexo C: API de la Aplicación Web

**POST /upload**
```json
// Request: multipart/form-data con campo "files[]"

// Response
{
    "success": true,
    "tipo_final": "G",
    "subtipo_final": "G2",
    "confianza": 87.5,
    "alternativas": [...],
    "detalles": {
        "luminosity_class": "V",
        "mk_full": "G2V",
        "ratios": { "CrI_FeI": 0.85, "CaI_FeI": 0.79, ... }
    },
    "plot_url": "/plot/spectrum.png"
}
```

**POST /api/luminosity**
```json
// Request
{
    "wavelength": [4000.0, 4000.5, ...],
    "flux": [0.98, 0.97, ...],
    "spectral_type": "G2"
}

// Response
{
    "success": true,
    "luminosity_class": "V",
    "mk_full": "G2V",
    "lum_name": "Enana (secuencia principal)",
    "indicators": { "CaI_FeI": 0.79, "BaII_FeI": 0.23 }
}
```

**POST /train_neural**
```json
// Request
{
    "model_type": "knn",
    "catalog_path": "data/elodie/",
    "test_size": 0.2,
    "n_neighbors": 5
}

// Response
{
    "success": true,
    "accuracy": 81.3,
    "n_samples": 856,
    "classes": ["A", "B", "F", "K", "M", "O"]
}
```

### Anexo D: Glosario

| Término | Definición |
|---------|------------|
| **Ancho equivalente (EW)** | Área de una línea de absorción expresada como ancho de rectángulo equivalente |
| **Continuo** | Nivel de flujo en ausencia de líneas de absorción |
| **FWHM** | Full Width at Half Maximum — anchura a media altura de un perfil espectral |
| **LTE** | Equilibrio termodinámico local |
| **MAD** | Median Absolute Deviation — estimador robusto de dispersión |
| **MK** | Sistema de clasificación Morgan-Keenan |
| **quality_flag** | Indicador de fiabilidad de una medición de EW |
| **Sigma-clipping** | Método de rechazo de outliers basado en desviación estándar robusta (MAD) |
| **SNR** | Relación señal-ruido (Signal-to-Noise Ratio) |
| **Tipo espectral** | Categoría de clasificación estelar (O, B, A, F, G, K, M) |
| **VR** | Velocidad radial estelar (desplazamiento Doppler) |

---

### Anexo E: Mejoras Metodológicas v3.1 (Marzo 2026)

Esta sección documenta las mejoras implementadas en la versión 3.1 del sistema.

#### E.1 Identificación de Líneas Mejorada

Se introdujo una tolerancia configurable de ±2 Å para identificar el mínimo de absorción observado con la longitud de onda de laboratorio:

```
|λ_obs − λ_lab| < 2 Å   →   línea identificada (quality_flag = 'OK')
|λ_obs − λ_lab| ≥ 2 Å   →   bandera UNRELIABLE_POSITION
```

Esto previene la identificación errónea de absorciones en pares cercanos como:
- Fe I 4144 ≈ He I 4144
- Fe II 5018 ≈ He I 5016
- Fe II 4923 ≈ He I 4921

Las mediciones se tratan como **"absorción integrada en una ventana centrada en λ_lab"**, no como identificaciones atómicas únicas.

#### E.2 Sistema de Banderas de Calidad

Cada medición de EW devuelve ahora un `quality_flag`:

| Bandera | Significado |
|---------|-------------|
| `OK` | Medición válida dentro de tolerancia |
| `NOT_DETECTED` | No se detectó absorción significativa |
| `UNRELIABLE_POSITION` | Mínimo fuera de ±2 Å del centro teórico |
| `TOO_SHALLOW` | Profundidad < 3% (posiblemente ruido) |
| `BLENDED` | Válida pero contaminada por línea vecina (< 3 Å) |
| `WINDOW_TOO_SMALL` | Insuficientes píxeles en la ventana |
| `OUT_OF_RANGE` | Línea fuera del rango espectral cubierto |

#### E.3 Algoritmo de Medición de EW Mejorado

El algoritmo de medición sigue ahora cinco pasos explícitos:

1. **Ventana inicial** según tipo de línea (LINE_TYPES + LINE_WINDOWS)
2. **Localización del mínimo** por detección de picos en flujo invertido
3. **Verificación de posición** (tolerancia ±2 Å)
4. **Verificación de profundidad** (profundidad > 3%)
5. **Detección de blend**: si hay otra absorción a < 3 Å, la ventana se reduce a 6 Å

#### E.4 Ventanas Dependientes del Tipo de Línea

Se implementó la categorización `LINE_TYPES` con ventanas nominales:

| Categoría | Ventana (Å) | Aplicación |
|-----------|:-----------:|------------|
| `Balmer` | 16–30 | Líneas de H con alas de Stark |
| `HeI_strong` | 10–15 | He I visible en O-B |
| `HeI_weak` | 6–10 | He I débil |
| `metal_lines` | ≤ 4 (adaptativa) | Líneas metálicas estrechas |
| `CaI_MgI_NaI` | 10–15 | Resonantes de metales neutros |
| `molecular` | 15–25 | Bandas moleculares TiO, VO, CaH, CH |

#### E.5 Ventanas Adaptativas por FWHM

Para líneas metálicas (`metal_lines`), la ventana de integración se adapta al FWHM medido:

```
window_adaptativa = clip(4 × FWHM, mín=4.0 Å, máx=ventana_nominal)
```

Esto asegura que espectros de alta resolución usen ventanas más estrechas, reduciendo la contaminación por líneas vecinas.

#### E.6 Rechazo de Rayos Cósmicos en Dos Pasos

El orden correcto del preprocesamiento es ahora:

```
1. Estimación rápida del continuo (percentil 90, sin sigma-clip)
2. Sigma-clip RELATIVO AL CONTINUO, por ventanas locales
   (evita sesgos por gradiente azul-rojo del espectro)
3. Detección de puntos de continuo (excluyendo absorción y spikes)
4. Ajuste final con spline suavizado: s = 0.5 × len(puntos) < len(puntos)
```

La desviación estándar robusta usa MAD en lugar de σ clásico:
```
σ_robusto = 1.4826 × MAD
```

#### E.7 Ajuste de Continuo Mejorado

El parámetro de suavizado del spline cambia de `s = len(puntos)` a `s = 0.5 × len(puntos)`, lo que:
- Previene que el spline pase exactamente por todos los puntos de continuo
- Produce un continuo más suave, especialmente en zonas con absorción densa
- Reduce la sobreestimación del continuo en espectros ruidosos

#### E.8 Archivo de Diagnóstico de Mediciones

La función `write_diagnostic_file()` genera `line_measurements.txt`:

```
#  SpectroClass — Archivo de Diagnóstico de Mediciones
#  Tipo espectral clasificado : G2 V
#  SNR estimado               : 87.3  [BUENO; mínimo recomendado: 30.0]
#  Desplazamiento medio (Δλ)  : +0.12 Å  (VR ≈ +8 km/s)
#
line_name              lambda_lab  lambda_meas  window_width      EqW quality_flag
----------------------------------------------------------------------------------------
Ca_II_K                   3933.70      3933.82         10.0    2.843 OK
H_delta                   4101.70      4101.65         20.0    4.102 OK
He_I_4471                 4471.50         ---          12.0    0.000 NOT_DETECTED
H_gamma                   4340.50      4340.48         20.0    3.897 OK
Fe_I_4383                 4383.50      4383.61          4.2    0.312 OK
```

#### E.9 Verificación de Velocidad Radial

La función `check_radial_velocity()` calcula el desplazamiento sistemático medio:

```
VR ≈ c × mean(λ_obs − λ_lab) / λ_ref
```

Si el desplazamiento supera 5 Å (~330 km/s), se emite una advertencia indicando que el espectro puede no estar corregido por velocidad radial.

#### E.10 Diagnósticos para Estrellas Tempranas (O-B)

Se añadieron las razones diagnóstico canónicas para subtipos O y B:

| Razón | Descripción |
|-------|-------------|
| `HeI4471_HeII4541` | He I 4471 / He II 4541 — termómetro principal tipo O |
| `NIII4640_HeII4686` | N III 4640 / He II 4686 — luminosidad en tipo O |
| `SiIV4089_HeI4026` | Si IV 4089 / He I 4026 — temperatura B0–B2 |
| `NV_NIII` | N V promedio / N III promedio — temperatura O3-O5 |
| `OIII5592_HeI5876` | O III 5592 / He I 5876 — confirmación B0-B2 |
| `SiIV4116_SiIII4553` | Si IV 4116 / Si III 4553 — temperatura B0.5 |

Los nuevos diagnósticos permiten refinar la clasificación de:
- **Subtipos O3–O5**: N V prominente, N IV fuerte, N III débil
- **Subtipos O6–O7**: N III ≈ He II en intensidad
- **Subtipos O8–O9**: He I > He II, Si III crece
- **Subtipos B0–B1**: Si IV ≈ Si III
- **Subtipos B2–B3**: Si III > Si IV, Si III > Si II

---

*Documento generado automáticamente - Sistema de Clasificación Espectral v3.1 — Marzo 2026*
