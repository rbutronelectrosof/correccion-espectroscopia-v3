# Métodos de Clasificación Espectral Estelar

> Documentación técnica detallada de cada módulo del sistema **SpectroClass**.
> Todos los algoritmos operan sobre el mismo espectro normalizado y el mismo diccionario
> de mediciones de anchos equivalentes (EW).
> Última actualización: marzo 2026.

---

## Índice

1. [Pre-procesamiento obligatorio](#0-pre-procesamiento-obligatorio)
   - 1.1 [Normalización al continuo](#11-normalización-al-continuo)
   - 1.2 [Medición de anchos equivalentes](#12-medición-de-anchos-equivalentes)
   - 1.3 [Ventanas de integración por línea (LINE_WINDOWS)](#13-ventanas-de-integración-por-línea-line_windows)
   - 1.4 [Razones diagnóstico (compute_spectral_ratios)](#14-razones-diagnóstico-compute_spectral_ratios)
2. [Módulo 1 — Clasificador Físico (26 nodos Gray & Corbally)](#módulo-1--clasificador-físico)
3. [Módulo 2 — Árbol de Decisión ML](#módulo-2--árbol-de-decisión-ml-scikit-learn)
4. [Módulo 3 — Template Matching](#módulo-3--template-matching-χ²)
5. [Sistema de Votación Ponderada](#sistema-de-votación-ponderada)
6. [Guía de subtipos y criterios de precisión](#guía-de-subtipos-y-criterios-de-precisión)
7. [Clasificación de luminosidad (luminosity_classification.py)](#clasificación-de-luminosidad)
8. [Árbol Interactivo — Interfaz web (26 nodos)](#árbol-interactivo--interfaz-web)
9. [Tabla resumen](#tabla-resumen)

---

## 0. Pre-procesamiento obligatorio

Todo espectro, sin importar el método de clasificación, pasa por dos etapas previas
idénticas antes de ser analizado. La calidad de estas etapas determina directamente
la exactitud de cualquier clasificador posterior.

### 1.1 Normalización al continuo

**Archivo:** `src/spectral_classification_corrected.py` — función `normalize_to_continuum()`

El objetivo es obtener un espectro en el que el continuo vale exactamente **1.0**,
de modo que las líneas de absorción sean depresiones por debajo de la unidad y las
líneas de emisión sean picos por encima.

#### Por qué NO se usa min-max

La normalización min-max divide cada punto por el valor máximo global. Esto destruye
la información física: si hay un rayo cósmico en el espectro (un pico espúreo muy
brillante), todos los demás valores quedan comprimidos hacia cero y las líneas
diagnóstico se vuelven inmedibles. Además, el "máximo real" del espectro no es el
continuo, sino el punto más caliente de la función de Planck dentro del rango observado.

#### Algoritmo paso a paso

```
Espectro crudo (λ, F)
        │
        ▼
┌─────────────────────────────────┐
│  PASO 1 · Sigma-clipping        │
│  Rechaza rayos cósmicos         │
│  σ = 3.0,  max_iter = 5         │
│  Máscara booleana: True=válido  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  PASO 2 · Puntos de continuo    │
│  Ventanas de 50 píxeles         │
│  En cada ventana: P90 del flujo │
│  (los 10% más brillantes =      │
│   continuo en esa región)       │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  PASO 3 · Ajuste de continuo    │
│  Spline cúbico (k=3)            │
│  sobre los puntos P90           │
│  Interpolado en todo λ          │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  PASO 4 · Normalización         │
│  F_norm(λ) = F(λ) / C(λ)       │
│  Continuo C(λ) → 1.0            │
└─────────────────────────────────┘
```

**Parámetros clave:**

| Parámetro | Valor por defecto | Descripción |
|-----------|:-----------------:|-------------|
| `window_size` | 50 píxeles | Tamaño de ventana para buscar continuo |
| `poly_order` | 3 | Grado del spline de ajuste |
| `sigma` | 3.0 | Umbral de rechazo de rayos cósmicos |

**Caso especial:** si hay menos de 4 puntos de continuo detectados (espectro muy
corto o muy contaminado), se usa el percentil 90 global como nivel de continuo uniforme.

---

### 1.2 Medición de anchos equivalentes

**Archivo:** `src/spectral_classification_corrected.py` — función `measure_equivalent_width()`

El ancho equivalente (EW) de una línea de absorción es el ancho de un rectángulo de
profundidad igual al continuo que tiene la misma área que la línea:

```
EW = ∫ [ 1 - F(λ)/F_cont(λ) ] dλ
```

Con F_cont = 1.0 (ya normalizado), se simplifica a:

```
EW = ∫ [ 1 - F_norm(λ) ] dλ       (en Å, positivo para absorción)
```

**Valor de retorno ampliado (v3.1):** La función ahora devuelve cuatro valores:

```python
ew, line_depth, quality_flag, lambda_measured = measure_equivalent_width(...)
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `ew` | float | Ancho equivalente en Å |
| `line_depth` | float | Profundidad relativa (0–1) |
| `quality_flag` | str | Estado: `OK`, `NOT_DETECTED`, `UNRELIABLE_POSITION`, `TOO_SHALLOW`, `BLENDED`, `WINDOW_TOO_SMALL` |
| `lambda_measured` | float | Longitud de onda observada del mínimo (0.0 si no detectada) |

**Identificación no unívoca de líneas:** Las mediciones se tratan como
*"absorción integrada en ventana centrada en λ_lab"*. Si el mínimo de absorción
está dentro de ±2 Å del laboratorio, se acepta la identificación sin asumir que
la absorción corresponde únicamente a ese ión. Esto es importante para pares como:

| Par ambiguo | Δλ |
|-------------|:---:|
| Fe I 4144 / He I 4144 | ~0 Å |
| Fe II 5018 / He I 5016 | 2 Å |
| Fe II 4923 / He I 4922 | 2 Å |

#### El algoritmo de medición en 5 pasos

```
Ventana inicial: LINE_WINDOWS o LINE_TYPES-dependiente
LINE_TYPES: adaptar por FWHM si es 'metal_lines'
                │
                ▼
┌──────────────────────────────────────────────┐
│  PASO 1 · Ventana de medición                │
│  ±window_width/2 alrededor de λ_lab          │
│  Si < 3 píxeles → quality_flag=WINDOW_TOO_SMALL│
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  PASO 2 · Detección del mínimo               │
│  find_peaks sobre flujo invertido            │
│  prominence ≥ 0.02, width ≥ 1 px            │
│  Si no hay picos → quality_flag=NOT_DETECTED │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  PASO 3 · Tolerancia de posición (±2 Å)      │
│  |λ_obs − λ_lab| < 2 Å  (LINE_ID_TOLERANCE) │
│  Si fuera → quality_flag=UNRELIABLE_POSITION │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  PASO 4 · Profundidad mínima (3%)            │
│  depth = 1 - F_min / continuo_local          │
│  Si depth < 0.03 → quality_flag=TOO_SHALLOW  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  PASO 5 · Detección de blend                 │
│  Si otra línea está < 3 Å (BLEND_SEPARATION) │
│  → reducir ventana a 6 Å (BLEND_WINDOW)      │
│  → quality_flag=BLENDED (pero sí se mide)    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  CÁLCULO FINAL                               │
│  EW = trapz(1 - F/F_cont, λ)                 │
│  Centrado en λ_obs (no en λ_lab)             │
│  Si flujo > 1.05×cont → emisión → EW < 0 ok  │
│  quality_flag = 'OK'                         │
└──────────────────────────────────────────────┘
```

#### Líneas diagnóstico medidas (85 en total)

| Grupo | Líneas | Tipos |
|-------|--------|-------|
| **Balmer** | Hα 6563, Hβ 4861, Hγ 4341, Hδ 4102, Hε 3970 | Todos |
| **He I** | 4026, 4388, 4472, 4922, 5016, 5876 Å | O, B |
| **He II** | 4200, 4542, 4686, 5411 Å | O |
| **N, C, O ionizados** | N V 4604/4620, N IV 4058, N III 4634/4641, C III 4650, C IV 4658, C II 4267, O II 4415/4417/4591/4596, O V 5114 Å | O, B |
| **Silicio** | Si IV 4089/4116, Si III 4553/4568/4575, Si II 4128/4131/5041/5056 Å | B |
| **Ti II** | 4468 Å | B tardía (B9.5) |
| **Mg II** | 4481 Å | B tardía |
| **Fe II** | 4173, 4233 Å | A tardía |
| **Sr II** | 4077, 4216 Å | A tardía / F (indicador luminosidad) |
| **Ca II** | K 3934, H 3968 Å | A, F, G, K |
| **Ca I** | 4227, 4455, 6122, 6162 Å | F, G, K |
| **Mg I** | 4703, 5183 Å | F, G (tripleta Mg b) |
| **Fe I** | 4046, 4071, 4144, 4260, 4271, 4326, 4383, 4957, 5270, 5328, 6495 Å | G, K, M |
| **Cr I** | 4254 Å | G/K (distinción decisiva) |
| **Ba II** | 4554 Å | K gigante (indicador luminosidad) |
| **Na I** | D 5896 Å | K tardía |
| **CH G-band** | 4300 Å | F, G (banda molecular) |
| **TiO clásico** | 4762, 4955, 5167 Å | M |
| **TiO extendido** | 5448, 6158, 6651 Å | M (discriminación enana/gigante) |
| **VO** | 7434, 7865 Å | M tardía (M5+) |
| **CaH** | 6382, 6750 Å | M enana (discriminación clase lum.) |

---

### 1.3 Ventanas de integración por línea (LINE_WINDOWS + LINE_TYPES)

**Archivo:** `src/spectral_classification_corrected.py` — diccionarios `LINE_WINDOWS` y `LINE_TYPES`

El parámetro `window_width` en `measure_equivalent_width()` es el **ancho total** de la ventana de integración (±window_width/2 a cada lado del centro). Un valor único para todas las líneas producía dos problemas:

- **Ventana demasiado estrecha para Balmer:** Con `window_width=15` la integración capturaba solo ±7.5 Å del centro, perdiendo las alas de las anchas líneas de Balmer en estrellas tipo A/B.
- **Contaminación cruzada en pares cercanos:** Fe I 4071 y Sr II 4077 están solo a 6 Å; O II 4415 y O II 4417 están a 2 Å. Con ventanas de 12–15 Å se mezclan las dos líneas en una sola integral.

#### Categorías de líneas (LINE_TYPES)

`LINE_TYPES` clasifica cada línea en una categoría que determina el tratamiento:

| Categoría | Ventana nominal | Descripción |
|-----------|:--------------:|-------------|
| `Balmer` | 16–30 Å | Hidrógeno con alas de Stark muy anchas |
| `HeI_strong` | 10–15 Å | He I claramente visible en O9–B5 |
| `HeI_weak` | 6–10 Å | He I débil en O tardío / B tardío |
| `metal_lines` | ≤ 4 Å adaptativa | Líneas metálicas estrechas con FWHM adaptativo |
| `CaI_MgI_NaI` | 10–15 Å | Líneas resonantes de metales neutros fuertes |
| `molecular` | 15–25 Å | Bandas moleculares TiO, VO, CaH, CH |

#### Ventanas específicas (LINE_WINDOWS)

| Línea | Ventana (Å) | Justificación |
|-------|:-----------:|---------------|
| H_alpha | 30.0 | Línea más ancha de Balmer; alas extensas en estrellas A |
| H_beta | 20.0 | Segunda línea más ancha |
| H_gamma | 20.0 | Balmer tardío |
| H_delta | 20.0 | Balmer tardío |
| H_epsilon | 16.0 | Balmer tardío; próxima a Ca II H |
| Fe_I_4071 | 8.0 | Par cercano con Sr II 4077 (solo 6 Å de separación) |
| Sr_II_4077 | 8.0 | Par cercano con Fe I 4071 |
| O_II_4591 | 8.0 | Par cercano con O II 4596 |
| O_II_4596 | 8.0 | Par cercano con O II 4591 |
| N_III_4634 | 10.0 | Par N III 4634/4641 |
| N_III_4641 | 10.0 | Par N III 4634/4641 |
| O_II_4415 | 6.0 | Par muy cercano; solo 2 Å entre sí |
| O_II_4417 | 6.0 | Par muy cercano; solo 2 Å entre sí |
| CH_Gband | 25.0 | Banda molecular ancha |
| TiO_5448–6651 | 20.0 | Bandas TiO anchas |
| VO_7434/7865 | 20.0 | Bandas VO anchas |
| CaH_6382/6750 | 15.0 | Bandas CaH moderadas |
| *resto* | 12.0 | Ventana por defecto |

#### Ventanas adaptativas por FWHM (líneas metálicas)

Para líneas de tipo `metal_lines`, `measure_diagnostic_lines()` estima el FWHM y adapta la ventana:

```python
fwhm = estimate_fwhm(wavelengths, flux_normalized, line_wave, window_width=win)
if fwhm > 0.5:
    adaptive_win = clip(4.0 × fwhm, 4.0 Å, win)
```

Esto es especialmente útil en espectros de **alta resolución** (R > 20 000) donde las líneas metálicas son muy estrechas y una ventana de 12 Å incluiría demasiado continuo.

La función `estimate_fwhm()` usa el método de semi-máximo: localiza los puntos a ambos lados del mínimo de absorción donde el flujo cruza el nivel `(F_min + F_cont)/2`.

---

### 1.4 Preprocesamiento: normalización mejorada y rechazo de rayos cósmicos

**Archivo:** `src/spectral_classification_corrected.py` — función `normalize_to_continuum()`

#### 1.4.1 Por qué el sigma-clip global falla en espectros no normalizados

El enfoque clásico (sigma-clip global → fit continuo) produce errores sistemáticos porque:

1. El flujo varía fuertemente con λ (cuerpo negro × función de respuesta del instrumento)
2. En estrellas tempranas, el flujo azul puede ser 5–10× mayor que el rojo
3. Un sigma-clip global rechaza incorrectamente los puntos extremos de cada extremo espectral

#### 1.4.2 Algoritmo de dos pasos

```
PASO 1: Estimación rápida del continuo (sin sigma-clip)
  └─ Percentil 90 en ventanas de 50 px → spline cúbico rough_continuum

PASO 2: Sigma-clip RELATIVO al continuo rough, por ventanas locales
  └─ flux_ratio = flux / rough_continuum
  └─ Para cada ventana: detectar spikes con MAD local
     (σ_robusto = 1.4826 × MAD, umbral = median + 3σ)
  └─ clean_mask[spike] = False

PASO 3: Puntos de continuo limpios
  └─ Solo pixels limpios (clean_mask = True)
  └─ Percentil 90 local → excluye absorción automáticamente

PASO 4: Spline final suavizado
  └─ s_factor = 0.5 × len(puntos_continuo)  →  s < len(puntos)
  └─ Previene sobreajuste y oscilaciones en zonas densas de absorción
```

#### 1.4.3 Sigma-clip robusto (MAD)

La función `sigma_clip()` usa ahora **MAD** en lugar de σ clásico:

```
σ_robusto = 1.4826 × MAD    (factor de normalización para distribución Gaussiana)
```

El MAD es mucho más resistente a outliers extremos (rayos cósmicos de 10–100σ) que la desviación estándar clásica, que quedaría inflada por los propios outliers.

---

### 1.5 Razones diagnóstico (compute_spectral_ratios)

**Archivo:** `src/spectral_classification_corrected.py` — función `compute_spectral_ratios(measurements) -> dict`

Calcula 27 razones de anchos equivalentes (21 originales + 6 nuevas para tipos O-B) que sintetizan los gradientes de temperatura e ionización a lo largo de la secuencia OBAFGKM. Las razones son más robustas que los EW absolutos frente a variaciones de escala de flujo.

| Razón | Fórmula | Aplicación |
|-------|---------|------------|
| `HeI_HeII` | He I 4471 / He II 4686 | Subtipos O3–O9 |
| `NIV_NIII` | N IV 4058 / N III 4641 | O muy tempranas (O3–O5) |
| `NIII_HeII` | N III 4641 / He II 4686 | Of vs O normal |
| `MgII_HeI` | Mg II 4481 / He I 4471 | B tardías vs B tempranas |
| `SiIV_SiIII` | Si IV 4089 / Si III 4553 | B0–B2 |
| `SiIII_SiII` | Si III 4553 / Si II 4128 | B2–B5 |
| `CII_OII` | C II 4267 / O II 4591 | B medias |
| `CaIIK_Heps` | Ca II K / Hε | **Criterio MK clásico A/F** |
| `SrII_FeI` | Sr II 4077 / Fe I 4071 | A tardía / luminosidad |
| `SrII4216_FeI` | Sr II 4216 / Fe I 4071 | A tardía / luminosidad |
| `CrI_FeI` | Cr I 4254 / Fe I 4260 | **Criterio G/K decisivo** |
| `FeI_H` | Fe I avg / H avg | F/G/K presión y temperatura |
| `CHband_FeI` | CH 4300 / Fe I 4260 | F5–G0 |
| `MgIb_FeI` | Mg I 5183 / Fe I 5270 | G0–G2 |
| `CaI_FeI` | Ca I 4227 / Fe I 4144 | G/K; sensible a luminosidad |
| `BaII_FeI` | Ba II 4554 / Fe I 4383 | K gigante vs enana |
| `NaI_CaI` | Na I 5896 / Ca I 6122 | K tardía |
| `TiO_index` | TiO total (suma 6 bandas) | M temprana; fuerza TiO |
| `VO_index` | VO 7434 + VO 7865 | M tardía (M5+) |
| `CaH_index` | CaH 6382 + CaH 6750 | M enana vs gigante |
| `TiO_CaH` | TiO total / CaH total | M; presión (enana vs gigante) |
| **`HeI4471_HeII4541`** ★ | He I 4471 / He II 4541 | **Criterio canónico O; < 1 → O3-O6** |
| **`NIII4640_HeII4686`** ★ | N III 4640 / He II 4686 | **Luminosidad O; mayor en supergigantes** |
| **`SiIV4089_HeI4026`** ★ | Si IV 4089 / He I 4026 | **Temperatura B0–B2** |
| **`NV_NIII`** ★ | N V (4604+4620)/2 / N III | **O3–O5 (T > 40 000 K)** |
| **`OIII5592_HeI5876`** ★ | O III 5592 / He I 5876 | **Confirmación B0–B2** |
| **`SiIV4116_SiIII4553`** ★ | Si IV 4116 / Si III 4553 | **Temperatura B0.5** |

★ Nuevas en v3.1

**Interpretación de razones clave para clases de luminosidad:**
- `SrII_FeI > 0.8` → probable gigante/supergigante A–F (Sr II sensible a gravedad)
- `BaII_FeI > 0.6` → gigante K (Ba II amplificado a baja gravedad)
- `TiO_CaH > 1.5` → M gigante; `TiO_CaH < 0.8` → M enana

**Interpretación de nuevas razones para O-B:**

| Razón | < 1.0 | ≈ 1.0 | > 1.0 |
|-------|-------|-------|-------|
| `HeI4471_HeII4541` | O3–O6 (He II domina) | O7 (iguales) | O8–B0 |
| `SiIV4089_HeI4026` | B2+ (Si IV débil) | B0.5–B1 | B0 (Si IV fuerte) |
| `NIII4640_HeII4686` | O normal | — | Supergigante O (Of) |
| `NV_NIII` | O6+ (N V ausente) | — | O3–O5 (N V visible) |

---

## Módulo 1 — Clasificador Físico

**Archivo:** `src/spectral_classification_corrected.py`
**Función principal:** `classify_star_decision_tree()` (árbol de 26 nodos)
**Función de despacho:** `classify_star_corrected()` (activa por defecto: `use_decision_tree=True`)
**Peso en votación:** 0.10 (con neural) / 0.15 (sin neural)

### Concepto

Este módulo implementa el esquema estándar de clasificación de Gray & Corbally (2009),
operando exclusivamente con **reglas físicas** derivadas de la formación de líneas
espectrales en función de la temperatura efectiva. No usa estadística ni aprendizaje
automático.

### Umbrales de detección calibrados

```python
THRESHOLD_He_II = 0.30 Å   # He II debe ser fuerte para diagnosticar tipo O
THRESHOLD_He_I  = 0.15 Å   # He I mínimo para considerarse detección
THRESHOLD       = 0.08 Å   # Umbral general para líneas débiles
```

Se requieren **≥ 2 líneas de He I** con EW > 0.15 Å para confirmar presencia de helio
neutro (evita falsos positivos por ruido en una sola línea).

**Validación cruzada:** si H_avg > 4.0 Å, se verifica que He_I / H_avg < 0.15 antes de
aceptar la detección de helio. Esto evita confundir líneas de Balmer adyacentes con He I
en estrellas tipo A.

---

### PASO 1 — Clasificación gruesa

```
          ¿He II fuerte? (EW > 0.30 Å en 4200, 4542 o 4686)
                    │
           SÍ ──────┴────── NO
           │                 │
           ▼                 ▼
      TEMPRANA        ¿He I presente?
      (O seguro)      (≥2 líneas > 0.15 Å)
                            │
                  SÍ ───────┴─────── NO
                  │                   │
                  ▼                   ▼
          ¿Balmer muy             ¿Balmer domina?
          fuerte? (>4.0)         (H_avg > 2×metales)
                  │                   │
               SÍ │ NO           SÍ ──┴── NO
                  │                │       │
             [caida a           INTER-  ¿Muchos
            INTERMED.]         MEDIA    metales?
                                        │
                                   SÍ ──┴── NO
                                   │        │
                                TARDÍA   TARDÍA
```

**Resultado:** `clasificacion_gruesa` ∈ {`'temprana'`, `'intermedia'`, `'tardía'`}

---

### PASO 2 — Estrellas tempranas: O, B, A

#### Rama O — He II fuerte presente

```
He II fuerte confirmado → Tipo O
           │
           ▼
    He I 4471 / He II 4542
           │
    ratio < 0.5   →  O5          (He I << He II, T ≥ 40 kK)
    ratio 0.5–0.7 →  O5          (con refinamiento N V)
    ratio 0.7–0.85→  O6
    ratio 0.85–1.3→  O7          (He I ≈ He II)
    He I 4387 < He II →  O8
    He I 4387 ≈ He II →  O9
    He I 4387 ≥ He II →  O9/O9.5
           │
           └── ¿Si III 4553 ≈ He II 4542?
               SÍ → O9.5   (indicador crítico O9-B0)
               NO → O9
```

**Diagnósticos adicionales para tipos muy tempranos:**

| Criterio | Subtipo |
|----------|---------|
| N V 4604 muy fuerte (> 0.5 Å) + O V presente | O3 |
| N V 4604 > 0.15 Å  o  He II/He I > 5 | O4 |
| He I/He II < 0.70 | O5 |
| He I/He II 0.70–0.85 | O6 |
| He I/He II 0.85–1.30 | O7 |
| He I 4387 < He II 4542 | O8 |
| He I 4387 ≈ He II 4542 (±20%) | O9 |
| Si III 4553 visible y ≈ He II | O9.5 |
| Si III 4553 > He II → transición | O9.7-B0 |

**Clase de luminosidad para tipo O:** se detecta mediante la presencia y forma de
líneas de emisión (Hα, He II 4686 en emisión) para asignar sufijos Ia, Ib, II, III, V.

#### Rama B — He I presente, He II ausente

```
He I presente, He II ausente → Tipo B
           │
           ▼
    ¿He I 4471 >> Mg II 4481?
    (He I > 2×Mg II  o  Mg II < 0.05 Å)
           │
    SÍ ────┴──── NO
    │             │
    ▼             ▼
 B TEMPRANA    B TARDÍA
 criterio Si   ratio Mg II / He I
```

**B temprana — criterio Silicio:**

| Condición | Subtipo |
|-----------|---------|
| He II 4686 aún visible + Si IV ≈ Si III | B0 |
| Si IV > Si III (ratio > 1.0) | B0.5 |
| Si IV > 0.1 Å y Si III ≥ Si II | B1 |
| Si III/Si II ratio > 1.8 | B1 |
| Si III/Si II ratio 1.2–1.8 (Si IV ausente) | B2 |
| Si III/Si II ratio 1.0–1.2 | B2-B3 |
| Si III/Si II ratio 0.7–1.0 | B3 |
| Si III/Si II ratio 0.3–0.7 | B4 |
| Si II >> Si III (ratio < 0.3) | B5 |

**B tardía — criterio Mg II / He I:**

| ratio Mg II 4481 / He I 4471 | Subtipo |
|:----------------------------:|---------|
| < 1.5  (Mg II ≈ He I) | B6-B7 |
| 1.5 – 3.0  (Mg II 2-3× He I) | B8 |
| ≥ 3.0  (Mg II >> He I) | B9 |
| Ti II 4468 ≈ He I 4471 (además) | **B9.5** |

> **Ti II 4468 Å** es el indicador decisivo de B9.5: aparece a la izquierda de He I 4471
> y cuando iguala su intensidad marca la transición B9.5 / A0.

**Confirmaciones adicionales:**
- O II 4591 presente → confirma B temprana
- C II 4267 presente → diagnóstico B
- He II 4686 vestigial → confirma B0

#### Rama A — sin He I en espectro temprano

Si la clasificación gruesa es `temprana` pero no se detecta He I:
```
Sin He I → Tipo A (subtipo A5 por defecto)
Confianza reducida a 80%
Advertencia: "Clasificación A por ausencia de He I"
```

---

### PASO 3 — Estrellas intermedias: A, F

Se ejecuta cuando `clasificacion_gruesa == 'intermedia'`.

#### Bandera `covers_blue`

Antes de cualquier comparación, el clasificador comprueba si el espectro cubre la
región azul de Ca II K:

```python
covers_blue = (λ_min ≤ 3950 Å)
```

- `covers_blue = True`  → se usa el **criterio canónico MK** (Ca II K vs Hε)
- `covers_blue = False` → se usa el **criterio alternativo** (Balmer + Mg II + Fe I)

La interfaz muestra un banner de color diferente según cuál criterio se aplicó.

#### Criterio canónico: Ca II K (3934 Å) vs H epsilon (3968 Å)

> **¿Por qué Ca II K y no Ca II H?**
>
> El calcio ionizado produce un **doblete**:
> - Línea **K** a 3933.7 Å — despejada, medible directamente
> - Línea **H** a 3968.5 Å — casi superpuesta con Hε (hidrógeno) a 3970 Å
>
> Al decir "Ca II K" se garantiza medir solo el calcio sin contaminación del hidrógeno.
> La nomenclatura histórica (letras de Fraunhofer) bautizó estas líneas antes de que
> se conociera su origen atómico.

```
ratio = EW(Ca II K 3934) / EW(Hε 3968)

ratio < 0.20  →  A0-A2   (Ca II K casi invisible, Balmer en máximo)
ratio 0.20–0.35 →  A3-A5
ratio 0.35–0.60 →  A7-A9   (Ca II K ≈ ½ Balmer)
ratio 0.60–1.20 →  F0       (Ca II K ≈ Balmer, punto de cruce)
ratio > 1.20  →  F5+      (Ca II K domina)
```

**Ca II K como termómetro estelar:**

| Tipo / T efectiva | Ca II K vs Balmer |
|:-----------------:|-------------------|
| A0-A2 / ~10 000 K | K casi invisible — Balmer en máximo absoluto (ej. Sirio A1V) |
| A5-A9 / ~8 000 K  | K visible, ≈ ¼–½ de Hδ |
| F0 / ~7 500 K     | K ≈ Balmer — punto de cruce |
| F5+ / < 6 500 K   | K domina sobre Balmer |

#### Criterio alternativo (sin Ca II K)

Cuando el espectro no cubre λ < 3950 Å, la confianza se reduce −15% y se usa:

```
¿H_avg > 6.0 y sin metales?          →  A0-A2
¿H_avg > 4.0 y metales débiles?
   ├── Con Mg II 4481  →  A5-A7
   └── Sin Mg II       →  A2-A5
¿H_avg > 2.0 y metales visibles?
   ├── H/metal > 5.0   →  F0-F2
   ├── H/metal 2.0–5.0 →  F5
   └── H/metal 1.0–2.0 →  F8
¿H_avg > 2.0 y sin metales?          →  A7-A9
Resto                                →  F5-F8
```

---

### PASO 4 — Estrellas tardías: F, G, K, M

Se ejecuta cuando `clasificacion_gruesa == 'tardía'`.

```
¿Balmer aún visible? (H_avg > 2.0 y H > metales)
         │
  SÍ ────┴──── NO
  │             │
  ▼             ▼
Tipo F        ¿Ca I 4227 fuerte?
              (> 0.5 Å y Ca I > H  o  H < 1.0)
                   │
           SÍ ─────┴───── NO
           │               │
           ▼               ▼
      EVALUAR TiO    DISTINGUIR G vs K
           │
┌──────────┼──────────┐
│          │          │
TiO fuerte  TiO débil  Sin TiO
asimetría   → K3-K5    → Paso 12
Fe I 4957
     │
 ┌───┼────────┐
 │   │        │
M5-M6 M2-M4  M0-M2
```

#### Tipo F — subtipos basados en ratio Fe I / Hβ

| H_avg | ratio Fe I / Hβ | Subtipo |
|:-----:|:---------------:|---------|
| > 5.0 | < 0.20 | F0-F2 |
| > 5.0 | ≥ 0.20 | F2-F5 |
| 3.0–5.0 | < 0.35 | F5-F7 |
| 3.0–5.0 | ≥ 0.35 | F7-F8 |
| < 3.0 | > 0.50 | F8-G0 (transición) |
| < 3.0 | ≤ 0.50 | F8 |

**F9-G0:** cuando H ≈ metales más intensos (Fe I, Ca I) el espectro está en la
transición a tipo G. Se asigna F8-G0 con advertencia.

#### Paso 12 — Distinguir G vs K

```
Comparar Hδ (4102 Å) con metales (Ca I 4227, Fe I 4046, Fe I 4144)
           │
    ┌──────┼────────────┐
    │      │            │
    ▼      ▼            ▼
H > todos  H ≈ Ca I   H < Ca I y Fe I
metales                      │
G0         Cr I 4254 / Fe I 4260
           │         (criterio G vs K)
      ┌────┼────┐
      │    │    │
   <0.90  ≈1.0 >1.10
   G2-G5  G5  G5-G8
```

Cuando H < Ca I y H < Fe I, el ratio **Cr I 4254 / Fe I 4260** actúa como árbitro:

| Cr I 4254 / Fe I 4260 | Clasificación |
|:---------------------:|---------------|
| < 1.0 (Cr < Fe) | **G tardía** (G8) |
| ≥ 1.0 (Cr > Fe) | **K temprana** (K0) |
| Ca I >> Fe I × 2 | K tardía (K5-K7) |

> **Física del criterio Cr/Fe:** a ~5 000 K (límite G/K) el cromo y el hierro tienen
> potenciales de ionización similares, pero sus abundancias relativas en la atmósfera
> cambian con la temperatura. En G tardías el Fe I domina sobre Cr I; en K tempranas
> Cr I alcanza o supera al Fe I. Este criterio fue propuesto por Gray & Corbally (2009)
> como el **discriminante más robusto** en la zona G5-K0.

#### Diagnóstico especializado para tipo M

**`evaluate_tio_bands()`** evalúa 3 bandas TiO:

| Banda TiO | λ (Å) | Significado |
|-----------|:-----:|-------------|
| Primera | 4762 | Aparece en K tardío / M temprano |
| Principal | 4955 | Banda más intensa y diagnóstica |
| Tardía | 5167 | Confirma temperatura < 3 500 K |

```
profundidad promedio de TiO (tio_avg_depth):

< 0.05        →  no es M (sin TiO)
< 0.12        →  M0 (TiO mínimo)
0.12–0.25     →  M0-M2
0.25–0.40     →  M2-M4
0.40–0.50     →  M4-M5 / M5-M6
≥ 0.50        →  M5-M6 / M6-M7
```

**`detect_FeI_4957_asymmetry()`** detecta la firma característica de tipo M:

La banda TiO 4955 Å eleva el flujo a la izquierda de Fe I 4957 Å creando asimetría:

```
ratio_asimetría = (flujo_izquierda − flujo_derecha) / profundidad_Fe I

Fe I simétrica              →  K5 (sin TiO)
Fe I asimétrica (ratio>0.3) →  M0-M2
Fe I muy asimétrica         →  M2-M4
Fe I casi desaparece        →  M5-M6

CRITERIO M6-M7 "dientes de sierra":
  Las 3 bandas TiO profundidad > 0.30  AND  Fe I 4957 depth < 0.05
  → El espectro muestra ondulaciones características ("dientes")
  → No hay líneas metálicas individuales distinguibles
  → Subtipo: M6-M7
```

---

### Estimación de confianza del clasificador físico

La confianza parte de 100% y se penaliza según contradicciones físicas:

| Condición | Penalización |
|-----------|:------------:|
| Tipo O pero He II < 0.10 Å | −30% |
| Tipo O con He II moderado (0.10–0.20 Å) | −20% |
| Tipo B pero He I < 0.20 Å | −25% |
| Tipo B pero He II > 0.10 Å (posible O tardía) | −30% |
| Tipo A pero H_avg < 4.0 Å | −30% |
| Tipo A pero He I > 0.10 Å (posible B) | −40% |
| Tipo F/G/K pero metales débiles (Ca K < 1, Fe < 0.3) | −35% |
| Tipo F/G/K pero H muy fuerte > 6.0 Å (posible A) | −30% |
| Tipo M pero Fe I < 0.5 Å | −25% |
| Sin Ca II K en tipos A/F/G/K | −15% adicional |

---

---

## Módulo 2 — Árbol de Decisión ML (scikit-learn)

**Archivo:** `src/spectral_validation.py` — clase `DecisionTreeClassifier`
**Modelo:** `models/decision_tree.pkl` (entrenado con catálogo ELODIE, 856 estrellas)
**Peso en votación:** 0.40 (con neural) / 0.70 (sin neural)
**Precisión:** ~84% sobre el catálogo ELODIE

### Concepto

Este módulo usa un árbol de decisión estadístico entrenado con 856 espectros reales
del catálogo ELODIE. A diferencia del Módulo 1, no conoce las reglas físicas: aprende
los límites de separación entre clases directamente de los datos.

La salida incluye **probabilidades por clase** (`predict_proba`), lo que permite
cuantificar la incertidumbre de la clasificación.

### Extracción de features (24 dimensiones)

#### Grupo 1 — 17 EWs directos (en Å)

```
 1. He II 4686    7.  Si IV 4089   13. Ca I 4227
 2. He I  4471    8.  Si III 4553  14. Fe I 4046
 3. H beta        9.  Si II 4128   15. Fe I 4144
 4. H gamma      10.  Mg II 4481   16. Fe I 4383
 5. H delta      11.  Ca II K      17. Fe I 4957
 6. H epsilon    12.  Ca II H
```

#### Grupo 2 — 7 ratios calculados

| # | Ratio | Interpretación física |
|---|-------|----------------------|
| 18 | He I 4471 / (He II 4686 + 0.01) | Discrimina O vs B |
| 19 | Si III 4553 / (Si II 4128 + 0.01) | Subtipos B0-B5 |
| 20 | Ca II K / (Hε + 0.01) | Límite A/F (criterio MK clásico) |
| 21 | Mg II 4481 / (He I 4471 + 0.01) | B tempranas vs tardías |
| 22 | Fe_avg / H_avg | Discrimina F/G/K |
| 23 | Ca I 4227 / (Hγ + 0.01) | G vs K |
| 24 | H_avg = (Hβ + Hγ + Hδ) / 3 | Intensidad Balmer global |

Donde `Fe_avg = (Fe I 4046 + Fe I 4144 + Fe I 4383) / 3`.

### Predicción y confianza

```
Vector X [24 features]
         │
         ▼
DecisionTreeClassifier.predict(X)
         │
         ├── tipo predicho: 'F'
         │
DecisionTreeClassifier.predict_proba(X)
         │
         ├── {'O': 2%, 'B': 1%, 'A': 1%, 'F': 94%, 'G': 1%, 'K': 1%, 'M': 0%}
         │
         └── confianza = probabilidad del tipo predicho = 94%
```

Si el archivo `.pkl` no existe, el módulo retorna `None` y su peso se redistribuye
entre los módulos restantes.

---

---

## Módulo 3 — Template Matching (χ²)

**Archivo:** `src/spectral_validation.py` — clase `TemplateMatchingClassifier`
**Peso en votación:** 0.10 (con neural) / 0.15 (sin neural)
**Naturaleza:** No paramétrico, no requiere entrenamiento

### Concepto

El Template Matching compara las mediciones EW del espectro problema con un
**conjunto de valores típicos de referencia** para cada tipo espectral,
usando la estadística chi-cuadrado reducida como métrica de similitud.

El tipo con el menor χ² (mejor ajuste) gana.

### Templates de referencia

| Línea | O | B | A | F | G | K | M |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| He II 4686 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| He I 4471 | 0.5 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| H beta | 2.0 | 3.0 | 9.0 | 4.0 | 2.0 | 1.0 | 0.5 |
| H gamma | — | — | 8.0 | 3.5 | — | — | — |
| H delta | — | — | 7.0 | — | — | — | — |
| Ca II K | 0.0 | 0.1 | 2.0 | 4.0 | 8.0 | 10.0 | 8.0 |
| Ca I 4227 | — | — | — | 0.8 | 1.5 | 2.5 | 3.0 |
| Fe I 4046 | 0.0 | 0.0 | 0.2 | 0.5 | 1.0 | 1.5 | 2.0 |
| TiO 4955 | — | — | — | — | — | — | 1.5 |

*Todos los valores en Ångström. "—" indica línea no incluida en ese template.*

### Cálculo del chi-cuadrado reducido

```
χ²_red(T) = (1/N) × Σ_líneas [ (EW_medido − EW_template)² / error² ]

error = max( 0.10 × EW_template,  0.05 Å )
```

### Conversión de χ² a confianza

```
confianza (%) = 100 / ( 1 + χ²_red / 2 )
```

| χ²_red | Confianza | Interpretación |
|:------:|:---------:|----------------|
| 0.5 | 80% | Ajuste excelente |
| 1.0 | 67% | Ajuste bueno |
| 2.0 | 50% | Ajuste moderado |
| 5.0 | 29% | Ajuste pobre |
| 10.0 | 17% | Ajuste muy malo |
| 21.1 | 9% | Ejemplo real: `Template (χ²=21.1)` |

---

---

## Sistema de Votación Ponderada

**Archivo:** `src/spectral_validation.py` — método `_weighted_vote()`

### Acumulación de votos

```
votos[tipo] += peso_método × (confianza_método / 100)
```

### Pesos según disponibilidad de módulo neural

| Método | Con neural | Sin neural |
|--------|:----------:|:----------:|
| Físico | 0.10 | 0.15 |
| Árbol ML | 0.40 | 0.70 |
| Template | 0.10 | 0.15 |
| KNN / CNN | 0.40 | — |
| **Total** | **1.00** | **1.00** |

### Efecto consenso

```
ratio = votos[tipo_1] / votos[tipo_2]

ratio > 2.0  →  confianza × 1.1   (máximo 95%)
ratio ≤ 2.0  →  confianza × 0.85  (penalización por ambigüedad)
```

### Justificación textual generada

| Método que votó | Texto generado |
|-----------------|----------------|
| Físico | `"Clasificador físico"` |
| Árbol ML | `"ML (94% prob.)"` |
| Template | `"Template (χ²=2.3)"` |
| KNN | `"KNN (100%)"` |
| CNN 1D | `"CNN_1D (87%)"` |

### Estimación de subtipo (votación gana sobre físico)

Si el tipo ganador de la votación difiere del tipo del módulo físico, el subtipo se
estima por ratios EW en `_estimate_subtype()`:

| Tipo | Criterio de subtipo |
|:----:|---------------------|
| O | He I / He II: < 0.5 → O5, < 1.0 → O7, ≥ 1.0 → O9 |
| B | Mg II / He I: < 1.5 → B6, < 3.0 → B8, ≥ 3.0 → B9 |
| A | H_avg: > 7 → A1, > 5.5 → A3, < 5.5 → A7 |
| F | H_avg / Fe_avg: > 5 → F2, > 2 → F5, < 2 → F8 |
| G | H_avg / Ca K: > 3 → G0, > 1.5 → G5, < 1.5 → G8 |
| K | Ca K: > 8 → K5, > 5 → K3, < 5 → K0 |
| M | TiO 4955: > 1.0 → M5, > 0.5 → M2, < 0.5 → M0 |

---

---

## Guía de subtipos y criterios de precisión

Esta sección documenta las razones de intensidad entre líneas diagnósticas que permiten
determinar subtipos con mayor precisión, siguiendo el esquema de Gray & Corbally (2009).

### Tipos O — Ionización de He y N

| Condición | Subtipo |
|-----------|---------|
| He I 4471 < He II 4542, ratio < 0.5 | O5 o anterior |
| He I 4471 < He II 4542, ratio 0.5–0.85 | O6–O7 |
| He I 4387 ≥ He II 4542 | O9 o posterior |
| Si III 4553 ≈ He II 4542 | O9–B0 |
| N V 4604 fuerte + O V presente | O3–O4 |

### Tipos B — He I vs Mg II (tardías) y Silicio (tempranas)

**B tempranas** — determinadas por los estados de ionización del Silicio:

| Si IV 4089 vs Si III 4553 vs Si II 4128 | Subtipo |
|-----------------------------------------|---------|
| Si IV ≈ Si III | B0 |
| Si IV > Si III | B0.5 |
| Si III >> Si II (ratio > 1.8) | B1 |
| Si III > Si II (1.2–1.8) | B2 |
| Si III ≈ Si II (0.7–1.2) | B3 |
| Si II > Si III (0.3–0.7) | B4 |
| Si II >> Si III (< 0.3) | B5 |

**B tardías** — determinadas por Mg II 4481 vs He I 4471:

| Mg II / He I | Subtipo | Descripción |
|:------------:|---------|-------------|
| < 1.5 | B6-B7 | Mg II ≈ He I |
| 1.5 – 3.0 | B8 | Mg II 2-3 veces He I |
| ≥ 3.0 | B9 | Mg II >> He I |
| + Ti II 4468 ≈ He I 4471 | **B9.5** | Ti II aparece a la izquierda de He I |

### Tipos A — Ca II K como termómetro

| Ca II K 3934 vs Hε 3968 | Subtipo | T efectiva |
|:-----------------------:|---------|:----------:|
| K casi ausente | A0–A2 | ~10 000 K |
| K débil (≈ ¼ Hδ) | A3–A5 | ~8 500 K |
| K ≈ ½ Hδ | A5–A7 | ~8 000 K |
| K ≈ ½ Hε | A7–A9 | ~7 500 K |

### Tipos F — H vs metales

| H vs metales | Subtipo |
|--------------|---------|
| H >> Fe I y Ca I | F0–F4 |
| H ≈ metales (primer cruce) | F5 |
| H < metales, Ca II K muy fuerte | F5–F9 |
| H ≈ metales más intensos + G-band tenue | F9–G0 |

### Tipos G/K — Cr I 4254 vs Fe I 4260 (criterio decisivo)

> **Regla de oro G vs K:** comparar Cr I 4254.3 Å con Fe I 4260.5 Å.
>
> - **Cr I < Fe I** → tipo G (tardía si los metales son fuertes)
> - **Cr I > Fe I** → tipo K (temprana)
>
> Este criterio funciona porque la razón de población de Cr I / Fe I cambia de forma
> monotónica y sensible en el rango 4 500–5 500 K, precisamente donde se encuentra la
> frontera G/K (~5 000 K, Sol G2V).

| H, Ca I, Fe I, Cr I | Subtipo G |
|---------------------|-----------|
| H > Ca I y Fe I | G0–G2 |
| H ≈ Ca I; Cr/Fe < 0.90 | G2–G5 |
| H ≈ Ca I; Cr/Fe ≈ 1.0 | G5 |
| H ≈ Ca I; Cr/Fe > 1.10 | G5–G8 |
| H < Ca I; Cr/Fe < 1.0 | G8 |

| Condición | Subtipo K |
|-----------|-----------|
| H < Ca I; Cr/Fe > 1.0 | K0–K2 |
| Fe I 4144 moderado | K2–K4 |
| Fe I 4046 fuerte o Ca I > 1.2 Å | K3–K5 |
| Fe I 4144 > 1.2 Å o Ca I > 2.0 Å | K5–K7 |
| TiO emergente | K7–M0 |

### Tipos M — TiO y desaparición de Fe I

| TiO y Fe I 4957 | Subtipo |
|-----------------|---------|
| TiO débil; Fe I 4957 claramente asimétrica | **M0** |
| TiO moderado (2 bandas) | M0–M2 |
| TiO fuerte; Fe I 4957 muy asimétrica | M2–M4 |
| Fe I 4957 **desaparece** por completo | **M4** |
| TiO muy fuerte (3 bandas > 0.30) | M5–M6 |
| 3 bandas TiO > 0.30 **y** Fe I 4957 depth < 0.05 | **M6–M7** ("dientes de sierra") |

---

---

## Clasificación de luminosidad

**Archivo:** `src/luminosity_classification.py`
**Funciones:** `estimate_luminosity_class()`, `combine_spectral_and_luminosity()`
**Endpoint Flask:** `POST /api/luminosity`

### Concepto

La clase de luminosidad MK (I–V) refleja la gravedad superficial de la estrella y por lo
tanto su estado evolutivo. Supergigantes (I) tienen atmósferas extendidas de baja densidad
que producen líneas más estrechas y distintas razones de ionización comparadas con enanas
(V). El módulo estima la clase de luminosidad a partir de indicadores sensibles a la
gravedad.

### Indicadores por grupo espectral

| Grupo | Indicadores principales | Clase I/II | Clase V |
|-------|------------------------|------------|---------|
| O | He II 4686 emisión / absorción | emisión → Ia/Ib | absorción → V |
| B | Si IV/Si II ratio; anchura de Balmer | Si IV grande → Ia | Balmer ancho → V |
| A | Sr II 4077/Fe I 4071 ratio | Sr II fuerte → gigante | Sr II débil → enana |
| F | Sr II 4216/Fe I; Ca I 4455 | Sr II > Fe I → II/III | — |
| G | Ca I/Fe I ratio | CaI fuerte relativo → III | CaI débil → V |
| K | Ba II 4554/Fe I 4383 ratio | Ba II fuerte → III | Ba II débil → V |
| M | TiO 6651/CaH 6750 ratio | TiO > CaH → M gigante | CaH > TiO → M enana |

### Endpoint `/api/luminosity`

```
POST /api/luminosity
Content-Type: application/json

Request:
{
  "wavelength": [...],    // array de longitudes de onda (Å)
  "flux": [...],          // array de flujo normalizado
  "spectral_type": "G2"   // tipo espectral ya clasificado
}

Response:
{
  "success": true,
  "luminosity_class": "V",
  "mk_full": "G2V",
  "lum_name": "Enana (secuencia principal)",
  "indicators": {
    "CaI_FeI": 0.791,
    "BaII_FeI": 0.23,
    ...
  }
}
```

### Visualización en la interfaz web

- **Pestaña "Clasificación Espectral Estelar":** muestra badge `mkFullBadge` (ej. `G2V`) y
  nombre de clase (`lumClassName`) directamente bajo el subtipo del clasificador físico. Los datos
  se leen de `diagnostics.luminosity_class` y `diagnostics.mk_full` devueltos por `/upload`.

- **Pestaña "Árbol Interactivo":** llama a `/api/luminosity` de forma asíncrona cuando se
  obtiene un resultado en el árbol y hay un espectro cargado. Muestra estado "calculando…",
  luego el tipo MK completo (`arbolMKFull`) con nombre de clase (`arbolLumName`) e indicadores
  como píldoras de color. Si no hay espectro cargado muestra aviso `arbolLumNoSpectrum`.

---

## Árbol Interactivo — Interfaz web

**Pestaña:** "Árbol Interactivo" (`id="tab-arbol"`)
**Archivo JS:** `webapp/static/script.js` — objeto `ARBOL_NODOS` y función `arbolDibujarEspectro()`

### Concepto

El árbol interactivo permite al usuario recorrer manualmente los mismos pasos del
clasificador físico, observando el espectro cargado en cada etapa y eligiendo la opción
que mejor describe lo que ve. Contiene **26 nodos** (ampliado de 18 originales).

### Nodos del árbol (26 en total)

| ID nodo | Pregunta diagnóstica | Resultado / Ramas |
|---------|---------------------|-------------------|
| `inicio` | ¿He II fuerte (> 0.30 Å)? | → tipoO / tipoB_helio |
| `tipoO_sub` | ¿He I ≈ He II (ratio 0.85–1.30)? | → tipoO_late / tipoO_early |
| `tipoO_early` | ¿N V 4604 fuerte? | → tipoO_N34 / resultado O5–O7 |
| `tipoO_late` 🆕 | ¿N III 4634 visible? | → O7–O8 / O8–O9 |
| `tipoO_N34` 🆕 | ¿O V 5114 presente? | → O3 / O4 |
| `tipoB_helio` | ¿He I presente (≥2 líneas > 0.15 Å)? | → tipoB_temprana / tipoB_tardia |
| `tipoB_temprana` | ¿Si III domina sobre Si II? | → tipoB_OII / B1–B2 |
| `tipoB_OII` 🆕 | ¿O II 4415/4417 visible? | → B1–B3 / B0–B1 |
| `tipoB_tardia` | ¿Si IV 4089 presente? | → B0–B1 / tipoB95 |
| `tipoB95` | ¿Ti II 4468 ≈ He I 4471? | → B9.5 / B9 |
| `tipoA` | ¿Ca II K / Hε ratio? | → A0–A2 / A3–A5 / A7+ |
| `tipoF` | ¿H ≈ metales (F5–F9)? | → tipoFG_CH / F0–F4 |
| `tipoFG_CH` 🆕 | ¿CH G-band 4300 visible? | → F5–G0 / F5–F8 |
| `tipoG` | ¿Cr I < Fe I (criterio G/K)? | → tipoG_Mg / K temprana |
| `tipoG_Mg` 🆕 | ¿Mg I b triplete 5183 visible? | → G0–G2 / G2–G5 |
| `tipoK` | ¿Ca II K >> Fe I, sin TiO? | → tipoK_Na / K5–K7 |
| `tipoK_Na` 🆕 | ¿Na I D 5896 fuerte? | → K0–K5 / K2–K4 |
| `tipoKM` | ¿TiO fuerte (bandas múltiples)? | → M2–M4 / tipoM_sub / tipoM_late |
| `tipoM_sub` 🆕 | ¿VO 7434 visible + CaH 6382? | → M2–M6 / M0–M2 |
| `tipoM_late` 🆕 | ¿VO 7865 > TiO 6651? | → M5–L0 / M5–M8 |
| *… 6 nodos adicionales* | Subtipos y confirmaciones | Resultados hoja |

*(Los nodos marcados con 🆕 son los 8 nuevos añadidos respecto a la versión anterior de 18 nodos.)*

### Líneas fuera del rango espectral

Cuando un paso del árbol requiere comparar una línea que está fuera de la cobertura
espectral del archivo cargado (caso típico: Ca II K 3934 Å en espectros que empiezan
en ~4000 Å), el sistema:

1. **Expande la vista** hacia la izquierda para incluir la longitud de onda de la línea
   (solo para líneas con λ < 6000 Å; no se expande hacia la derecha).
2. **Dibuja la línea de forma diferenciada:**
   - Trazo discontinuo corto (4,4 en vez de 10,6)
   - Color gris (#64748b) en vez del color diagnóstico
   - Etiqueta: "no detectada / fuera de rango"
   - Sin área cliqueable de zoom
3. **Muestra la región "sin cobertura"** con un fondo oscuro semitransparente y el
   texto "sin cobertura / λ < XXXX Å" junto a una línea vertical que marca el inicio
   real del espectro.

Esto permite al usuario ver **dónde debería estar** Ca II K y entender por qué el
sistema usa el criterio alternativo.

### Información del botón de ayuda (ícono ?)

Cada paso del árbol incluye un globo de ayuda con:
- Física de la línea diagnóstica
- Qué indica cada opción sobre la temperatura estelar
- Rango de tipos espectrales posibles

En el paso Ca II K vs Balmer, el globo explica el doblete Ca II H & K, por qué se
prefiere la línea K, y la tabla de Ca II K como termómetro de A0 a F5.

---

---

## Tabla resumen

| Módulo | Tipo | Peso (sin/con neural) | Precisión | Fortaleza | Debilidad |
|--------|------|-----------------------|:---------:|-----------|-----------|
| **Físico** | Reglas Gray & Corbally (26 nodos) | 15% / 10% | ~60-90% | Tipos O y B; transparente y auditable | F/G/K sin Ca II K (~5%) |
| **Árbol ML** | Estadístico (sklearn) | 70% / 40% | ~84% | Tipos tardíos F/G/K; datos reales ELODIE | Depende del archivo `.pkl` |
| **Template** | Comparación χ² | 15% / 10% | ~65% | Sin entrenamiento; siempre disponible | No captura variaciones metalicidad/logg |
| **KNN** | ML vecinos | — / hasta 40% | variable | Suplementa árbol; mismo espacio features | Requiere entrenamiento previo |
| **CNN 1D** | Deep Learning | — / hasta 40% | variable | Usa espectro completo; captura morfología | Requiere TensorFlow + más datos |
| **Luminosidad** | Razones sensibles a gravedad | — | estimación | Produce tipo MK completo (ej. G2V) | Aproximado; requiere rango ~4000–7900 Å |

### Endpoints Flask relevantes

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/upload` | POST | Clasificar espectro (devuelve tipo + luminosidad) |
| `/batch_upload` | POST | Procesar múltiples espectros |
| `/api/luminosity` | POST | Estimar clase de luminosidad separadamente |
| `/export_csv` | POST | Exportar resultados |
| `/train_tree` | POST | Entrenar árbol ML |
| `/health` | GET | Estado del servidor |

*La precisión del clasificador físico es alta para O y B (criterios físicos robustos)
pero se reduce en F/G/K cuando el espectro no cubre la región de Ca II K (λ < 3934 Å).
El criterio Cr I 4254 / Fe I 4260 mejora la distinción G/K en ambos clasificadores
(árbol de decisión y físico/legacy) cuando estas líneas son medibles.*

---

---

## Nuevas Funciones en v3.1

### `estimate_fwhm(wavelengths, flux_normalized, line_center, window_width=12.0)`

Estima el FWHM de una línea espectral mediante el método de semi-máximo.

**Uso principal:** adaptar ventanas de integración en espectros de alta resolución.

**Algoritmo:**
1. Localizar el mínimo de la línea dentro de la ventana
2. Calcular el nivel de semi-máximo: `F_medio = (F_min + F_cont) / 2`
3. Encontrar los cruces izquierdo y derecho del semi-máximo
4. `FWHM = λ_derecha − λ_izquierda`

Devuelve `0.0` si el FWHM no puede estimarse (línea demasiado débil o ancha).

---

### `compute_snr(wavelengths, flux_normalized, region=None)`

Estima la relación señal-ruido (SNR) en una región limpia del espectro.

```python
snr = compute_snr(wavelengths, flux_normalized)
```

**Algoritmo:** usa la variación píxel a píxel (`np.diff`) para separar señal de ruido:
```
ruido = std(diff(flux)) / sqrt(2)
SNR = mediana(flux) / ruido
```

**Valores de referencia:**
- SNR < 30: clasificación poco fiable
- SNR 30–100: clasificación aceptable
- SNR > 100: clasificación óptima (como el catálogo ELODIE)

**Constantes asociadas:**
- `SNR_MINIMUM = 30.0`
- `RESOLUTION_MINIMUM = 2000.0`

---

### `write_diagnostic_file(measurements, output_path, spectral_type, subtype, snr, rv_info)`

Genera el archivo de diagnóstico `line_measurements.txt` con todas las mediciones.

**Columnas:** `line_name`, `lambda_lab`, `lambda_meas`, `window_width`, `EqW`, `quality_flag`

**Ejemplo de salida:**
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

---

### `check_radial_velocity(measurements, rv_tolerance=5.0)`

Estima el desplazamiento radial sistemático a partir de las líneas detectadas.

```python
mean_shift, rv_est, rv_warning = check_radial_velocity(measurements)
```

**Metodología:**
1. Recopilar `λ_obs − λ_lab` para todas las líneas con `quality_flag = 'OK'`
2. Eliminar outliers (2σ) para estimación robusta
3. Convertir a velocidad: `VR = c × Δλ_medio / 4500 Å`
4. Si `|Δλ_medio| > 5 Å`, emitir advertencia

**Nota:** La VR se estima como diagnóstico. El sistema **no corrige** automáticamente el espectro; se asume VR = 0 km/s a menos que el usuario aplique corrección previa.

---

### `process_spectrum_with_diagnostics(wavelengths, flux, ...)`

Función de alto nivel que ejecuta el pipeline completo de preprocesamiento y diagnóstico:

```python
flux_norm, continuum, measurements, snr, rv_info, diag_file = \
    process_spectrum_with_diagnostics(wavelengths, flux,
        output_dir="resultados/",
        object_name="HD123456",
        spectral_type="G", subtype="G2",
        write_diag_file=True)
```

**Pasos internos:**
1. `normalize_to_continuum()` — normalización en dos pasos
2. `measure_diagnostic_lines()` — medición con quality_flags y FWHM adaptativo
3. `compute_snr()` — estimación de SNR
4. `check_radial_velocity()` — diagnóstico de VR
5. `write_diagnostic_file()` — archivo de diagnóstico (opcional)

---

## Referencias

- Gray, R. O., & Corbally, C. J. (2009). *Stellar Spectral Classification*. Princeton University Press.
- Prugniel, P., & Soubiran, C. (2001). *A database of high and medium-resolution stellar spectra*. A&A, 369, 1048.
- Jacoby, G. H., Hunter, D. A., & Christian, C. A. (1984). *A library of stellar spectra*. ApJS, 56, 257.
- Jaschek, C., & Jaschek, M. (1990). *The Classification of Stars*. Cambridge University Press.
- Fraunhofer, J. (1817). Nomenclatura histórica de líneas espectrales solares (letras A–K).
- Kirkpatrick, J. D., Reid, I. N., & Liebert, J. (1999). *Dwarfs Cooler than M: The Definition of Spectral Type L*. ApJ, 519, 802. *(Criterios VO y CaH para tipos M tardíos)*
- Sota, A., Maíz Apellániz, J., et al. (2014). *The Galactic O-Star Spectroscopic Survey (GOSSS) II*. ApJS, 211, 10. *(Criterios modernos para subtipos O; N III, C III, líneas de emisión)*
