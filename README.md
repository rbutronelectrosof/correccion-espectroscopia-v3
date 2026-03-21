<div align="center">

# 🌟 SpectroClass v3.1

### Sistema Automático de Clasificación Espectral Estelar

*Automatic Stellar Spectral Classification System*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Astropy](https://img.shields.io/badge/Astropy-5.0%2B-orange?style=for-the-badge)](https://www.astropy.org/)

**Roberto Butron** · Facultad de Ciencias Exactas y Naturales, UNCuyo · Mendoza, Argentina

[Características](#-características) · [Instalación](#-instalación) · [Uso](#-uso) · [Arquitectura](#-arquitectura) · [API](#-api-rest) · [Referencias](#-referencias)

</div>

---

## ¿Qué es SpectroClass?

SpectroClass es un software de **clasificación espectral estelar automática** que implementa el sistema MK (Morgan–Keenan) mediante un enfoque de votación ponderada entre tres métodos independientes:

1. **Clasificador físico** — árbol jerárquico de 26 nodos basado en criterios astrofísicos (Gray & Corbally 2009)
2. **Árbol de decisión ML** — scikit-learn entrenado con el catálogo ELODIE (856 espectros)
3. **Redes neuronales** — KNN y CNN-1D sobre espectros remuestreados a grilla uniforme

El resultado es el **tipo espectral completo** (ej. `G2V`, `K3III`, `B2Ia`) con clase de luminosidad MK (I–V), confianza porcentual y hasta tres alternativas con justificación diagnóstica trazable.

> Presentado en: *Congreso de Evolución Estelar, Exoplanetas y Dinámica de Sistemas Estelares* — Abril 2026

---

## ✨ Características

### Clasificación
- **Secuencia completa OBAFGKM** — tipos O2 hasta M9, con nodos especializados para O3/O4, O tardío, M tardío con VO+CaH
- **Clase de luminosidad MK** — Ia, Ib, II, III, IV, V con indicadores sensibles a gravedad superficial
- **85 líneas diagnóstico** medidas por ancho equivalente (integración trapezoidal, ventanas adaptativas 6–30 Å)
- **21 razones diagnóstico** que cubren toda la secuencia: ionización (He I/He II, Si III/Si II), temperatura (Ca II K/Hε, Cr I/Fe I) y gravedad (Sr II/Fe I, Ba II/Fe I, TiO/CaH, Y II/Fe I)
- **Normalización científica al continuo** con rechazo de rayos cósmicos por sigma-clipping MAD

### Interfaz Web
- Carga de archivos `.txt` y `.fits` (con calibración WCS automática)
- Visualización interactiva con zoom en líneas diagnóstico
- **Árbol interactivo de clasificación** paso a paso (26 nodos, ~6 preguntas)
- Procesamiento por lotes con exportación a CSV, TXT y PDF
- Tablas MK completas integradas en la ayuda (tipos O–M, temperatura y luminosidad)

### Modelos Pre-entrenados
| Modelo | Tipo | Tamaño | Accuracy |
|--------|------|--------|----------|
| `decision_tree.pkl` | scikit-learn | 4 MB | Ver `validation_report.txt` |
| `knn_model.pkl` | K-Nearest Neighbors | 116 KB | — |
| `cnn_model.h5` | CNN-1D TensorFlow | 25 MB | — |

---

## 🏗️ Arquitectura

```
Espectro (.txt / .fits)
         │
         ▼
  normalize_to_continuum()          ← Continuo local iterativo + sigma-clip
         │
         ▼
  measure_diagnostic_lines()        ← 85 líneas · ventanas LINE_WINDOWS
         │
         ▼
  compute_spectral_ratios()         ← 21 razones de ionización, T y gravedad
         │
    ┌────┴────────────────┐
    │                     │
    ▼                     ▼
classify_star_corrected() SpectralValidator (ensemble)
  (árbol 26 nodos)          ├─ Físico
                            ├─ DecisionTree ML
                            └─ KNN / CNN-1D
                                     │
                                     ▼
                          estimate_luminosity_class()
                                     │
                                     ▼
                          Tipo MK + Luminosidad + Confianza
```

### Líneas diagnóstico incluidas

| Grupo | Líneas |
|-------|--------|
| Balmer | Hα, Hβ, Hγ, Hδ, Hε (ventanas 16–30 Å) |
| He I | λ3820, 4009, 4026, 4121, 4143, 4388, 4471, 4713, 4922, 5876, 6678, 7065 |
| He II | λ4200, 4542, 4686 |
| Si IV/III/II | λ4089, 4116, 4121, 4128, 4131, 4553, 4568, 4575 |
| N III/V, C III, O II | λ4603-4641, λ4647-4652, λ4070-4076, 4348 |
| Ca II (H&K), Ca I, Na I D | λ3933, 3968, 4227, 5890, 5896 |
| Fe I, Fe II, Cr I, Ti II | λ4046, 4144, 4250, 4271, 4325, 4383 · λ4178, 4233, 4399, 4444 · λ4254, 4275, 4290 |
| Sr II, Ba II, Y II | λ4077, 4215 · λ4554 · λ4376 |
| CH G-band, Mg I b, Mg II | λ4300, 5167, 5183 · λ4481 |
| TiO | λ5167, 5759, 6159, 7054, 7666, 7786 |
| VO, CaH, MgH, CN | λ5736, 7434, 7865 · λ6382, 6750, 6908, 6946 · λ4770 · λ3883, 4215 |

---

## 📁 Estructura del Proyecto

```
SpectroClass/
├── webapp/                          # Aplicación web Flask
│   ├── app.py                       # Servidor Flask + API REST (7 endpoints)
│   ├── templates/
│   │   └── index.html               # UI con 7 pestañas + árbol interactivo
│   ├── static/
│   │   ├── script.js                # Árbol 26 nodos + visualización
│   │   └── style.css                # Diseño responsivo + badges MK
│   ├── results/                     # Plots generados (ignorados en git)
│   └── uploads/                     # Archivos temporales (ignorados en git)
│
├── src/                             # Módulos Python principales
│   ├── spectral_classification_corrected.py  # Motor físico (85 líneas)
│   ├── luminosity_classification.py          # Luminosidad MK (I–V)
│   ├── spectral_validation.py                # Ensemble multi-método
│   ├── neural_classifiers.py                 # KNN / CNN-1D
│   ├── train_neural_models.py                # Entrenamiento KNN/CNN
│   └── train_and_validate.py                 # Entrenamiento árbol ML
│
├── scripts/                         # Scripts de línea de comandos
│   ├── procesar_una_estrella.py     # Procesar espectro individual con PDF
│   └── procesar_lote_estrellas.py   # Procesamiento por lotes
│
├── data/
│   ├── elodie/                      # Catálogo ELODIE (856 espectros)
│   └── espectros/                   # Espectros de referencia por tipo
│
├── models/                          # Modelos entrenados
│   ├── decision_tree.pkl            # Árbol ML (4 MB)
│   ├── knn_model.pkl                # KNN (116 KB)
│   ├── cnn_model.h5                 # CNN-1D (25 MB, opcional)
│   └── confusion_matrix.png         # Matriz de confusión del árbol ML
│
├── docs/                            # Documentación técnica
│   ├── abstract_SpectroClass.md     # Abstract (ES/EN) para congreso 2026
│   ├── METODOS_CLASIFICACION.md     # Descripción de métodos
│   └── INFORME_SISTEMA.md           # Informe técnico completo
│
├── results/                         # Salidas de procesamiento por lotes
├── requirements.txt                 # Dependencias base
├── requirements_con_tensorflow.txt  # Con TensorFlow para CNN
├── INSTALAR.bat                     # Instalador automático Windows
├── iniciar.bat                      # Iniciador Windows
└── LICENSE                          # MIT
```

---

## 🚀 Instalación

### Requisitos previos
- Python **3.10** o superior
- Git

### Instalación rápida (Windows)

```bat
git clone https://github.com/tu-usuario/SpectroClass.git
cd SpectroClass
INSTALAR.bat
```

### Instalación manual (Windows / Linux / macOS)

```bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/SpectroClass.git
cd SpectroClass

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Instalar dependencias base
pip install -r requirements.txt

# 4. (Opcional) Con soporte CNN/TensorFlow
pip install -r requirements_con_tensorflow.txt
```

### Dependencias

| Paquete | Versión | Uso |
|---------|---------|-----|
| `numpy` | ≥1.21 | Cómputo numérico |
| `scipy` | ≥1.7 | Procesamiento de señales |
| `pandas` | ≥1.3 | Manejo de datos |
| `matplotlib` | ≥3.4 | Visualización |
| `scikit-learn` | ≥1.0 | Árbol de decisión ML |
| `Flask` | ≥2.0 | Servidor web |
| `astropy` | ≥5.0 | Lectura de archivos FITS |
| `tensorflow` | ≥2.10 | CNN-1D *(opcional)* |

---

## 🔭 Uso

### Aplicación web

```bash
# Windows
iniciar.bat

# Linux / macOS
python webapp/app.py
```

Abrir en el navegador: **http://localhost:5000**

La interfaz tiene 7 pestañas:

| Pestaña | Descripción |
|---------|-------------|
| 📊 Análisis Detallado | Tabla completa de 85 líneas con EW medidos |
| 🔭 Clasificador Espectral | Carga de archivo → resultado completo |
| 📁 Procesamiento por Lote | Múltiples espectros → CSV/PDF |
| 🛠️ Herramientas | Entrenamiento de modelos |
| 🧠 Redes Neuronales | Selección y predicción neural |
| 🌿 Árbol Interactivo | Clasificación visual paso a paso (26 nodos) |
| ❓ Ayuda | Guía + tablas MK completas integradas |

### Uso programático (Python)

```python
import sys
sys.path.insert(0, 'src')

from spectral_classification_corrected import (
    normalize_to_continuum,
    measure_diagnostic_lines,
    classify_star_corrected,
    compute_spectral_ratios,
)
from luminosity_classification import estimate_luminosity_class

import numpy as np

# Cargar espectro
data = np.loadtxt('mi_espectro.txt', delimiter=',', skiprows=1)
wavelengths, flux = data[:, 0], data[:, 1]

# Normalizar al continuo (con sigma-clipping)
flux_norm, continuum = normalize_to_continuum(wavelengths, flux)

# Medir anchos equivalentes (85 líneas)
measurements = measure_diagnostic_lines(wavelengths, flux_norm)

# Calcular razones diagnóstico
ratios = compute_spectral_ratios(measurements)

# Clasificar tipo espectral
tipo, subtipo, diagnostics = classify_star_corrected(measurements)

# Estimar clase de luminosidad
lum_class = estimate_luminosity_class(tipo, measurements, ratios)

print(f"Clasificación: {tipo}{subtipo}{lum_class}")
# Ejemplo: G2V
```

### Procesamiento por lotes (CLI)

```bash
# Un espectro con reporte PDF detallado
python scripts/procesar_una_estrella.py mi_espectro.fits

# Directorio completo
python scripts/procesar_lote_estrellas.py data/mis_espectros/ --output results/
```

---

## 📡 API REST

La aplicación expone los siguientes endpoints:

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/upload` | Clasificar un espectro → tipo + luminosidad + ratios + plot |
| `POST` | `/batch_upload` | Procesar múltiples espectros |
| `POST` | `/api/luminosity` | Estimar clase de luminosidad independientemente |
| `POST` | `/export_csv` | Exportar resultados a CSV |
| `POST` | `/train_tree` | Entrenar árbol de decisión ML |
| `POST` | `/train_neural` | Entrenar KNN / CNN-1D |
| `GET`  | `/health` | Estado del servidor |

### Ejemplo de llamada

```bash
curl -X POST http://localhost:5000/upload \
  -F "file=@HD001835_G3V.txt" \
  | python -m json.tool
```

```json
{
  "tipo_espectral": "G",
  "subtipo": "2",
  "clase_luminosidad": "V",
  "clasificacion_completa": "G2V",
  "confianza": 87.4,
  "alternativas": ["G3V", "G1V", "G2IV"],
  "diagnostics": { ... }
}
```

---

## 📋 Formato de Datos

### Archivos `.txt`

```
wavelength,flux
3900.0,0.9234
3900.5,0.9245
4000.0,0.8721
```

- Primera columna: longitud de onda en Ångströms
- Segunda columna: flujo (normalizado o arbitrario)
- Separador: coma, espacio o tabulador
- Primera fila puede ser encabezado (se detecta automáticamente)

### Archivos `.fits`

Requiere headers WCS: `CRVAL1`, `CDELT1`, `CRPIX1`

El sistema reconstruye la calibración en longitud de onda automáticamente desde la solución WCS almacenada en la cabecera FITS.

---

## 🧪 Entrenamiento de Modelos

```bash
# Entrenar árbol de decisión ML
python src/train_and_validate.py

# Entrenar KNN y CNN-1D
python src/train_neural_models.py

# O desde la interfaz web → pestaña Herramientas
```

Los modelos entrenados se guardan automáticamente en `models/`.

---

## 🌿 Árbol Interactivo — 26 Nodos

El árbol replica el flujo clásico de clasificación visual, con nodos especializados:

```
inicio
├── ¿He II presente? (λ4542, λ4686)
│   ├── SÍ → tipoO
│   │   ├── ¿N V 4603 fuerte? → O3/O4
│   │   ├── ¿He I > He II? → O7-O9 (He I λ4713 vs He II λ4686)
│   │   └── ¿N III 4634 visible? → O tardío
│   └── NO → ¿He I sin He II?
│       ├── SÍ → tipoB
│       │   ├── ¿Si IV/He I > 1? → B0-B1 (Si IV λ4116 / He I λ4121)
│       │   ├── ¿O II 4070-4076? → B1-B4 luminosidad
│       │   └── ¿Mg II 4481 > He I 4471? → B8-B9
│       └── NO → ¿Balmer máximo?
│           ├── SÍ → tipoA (Ca II K vs Hε)
│           └── NO → ¿Ca II K ~ Hε?
│               ├── SÍ → tipoF
│               │   └── ¿CH G-band 4300? → F5-G0
│               └── NO → ¿muchos metales?
│                   ├── tipoG (Y II λ4376, Mg b λ5183)
│                   ├── tipoK (Na I D λ5896, Cr I λ4254/4275/4290)
│                   └── tipoM (TiO + VO + CaH + MgH)
```

---

## 📊 Catálogos Incluidos

| Catálogo | Nº espectros | Uso |
|----------|-------------|-----|
| ELODIE (Prugniel & Soubiran 2001) | 856 | Entrenamiento ML y KNN |
| Espectros de referencia | ~50 | Template matching |

---

## 📖 Referencias

- Gray, R. O., & Corbally, C. J. (2009). *Stellar Spectral Classification*. Princeton University Press.
- Prugniel, P., & Soubiran, C. (2001). *A database of high and medium-resolution stellar spectra* (ELODIE). A&A, 369, 1048.
- Sota, A., et al. (2014). *The Galactic O-Star Spectroscopic Survey (GOSSS) II*. ApJS, 211, 10.
- Jacoby, G. H., Hunter, D. A., & Christian, C. A. (1984). *A library of stellar spectra*. ApJS, 56, 257.
- Kirkpatrick, J. D., Reid, I. N., & Liebert, J. (1999). *Dwarfs cooler than M*. ApJ, 519, 802.
- Morgan, W. W., Keenan, P. C., & Kellman, E. (1943). *An Atlas of Stellar Spectra*. University of Chicago Press.

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Haz un fork del repositorio
2. Crea una rama para tu feature: `git checkout -b feature/nueva-linea-diagnostico`
3. Realiza tus cambios y haz commit: `git commit -m 'Agregar Y II λ4376 como criterio de luminosidad G-K'`
4. Sube la rama: `git push origin feature/nueva-linea-diagnostico`
5. Abre un Pull Request describiendo los cambios

### Áreas de mejora sugeridas

- [ ] Soporte para espectros de baja resolución (R < 1000)
- [ ] Clasificación de estrellas peculiares (Am, Ap, Ba)
- [ ] Exportación a VO-Table (formato estándar IVOA)
- [ ] Tests unitarios para los módulos `src/`
- [ ] Integración con bases de datos online (SIMBAD, VizieR)

---

## 📄 Licencia

Distribuido bajo licencia **MIT**. Ver archivo [LICENSE](LICENSE) para más información.

---

<div align="center">

Desarrollado en la **Facultad de Ciencias Exactas y Naturales, UNCuyo**
Mendoza, Argentina · 2025–2026

*"Classification is the art of making distinctions." — W. W. Morgan*

</div>
