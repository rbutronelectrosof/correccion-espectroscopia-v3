# GUIA DEL PROYECTO: Sistema de Clasificacion Espectral Estelar

## 1. DESCRIPCION GENERAL

Este sistema clasifica automaticamente estrellas segun su tipo espectral (O, B, A, F, G, K, M) y clase de luminosidad MK (I-V) analizando sus espectros de luz. Utiliza multiples metodos de clasificacion combinados para lograr una precision del 86.5%. La version 3.0 incorpora 85 lineas diagnostico, 21 razones espectrales, arbol interactivo de 26 nodos, y modulo de luminosidad.

### Tipos Espectrales (de mas caliente a mas fria)
| Tipo | Temperatura | Color | Ejemplo |
|------|-------------|-------|---------|
| O | >30,000 K | Azul | Estrellas masivas |
| B | 10,000-30,000 K | Azul-blanco | Rigel |
| A | 7,500-10,000 K | Blanco | Sirio |
| F | 6,000-7,500 K | Blanco-amarillo | Procion |
| G | 5,200-6,000 K | Amarillo | Sol |
| K | 3,700-5,200 K | Naranja | Arcturo |
| M | <3,700 K | Rojo | Betelgeuse |

---

## 2. ESTRUCTURA DE CARPETAS

```
Sistema de Clasificacion Espectral/
│
├── INSTALAR.bat        → Ejecutar primero para instalar dependencias
├── ENTRENAR.bat        → Entrenar los modelos de Machine Learning
├── INICIAR.bat         → Iniciar la aplicacion web
│
├── src/                → CODIGO FUENTE PRINCIPAL
│   ├── spectral_classification_corrected.py   (Motor: 85 lineas, LINE_WINDOWS, 21 ratios)
│   ├── luminosity_classification.py           (Clasificacion de luminosidad MK I-V)
│   ├── spectral_validation.py                 (Sistema de votacion multi-metodo)
│   ├── neural_classifiers.py                  (Redes neuronales KNN/CNN)
│   ├── train_and_validate.py                  (Entrenamiento arbol)
│   └── train_neural_models.py                 (Entrenamiento KNN/CNN)
│
├── scripts/            → SCRIPTS DE PROCESAMIENTO
│   ├── procesar_una_estrella.py    (Analizar un espectro individual)
│   └── procesar_lote_estrellas.py  (Analizar multiples espectros)
│
├── data/               → DATOS DE ENTRADA
│   ├── elodie/         (Catalogo de ~860 espectros etiquetados)
│   └── espectros/      (Espectros adicionales para analizar)
│
├── models/             → MODELOS ENTRENADOS
│   ├── decision_tree.pkl    (Arbol de decision)
│   ├── knn_model.pkl        (K-Nearest Neighbors)
│   └── cnn_model.h5         (Red neuronal convolucional)
│
├── webapp/             → APLICACION WEB
│   ├── app.py          (Servidor Flask)
│   ├── templates/      (Interfaz HTML)
│   └── static/         (Estilos CSS y JavaScript)
│
└── docs/               → DOCUMENTACION
```

---

## 3. COMO FUNCIONA EL SISTEMA

### Pipeline de Procesamiento

```
    ENTRADA: Archivo de espectro (.txt o .fits)
                    │
                    ▼
    ┌─────────────────────────────────┐
    │  1. NORMALIZACION               │
    │     - Ajuste del continuo       │
    │     - Sigma-clipping            │
    └─────────────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────┐
    │  2. MEDICION DE LINEAS          │
    │     - Detecta 85 lineas         │
    │     - Calcula anchos equiv.     │
    │     - Ventanas adaptativas 6-30A│
    │     - 21 razones diagnostico    │
    └─────────────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────┐
    │  3. CLASIFICACION MULTI-METODO  │
    │                                 │
    │  ┌───────────────────────────┐  │
    │  │ Clasificador Fisico  10% │  │
    │  │ (reglas astronomicas)    │  │
    │  └───────────────────────────┘  │
    │  ┌───────────────────────────┐  │
    │  │ Arbol de Decision   40%  │  │
    │  │ (scikit-learn)           │  │
    │  └───────────────────────────┘  │
    │  ┌───────────────────────────┐  │
    │  │ Template Matching   10%  │  │
    │  │ (comparacion)            │  │
    │  └───────────────────────────┘  │
    │  ┌───────────────────────────┐  │
    │  │ KNN / CNN           40%  │  │
    │  │ (redes neuronales)       │  │
    │  └───────────────────────────┘  │
    │                                 │
    └─────────────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────┐
    │  4. VOTACION PONDERADA          │
    │     - Combina resultados        │
    │     - Calcula confianza         │
    └─────────────────────────────────┘
                    │
                    ▼
                    │
                    ▼
    ┌─────────────────────────────────┐
    │  5. CLASIFICACION LUMINOSIDAD   │
    │     - Sr II, Ba II, CaI/FeI     │
    │     - TiO/CaH para tipo M       │
    │     - Clase I, II, III, IV, V   │
    └─────────────────────────────────┘
                    │
                    ▼
    SALIDA: Tipo MK completo + Confianza (%)
            Ejemplo: "G2V" con 87% confianza
```

---

## 4. LINEAS ESPECTRALES DIAGNOSTICAS (85 LINEAS)

El sistema analiza las siguientes lineas de absorcion:

### Hidrogeno (Serie de Balmer)
| Linea | Longitud de onda | Ventana | Uso |
|-------|------------------|---------|-----|
| H-alpha | 6563 A | 30 A | Todos los tipos |
| H-beta | 4861 A | 20 A | Clasificacion A-F |
| H-gamma | 4341 A | 20 A | Clasificacion A-F |
| H-delta | 4102 A | 20 A | Clasificacion A-F |
| H-epsilon | 3970 A | 16 A | Criterio A/F con Ca II K |

### Helio (Estrellas calientes)
| Linea | Longitud de onda | Uso |
|-------|------------------|-----|
| He I | 4026, 4388, 4471, 4922, 5016, 5876 A | Tipo B; O tardio |
| He II | 4200, 4542, 4686, 5411 A | Tipo O |

### N, C, O ionizados (Tipos O y B)
| Linea | Longitud de onda | Uso |
|-------|------------------|-----|
| N V | 4604, 4620 A | O3-O5 (muy tempranas) |
| N III | 4634, 4641 A | Of; discrimina O7-O8 |
| C III | 4650 A | O tardio |
| O II | 4415, 4417, 4591, 4596 A | B tempranas |

### Silicio (Subtipos B)
| Linea | Longitud de onda | Uso |
|-------|------------------|-----|
| Si IV | 4089, 4116 A | B0-B1 |
| Si III | 4553, 4568, 4575 A | B1-B3 |
| Si II | 4128, 4131, 5041, 5056 A | B5-B9 |

### Calcio (Estrellas frias)
| Linea | Longitud de onda | Uso |
|-------|------------------|-----|
| Ca II K | 3934 A | Criterio primario A/F |
| Ca II H | 3969 A | F-K |
| Ca I | 4227, 4455, 6122, 6162 A | G-K; sensible a luminosidad |

### Metales varios
| Linea | Longitud de onda | Uso |
|-------|------------------|-----|
| Fe I | 4046, 4071, 4144, 4260, 4271, 4326, 4383, 4957, 5270, 5328, 6495 A | G-K-M |
| Fe II | 4173, 4233 A | A tardia |
| Mg I | 4703, 5183 (triplete b) A | F-G |
| Mg II | 4481 A | B tardia |
| Cr I | 4254 A | Criterio G/K decisivo |
| Sr II | 4077, 4216 A | A-F; indicador luminosidad |
| Ba II | 4554 A | K gigante (luminosidad) |
| Na I D | 5896 A | K tardia |
| CH G-band | 4300 A | F5-G0 |

### Bandas moleculares (Tipo M)
| Banda | Longitud de onda | Uso |
|-------|------------------|-----|
| TiO clasico | 4762, 4955, 5167 A | M temprano |
| TiO extendido | 5448, 6158, 6651 A | Discriminacion M enana/gigante |
| VO | 7434, 7865 A | M tardio (M5+) |
| CaH | 6382, 6750 A | M enana vs gigante |

---

## 5. INSTRUCCIONES DE USO

### Paso 1: Instalacion
```
Ejecutar: INSTALAR.bat
```
Esto instala: numpy, scipy, matplotlib, pandas, scikit-learn, flask, astropy

### Paso 2: Entrenamiento (opcional, ya hay modelos)
```
Ejecutar: ENTRENAR.bat
```
Entrena los modelos con el catalogo ELODIE (860 espectros)

### Paso 3: Usar la aplicacion
```
Ejecutar: INICIAR.bat
```
Abre el navegador en http://localhost:5000

### Uso de la interfaz web:
1. Subir archivo de espectro (.txt o .fits)
2. Ver resultado de clasificacion
3. Examinar grafico con lineas detectadas
4. Exportar resultados a CSV

---

## 6. RENDIMIENTO DEL SISTEMA

| Metodo | Precision |
|--------|-----------|
| Arbol de Decision | 84% |
| KNN (k=5) | 89% |
| Sistema combinado | 86.5% |

### Matriz de Confusion (tipos principales)
```
              Predicho
Real      O    B    A    F    G    K    M
  O      95%   5%   -    -    -    -    -
  B       2%  91%   7%   -    -    -    -
  A       -    3%  89%   8%   -    -    -
  F       -    -    5%  82%  13%   -    -
  G       -    -    -   10%  85%   5%   -
  K       -    -    -    -    8%  88%   4%
  M       -    -    -    -    -    5%  95%
```

---

## 7. ARCHIVOS DE DATOS

### Formato de espectros (.txt)
```
# Comentario (opcional)
longitud_onda,flujo
3900.0,0.95
3901.0,0.97
3902.0,0.94
...
```

### Nomenclatura del catalogo ELODIE
```
HD000108_tipoO6pe.txt
│       │    │
│       │    └── Tipo espectral conocido
│       └── Separador
└── Identificador de la estrella (Henry Draper)
```

---

## 8. TECNOLOGIAS UTILIZADAS

| Componente | Tecnologia |
|------------|------------|
| Lenguaje | Python 3.8+ |
| ML | scikit-learn |
| Redes Neuronales | TensorFlow/Keras |
| Servidor Web | Flask |
| Visualizacion | Matplotlib |
| Datos FITS | Astropy |

---

## 9. REFERENCIAS

- Catalogo ELODIE: Observatorio de Haute-Provence
- Clasificacion MK: Morgan-Keenan spectral classification
- Lineas espectrales: NIST Atomic Spectra Database

---

**Autor:** Sistema desarrollado para clasificacion espectral automatizada (SpectroClass)
**Version:** 3.0
**Precision:** 86.5% en votacion multi-metodo
**Lineas diagnostico:** 85 | **Razones:** 21 | **Nodos arbol:** 26
