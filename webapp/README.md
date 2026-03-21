# 🌟 Aplicación Web - Clasificación Espectral Automática

Aplicación web interactiva para clasificar espectros estelares automáticamente, con visualización de líneas diagnóstico, clasificación de luminosidad MK y exportación de resultados.

## 🚀 Características

- ✅ **Subida de archivos múltiples** (.txt, .fits)
- ✅ **Clasificación automática** con 85 líneas diagnóstico y árbol de 26 nodos
- ✅ **Clasificación de luminosidad MK** — tipo completo (ej. G2V, K3III) en ambas pestañas
- ✅ **Visualización interactiva** con zoom en líneas diagnóstico
- ✅ **Exportación de resultados** a CSV, TXT y PDF
- ✅ **Interfaz moderna** con drag & drop
- ✅ **Procesamiento en lote** de múltiples espectros
- ✅ **Árbol interactivo de 26 nodos** con criterios visuales paso a paso
- ✅ **Valores característicos** (anchos equivalentes de todas las líneas y 21 razones diagnóstico)
- ✅ **Pestaña Ayuda** con documentación de todos los criterios en castellano

## 📋 Requisitos

- Python 3.8 o superior
- Navegador web moderno (Chrome, Firefox, Edge, Safari)

## 🔧 Instalación

### 1. Instalar dependencias

En la terminal, navega al directorio `webapp/`:

```bash
cd webapp
pip install -r requirements.txt
```

**Dependencias principales:**
- Flask 3.0.0 (servidor web)
- NumPy 1.24.3 (cálculos numéricos)
- Matplotlib 3.7.2 (visualización)
- SciPy 1.11.1 (procesamiento de señales)
- Pandas 2.0.3 (exportación de datos)
- Astropy 5.3.4 (archivos FITS)

### 2. Verificar estructura de archivos

Asegúrate de que tienes esta estructura:

```
webapp/
├── app.py                  # Aplicación Flask principal
├── requirements.txt        # Dependencias
├── templates/
│   ├── index.html         # Página principal
│   └── info.html          # Página de información
├── static/
│   ├── style.css          # Estilos
│   └── script.js          # JavaScript
├── uploads/               # Archivos temporales (se crea automáticamente)
└── results/               # Resultados generados (se crea automáticamente)
```

### 3. Copiar módulo de clasificación

El archivo `spectral_classification_corrected.py` debe estar en el directorio padre:

```
crreccion espestrocopia/
├── spectral_classification_corrected.py  ← Debe estar aquí
└── webapp/
    └── app.py
```

Si no está ahí, cópialo:

```bash
# Desde el directorio webapp/
cp ../spectral_classification_corrected.py ../
```

## ▶️ Uso

### 1. Iniciar el servidor

```bash
python app.py
```

Verás un mensaje como este:

```
======================================================================
CLASIFICACIÓN ESPECTRAL - Servidor Web
======================================================================
Líneas espectrales configuradas: 85
Directorio de subidas: uploads/
Directorio de resultados: results/

Iniciando servidor en: http://localhost:5000
Presiona Ctrl+C para detener
======================================================================
```

### 2. Abrir en el navegador

Abre tu navegador web y ve a:

```
http://localhost:5000
```

### 3. Subir espectros

Hay dos formas:

**Opción A: Drag & Drop**
1. Arrastra archivos .txt o .fits desde tu explorador de archivos
2. Suelta sobre el área de subida

**Opción B: Seleccionar archivos**
1. Haz clic en "Seleccionar Archivos"
2. Elige uno o varios archivos
3. Haz clic en "Abrir"

### 4. Procesar

1. Verifica que los archivos estén listados
2. Haz clic en "🚀 Procesar Espectros"
3. Espera a que se complete el procesamiento

### 5. Ver resultados

Para cada espectro verás:
- **Tipo espectral clasificado** (O, B, A, F, G, K, M)
- **Subtipo** (ej: O4, B6-B7, A0-A2)
- **Tipo MK completo** (ej: G2V, K3III) con nombre de clase de luminosidad
- **Número de líneas detectadas** (de 85 posibles)
- **Rango de longitudes de onda**
- **Visualización completa** con:
  - Espectro completo con líneas diagnóstico marcadas
  - Gráfico de barras de anchos equivalentes
  - Zoom en regiones diagnóstico
- **Árbol Interactivo**: botón para explorar el espectro nodo a nodo (26 decisiones)

### 6. Exportar resultados

Haz clic en "📥 Exportar CSV" para descargar un archivo CSV con:
- Nombre del archivo
- Objeto
- Tipo original (si está en el nombre del archivo)
- Tipo clasificado
- Subtipo
- Número de líneas detectadas
- Rango de λ
- Fecha de procesamiento

## 📊 Formatos de Archivo Soportados

### Archivos .txt

Formato esperado:
```
wavelength,flux
3900.0,1.234
3900.5,1.245
3901.0,1.256
...
```

**Características:**
- Delimitador: coma (`,`)
- Primera línea: encabezado (se omite)
- Dos columnas: longitud de onda (Å) y flujo

### Archivos .fits

**Características:**
- Headers requeridos:
  - `CRVAL1`: Longitud de onda del píxel de referencia
  - `CRPIX1`: Píxel de referencia
  - `CDELT1`: Incremento de longitud de onda
- Headers opcionales:
  - `OBJECT`: Nombre del objeto
  - `SPTYPE`: Tipo espectral original

## 🎨 Interfaz

### Página Principal

- **Área de subida**: Drag & drop o selección de archivos
- **Lista de archivos**: Muestra archivos seleccionados con opción de eliminar
- **Botones de control**: Procesar y Limpiar
- **Barra de progreso**: Muestra el avance del procesamiento
- **Resultados**: Tarjetas con información y visualización de cada espectro
- **Exportación**: Botón para descargar CSV con todos los resultados

### Página de Información

Accede desde el enlace "Documentación" en el footer, o directamente:

```
http://localhost:5000/info
```

Contiene:
- Problemas críticos corregidos
- Esquema de clasificación
- Líneas espectrales configuradas
- Referencias científicas

## 🔍 Detalles de Líneas

Para cada espectro clasificado:

1. Haz clic en "📋 Ver Líneas Detectadas"
2. Se abre una tabla con:
   - Nombre de la línea
   - Longitud de onda (Å)
   - Ancho equivalente (Å)
   - Profundidad de la línea

## 📸 Visualizaciones

Cada resultado incluye una imagen con múltiples paneles:

**Panel 1: Espectro Completo**
- Espectro normalizado
- Líneas espectrales marcadas
- Continuo de referencia

**Panel 2: Anchos Equivalentes**
- Gráfico de barras de EW
- Código de colores por tipo de línea

**Paneles 3+: Zoom en Regiones Diagnóstico** (automático según tipo)
- **Tipo O**: He II 4686, He I vs He II, N V 4604
- **Tipo B**: He I vs Mg II, Si IV vs Si III vs Si II
- **Tipo A**: H beta, Ca II K vs H gamma
- **Tipos F-G-K**: Ca II K/H, metales

### Ampliar imagen

Haz clic sobre cualquier imagen para verla a pantalla completa.

## 🛠️ Solución de Problemas

### Error: "No module named 'flask'"

**Solución:**
```bash
pip install flask
```

### Error: "No module named 'spectral_classification_corrected'"

**Solución:**
Asegúrate de que `spectral_classification_corrected.py` está en el directorio padre de `webapp/`.

### El servidor no inicia

**Posibles causas:**
1. Puerto 5000 ya en uso
   ```bash
   # Cambiar puerto en app.py línea final:
   app.run(debug=True, host='0.0.0.0', port=5001)
   ```

2. Permisos insuficientes
   ```bash
   # Ejecutar con permisos de administrador (Windows)
   # o usar sudo (Linux/Mac)
   ```

### Archivos no se procesan

**Verificar:**
1. Formato del archivo (.txt o .fits)
2. Estructura del archivo (ver sección "Formatos de Archivo")
3. Tamaño del archivo (< 50 MB)
4. Mensajes de error en la consola del servidor

### Gráficos no se muestran

**Verificar:**
1. Directorio `results/` existe y tiene permisos de escritura
2. Matplotlib está instalado correctamente:
   ```bash
   pip install --upgrade matplotlib
   ```

## 🌐 Acceso Remoto

Para acceder desde otros dispositivos en la red local:

1. Encuentra tu IP local:
   ```bash
   # Windows
   ipconfig

   # Linux/Mac
   ifconfig
   ```

2. El servidor ya está configurado con `host='0.0.0.0'`

3. Desde otro dispositivo, accede a:
   ```
   http://TU_IP:5000
   ```

**Ejemplo:**
```
http://192.168.1.100:5000
```

## 📝 API Endpoints

Si quieres usar la API directamente:

### GET `/health`

Verifica el estado del servidor.

**Respuesta:**
```json
{
  "status": "OK",
  "lines_configured": 85,
  "version": "3.0"
}
```

### POST `/upload`

Sube y procesa espectros.

**Request:**
- Content-Type: `multipart/form-data`
- Campo: `files[]` (uno o varios archivos)

**Respuesta:**
```json
{
  "results": [
    {
      "success": true,
      "filename": "estrella.txt",
      "objeto": "HD015570",
      "tipo_original": "O4",
      "tipo_clasificado": "O",
      "subtipo": "O4",
      "diagnostics": {...},
      "lineas_detectadas": [...],
      "n_lineas": 15,
      "rango_lambda": [3900.0, 6800.0],
      "n_puntos": 5000,
      "plot_path": "results/estrella.txt_plot.png",
      "timestamp": "2025-01-24 15:30:00"
    }
  ],
  "errors": [],
  "n_success": 1,
  "n_errors": 0
}
```

### POST `/api/luminosity`

Estima la clase de luminosidad MK a partir de un espectro y un tipo espectral conocido.

**Request:**
```json
{
  "wavelength": [4000.0, 4000.5, ...],
  "flux": [0.98, 0.97, ...],
  "spectral_type": "G2"
}
```

**Respuesta:**
```json
{
  "success": true,
  "luminosity_class": "V",
  "mk_full": "G2V",
  "lum_name": "Enana (secuencia principal)",
  "indicators": { "CaI_FeI": 0.79, "BaII_FeI": 0.23 }
}
```

### POST `/export_csv`

Exporta resultados a CSV.

**Request:**
```json
{
  "results": [...]  // Array de resultados del endpoint /upload
}
```

**Respuesta:**
Archivo CSV descargable.

## 🔒 Seguridad

**Advertencias:**
- No exponer a Internet sin autenticación
- Solo para uso local o red confiable
- Los archivos subidos se eliminan después del procesamiento
- Tamaño máximo: 50 MB por archivo

## 📚 Documentación Adicional

- **INFORME_PROBLEMAS_CRITICOS.md**: Detalles de correcciones implementadas
- **RESUMEN_MEJORAS.md**: Resumen de mejoras de versión 2.0
- **test_clasificacion_O.md**: Validación de clasificación tipo O

## 🐛 Reportar Problemas

Si encuentras algún error:
1. Anota el mensaje de error completo
2. Guarda el archivo que causó el problema
3. Revisa la consola del servidor para más detalles

## 📄 Licencia

Proyecto educativo para clasificación espectral.

---

**Versión:** 3.0
**Fecha:** Marzo 2026
**Autor:** Sistema de Clasificacion Espectral (SpectroClass)
