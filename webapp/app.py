#!/usr/bin/env python3
"""
APLICACIÓN WEB - Clasificación Espectral Interactiva
====================================================

Permite subir espectros (.txt o .fits) y obtener:
- Clasificación automática
- Visualización con zoom en líneas diagnóstico
- Valores característicos (anchos equivalentes)
- Exportación de resultados en PDF/CSV

Uso:
    python app.py

Luego abrir navegador en: http://localhost:5000
"""

from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import os
import sys
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para servidor
import matplotlib.pyplot as plt
import pandas as pd
from werkzeug.utils import secure_filename
import io
import base64
from datetime import datetime

# Importar módulos de clasificación (desde src/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))
from spectral_classification_corrected import (
    normalize_to_continuum,
    measure_diagnostic_lines,
    classify_star_corrected,
    plot_spectrum_corrected,
    SPECTRAL_LINES
)
try:
    from luminosity_classification import (
        estimate_luminosity_class,
        combine_spectral_and_luminosity,
    )
    _LUM_AVAILABLE = True
except ImportError:
    _LUM_AVAILABLE = False


def convert_numpy_types(obj):
    """
    Convierte tipos numpy a tipos Python nativos para serialización JSON.
    Maneja recursivamente diccionarios y listas.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    else:
        return obj

# Importar validador multi-método (NUEVO)
try:
    from spectral_validation import SpectralValidator
    MULTI_METHOD_AVAILABLE = True
except ImportError:
    MULTI_METHOD_AVAILABLE = False
    print("[!] spectral_validation.py no disponible. Solo se usara el clasificador fisico.")

# Configuración de Flask
app = Flask(__name__)
# Usar rutas absolutas para que send_file y os.path coincidan
_webapp_dir = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER']  = os.path.join(_webapp_dir, 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(_webapp_dir, 'results')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'fits', 'fit'}

# Crear directorios si no existen
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Verifica si la extensión del archivo está permitida"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_spectrum_file(filepath):
    """
    Carga un espectro desde archivo .txt o .fits

    Returns
    -------
    wavelengths, flux, error, metadata
    """
    ext = filepath.rsplit('.', 1)[1].lower()

    if ext == 'txt':
        try:
            # Detectar delimitador automáticamente (probar diferentes encodings)
            first_lines = []
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        first_lines = [f.readline() for _ in range(5)]
                    break
                except:
                    continue

            # Buscar línea con datos numéricos
            data_line = None
            delimiter = None  # Por defecto: whitespace
            for line in first_lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Verificar si tiene números
                    parts_tab = line.split('\t')
                    parts_comma = line.split(',')
                    parts_space = line.split()

                    # Intentar detectar delimitador
                    if len(parts_tab) >= 2:
                        try:
                            float(parts_tab[0])
                            data_line = line
                            delimiter = '\t'
                            break
                        except:
                            pass
                    if len(parts_comma) >= 2:
                        try:
                            float(parts_comma[0])
                            data_line = line
                            delimiter = ','
                            break
                        except:
                            pass
                    if len(parts_space) >= 2:
                        try:
                            float(parts_space[0])
                            data_line = line
                            delimiter = None  # whitespace
                            break
                        except:
                            pass

            # Cargar datos con delimitador detectado
            try:
                if delimiter == '\t':
                    data = np.loadtxt(filepath, delimiter='\t', comments='#', encoding='latin-1')
                elif delimiter == ',':
                    # Intentar con skiprows=1 primero (tiene header)
                    try:
                        data = np.loadtxt(filepath, delimiter=',', skiprows=1, encoding='latin-1')
                    except:
                        data = np.loadtxt(filepath, delimiter=',', comments='#', encoding='latin-1')
                else:
                    # Whitespace delimited
                    data = np.loadtxt(filepath, comments='#', encoding='latin-1')
            except Exception as load_error:
                # Fallback: intentar con pandas que es más flexible
                import pandas as pd
                try:
                    df = pd.read_csv(filepath, sep=None, engine='python', comment='#', header=None, encoding='latin-1')
                    data = df.values
                except:
                    raise load_error

            wavelengths = data[:, 0]
            flux = data[:, 1]

            # Extraer tipo original del nombre si está presente
            filename = os.path.basename(filepath)
            tipo_original = 'N/A'
            if '_tipo' in filename.lower():
                # Formato: HD123456_tipoB6V.txt
                parts = filename.lower().split('_tipo')
                if len(parts) > 1:
                    tipo_original = parts[1].replace('.txt', '').upper()

            metadata = {
                'objeto': filename.split('_')[0],
                'tipo_original': tipo_original
            }
            return wavelengths, flux, None, metadata
        except Exception as e:
            return None, None, f"Error al leer archivo TXT: {str(e)}", None

    elif ext in ['fits', 'fit']:
        try:
            from astropy.io import fits
            hdul = fits.open(filepath)
            data = hdul[0].data
            header = hdul[0].header
            hdul.close()

            # Calcular eje de longitud de onda
            crval1 = header.get('CRVAL1', 0)
            crpix1 = header.get('CRPIX1', 1)
            cdelt1 = header.get('CDELT1', 1)

            num_pixels = len(data)
            pixels = np.arange(1, num_pixels + 1)
            wavelengths = crval1 + (pixels - crpix1) * cdelt1

            # Cabecera FITS completa (todas las tarjetas)
            fits_header = []
            for card in header.cards:
                kw = str(card.keyword).strip()
                if not kw:
                    continue
                fits_header.append({
                    'clave':      kw,
                    'valor':      str(card.value).strip(),
                    'comentario': str(card.comment).strip()
                })

            metadata = {
                'objeto':       header.get('OBJECT',   'Desconocido'),
                'tipo_original':header.get('SPTYPE',   'N/A'),
                'fits_header':  fits_header,
                'file_format':  'FITS'
            }

            return wavelengths, data, None, metadata
        except Exception as e:
            return None, None, f"Error al leer archivo FITS: {str(e)}", None
    else:
        return None, None, "Formato no soportado", None


def process_spectrum(filepath, filename, use_multi_method=True,
                     include_neural=True, neural_weight=0.40, preferred_neural='auto'):
    """
    Procesa un espectro completo: carga, normaliza, mide, clasifica

    Parameters
    ----------
    filepath : str
        Ruta al archivo
    filename : str
        Nombre del archivo
    use_multi_method : bool
        Si True, usar validación multi-método
        Si False, solo clasificador físico
    include_neural : bool
        Si incluir el modelo neural en la votación
    neural_weight : float
        Peso del modelo neural (0.0–0.5)
    preferred_neural : str
        'auto', 'knn' o 'cnn_1d'

    Returns
    -------
    dict con resultados (incluye confianza y alternativas si multi_method activo)
    """
    # Cargar
    wavelengths, flux, error, metadata = load_spectrum_file(filepath)
    if error:
        return {'error': error}

    # Construir pesos personalizados según configuración del usuario
    if include_neural and neural_weight > 0:
        remaining = 1.0 - neural_weight
        custom_weights = {
            'physical':         round((0.10 / 0.60) * remaining, 4),
            'decision_tree':    round((0.40 / 0.60) * remaining, 4),
            'template_matching':round((0.10 / 0.60) * remaining, 4),
            'neural':           neural_weight
        }
    else:
        custom_weights = {
            'physical': 0.15,
            'decision_tree': 0.70,
            'template_matching': 0.15
        }

    # Valores por defecto (se sobreescriben en los bloques siguientes)
    tipo_fisico    = None
    subtipo_fisico = None

    # Usar sistema multi-método si está disponible
    if MULTI_METHOD_AVAILABLE and use_multi_method:
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(project_root, 'models')
            validator = SpectralValidator(
                models_dir=models_dir,
                weights=custom_weights,
                use_neural=include_neural,
                preferred_neural=preferred_neural
            )
            result_multimethod = validator.classify(wavelengths, flux, verbose=False)

            # Extraer resultados
            spectral_type  = result_multimethod['tipo_final']
            subtype        = result_multimethod['subtipo_final']   # coherente con tipo_final
            tipo_fisico    = result_multimethod.get('tipo_fisico', spectral_type)
            subtipo_fisico = result_multimethod.get('subtipo_fisico', subtype)
            confianza      = result_multimethod['confianza']
            alternativas   = result_multimethod['alternativas']
            measurements   = result_multimethod['measurements']
            flux_normalized = result_multimethod.get('flux_normalized')  # Si lo incluimos

            # Si flux_normalized no está en result, normalizar manualmente
            if flux_normalized is None:
                flux_normalized, continuum = normalize_to_continuum(wavelengths, flux)

            diagnostics = result_multimethod['detalles']['physical']['diagnostics']

        except Exception as e:
            import traceback
            print(f"[!] Error en clasificacion multi-metodo: {e}")
            print("   Usando solo clasificador físico como fallback")
            traceback.print_exc()
            use_multi_method = False  # Fallback a método simple

    # Clasificación simple (físico solamente)
    if not use_multi_method or not MULTI_METHOD_AVAILABLE:
        try:
            # Normalizar
            flux_normalized, continuum = normalize_to_continuum(wavelengths, flux)

            # Medir líneas
            measurements = measure_diagnostic_lines(wavelengths, flux_normalized)

            # Clasificar
            spectral_type, subtype, diagnostics = classify_star_corrected(
                measurements, wavelengths, flux_normalized
            )

            # Sin métricas de confianza ni alternativas en modo simple
            confianza = None
            alternativas = []

        except Exception as e:
            import traceback
            print("\n" + "="*80)
            print("ERROR EN CLASIFICACIÓN:")
            print("="*80)
            traceback.print_exc()
            print("="*80 + "\n")
            return {'error': f"Error en clasificación: {str(e)}"}

    # Generar visualización
    try:
        fig_path = os.path.join(app.config['RESULTS_FOLDER'], f"{filename}_plot.png")
        plot_spectrum_corrected(
            wavelengths, flux_normalized, measurements,
            spectral_type, subtype,
            metadata['objeto'], metadata['tipo_original'],
            save_path=fig_path
        )
    except Exception as e:
        return {'error': f"Error generando gráfico: {str(e)}"}

    # Preparar líneas detectadas (EW significativo) y medición completa
    detected_lines = []
    todas_lineas   = []
    for line_name, data in measurements.items():
        entry = {
            'nombre':           line_name.replace('_', ' '),
            'longitud_onda':    data['wavelength'],
            'ancho_equivalente':round(data['ew'],    3),
            'profundidad':      round(data['depth'], 3)
        }
        todas_lineas.append(entry)
        if data['ew'] > 0.05:
            detected_lines.append(entry)

    # Ordenar por EW decreciente
    detected_lines.sort(key=lambda x: x['ancho_equivalente'], reverse=True)
    todas_lineas.sort(key=lambda x: x['longitud_onda'])

    # Datos de espectro para visualización SVG (muestreo a máx 2000 puntos)
    try:
        _wav  = np.asarray(wavelengths, dtype=float)
        _flux = np.nan_to_num(np.asarray(flux_normalized, dtype=float), nan=1.0, posinf=1.0, neginf=0.0)
        _samp = max(1, len(_wav) // 2000)
        spectrum_data = {
            'wavelength': _wav[::_samp].tolist(),
            'flux':       _flux[::_samp].tolist(),
            'wmin': float(_wav.min()),
            'wmax': float(_wav.max())
        }
    except Exception:
        spectrum_data = None

    # ── Clase de luminosidad MK ──────────────────────────────────────────
    # Garantizar que luminosity_class, mk_full y lum_name estén presentes
    # (pueden venir ya en diagnostics si classify_star_corrected los añadió).
    _lum_names_map = {
        'Ia':  'Supergigante muy luminosa',
        'Ib':  'Supergigante',
        'II':  'Gigante brillante',
        'III': 'Gigante',
        'IV':  'Subgigante',
        'V':   'Secuencia principal (enana)',
    }

    luminosity_class = diagnostics.get('luminosity_class', '')
    mk_full          = diagnostics.get('mk_full', '')
    lum_name         = diagnostics.get('lum_name', '')

    # Calcular si no vino en diagnostics
    if not luminosity_class and _LUM_AVAILABLE:
        try:
            luminosity_class = estimate_luminosity_class(measurements, spectral_type)
            mk_full          = combine_spectral_and_luminosity(
                subtype if subtype else spectral_type, luminosity_class
            )
        except Exception:
            luminosity_class = 'V'
            mk_full = (subtype or spectral_type) + 'V'

    if not lum_name:
        lum_name = _lum_names_map.get(luminosity_class, luminosity_class)

    # Asegurar que diagnostics también los tiene (para la UI)
    diagnostics['luminosity_class'] = luminosity_class
    diagnostics['mk_full']          = mk_full
    diagnostics['lum_name']         = lum_name

    # Construir resultado
    result = {
        'success': True,
        'filename': filename,
        'objeto': metadata['objeto'],
        'tipo_original': metadata['tipo_original'],
        'tipo_clasificado': spectral_type,
        'subtipo':          subtype,
        'tipo_fisico':      tipo_fisico    or spectral_type,
        'subtipo_fisico':   subtipo_fisico or subtype,
        # ── Luminosidad MK — accesibles desde el nivel raíz del resultado ──
        'luminosity_class': luminosity_class,
        'mk_full':          mk_full,
        'lum_name':         lum_name,
        # ───────────────────────────────────────────────────────────────────
        'diagnostics': diagnostics,
        'lineas_detectadas': detected_lines,
        'todas_lineas':      todas_lineas,
        'spectrum_data':     spectrum_data,
        'fits_header':       metadata.get('fits_header', None),
        'file_format':       metadata.get('file_format', 'TXT'),
        'n_lineas': len(detected_lines),
        'rango_lambda': [round(wavelengths[0], 1), round(wavelengths[-1], 1)],
        'n_puntos': len(wavelengths),
        'plot_path': fig_path,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'multi_method_used': use_multi_method and MULTI_METHOD_AVAILABLE
    }

    # Agregar métricas adicionales si multi-método activo
    if use_multi_method and MULTI_METHOD_AVAILABLE:
        result['confianza'] = round(confianza, 1)
        result['alternativas'] = alternativas

    # Convertir tipos numpy a tipos Python nativos para JSON
    return convert_numpy_types(result)


@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Procesa archivo(s) subido(s)"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No se encontraron archivos'}), 400

    files = request.files.getlist('files[]')

    if len(files) == 0:
        return jsonify({'error': 'No se seleccionaron archivos'}), 400

    # Parámetros de votación neural enviados desde el frontend
    include_neural  = request.form.get('include_neural', '1') == '1'
    neural_weight   = float(request.form.get('neural_weight', 0.40))
    neural_model    = request.form.get('neural_model', 'auto')   # 'auto', 'knn', 'cnn_1d'

    results = []
    errors = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Procesar
            result = process_spectrum(
                filepath, filename,
                include_neural=include_neural,
                neural_weight=neural_weight,
                preferred_neural=neural_model
            )

            if 'error' in result:
                errors.append({'file': filename, 'error': result['error']})
            else:
                results.append(result)

            # Limpiar archivo subido
            try:
                os.remove(filepath)
            except:
                pass
        else:
            errors.append({'file': file.filename, 'error': 'Formato no permitido'})

    return jsonify({
        'results': results,
        'errors': errors,
        'n_success': len(results),
        'n_errors': len(errors)
    })


@app.route('/upload_single', methods=['POST'])
def upload_single():
    """Sube un archivo al servidor y devuelve su ruta absoluta para usarla en test_spectrum_advanced."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No se recibió ningún archivo'}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'success': False, 'error': 'Archivo vacío'}), 400

        if not allowed_file(file.filename):
            ext = file.filename.rsplit('.', 1)[-1] if '.' in file.filename else 'sin extensión'
            return jsonify({'success': False, 'error': f'Formato .{ext} no permitido. Usa .txt, .fits o .fit'}), 400

        filename = secure_filename(file.filename)
        if not filename:
            filename = 'espectro_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.txt'

        # Usar ruta absoluta para evitar problemas con CWD
        upload_dir = os.path.join(project_root, 'webapp', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)

        return jsonify({'success': True, 'server_path': filepath, 'filename': filename})

    except Exception as e:
        return jsonify({'success': False, 'error': f'Error al guardar archivo: {str(e)}'}), 500


@app.route('/result/<filename>')
def show_result(filename):
    """Muestra resultado de clasificación"""
    # Buscar archivo de resultado guardado
    # (Aquí podrías implementar caché de resultados si lo necesitas)
    return render_template('result.html', filename=filename)


@app.route('/plot/<filename>')
def get_plot(filename):
    """Devuelve el gráfico generado"""
    # RESULTS_FOLDER ya es ruta absoluta; send_file necesita ruta absoluta
    plot_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if os.path.isfile(plot_path):
        return send_file(plot_path, mimetype='image/png')
    return jsonify({'error': f'Grafico no encontrado: {filename}'}), 404


@app.route('/export_csv', methods=['POST'])
def export_csv():
    """Exporta resultados a CSV"""
    data = request.json

    if not data or 'results' not in data:
        return jsonify({'error': 'No hay datos para exportar'}), 400

    results = data['results']

    # Crear DataFrame
    rows = []
    for r in results:
        rows.append({
            'Archivo': r['filename'],
            'Objeto': r['objeto'],
            'Tipo Original': r['tipo_original'],
            'Tipo Clasificado': r['tipo_clasificado'],
            'Subtipo': r['subtipo'],
            'Líneas Detectadas': r['n_lineas'],
            'Rango λ (Å)': f"{r['rango_lambda'][0]}-{r['rango_lambda'][1]}",
            'Puntos Espectrales': r['n_puntos'],
            'Fecha Procesamiento': r['timestamp']
        })

    df = pd.DataFrame(rows)

    # Guardar en memoria
    output = io.StringIO()
    df.to_csv(output, index=False, encoding='utf-8')
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'clasificacion_espectral_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )


@app.route('/export_detailed_csv/<filename>')
def export_detailed_csv(filename):
    """Exporta mediciones detalladas de líneas a CSV"""
    # Recargar resultados desde caché si es necesario
    # Por ahora, devolver error
    return jsonify({'error': 'Función no implementada aún'}), 501


@app.route('/info')
def info():
    """Página con información sobre el método"""
    return render_template('info.html')


@app.route('/health')
def health():
    """Endpoint de salud para verificar que el servidor está corriendo"""
    return jsonify({
        'status': 'OK',
        'lines_configured': len(SPECTRAL_LINES),
        'version': '3.0'
    })


@app.route('/run_script', methods=['POST'])
def run_script():
    """Ejecuta un script .bat desde la webapp"""
    data = request.json
    script_name = data.get('script')

    # Scripts permitidos (por seguridad)
    allowed_scripts = {
        '1_INSTALAR_DEPENDENCIAS': '1_INSTALAR_DEPENDENCIAS.bat',
        '2_ENTRENAR_MODELOS': '2_ENTRENAR_MODELOS.bat',
        '4_TEST_ESPECTRO': '4_TEST_ESPECTRO.bat',
        '5_VER_METRICAS': '5_VER_METRICAS.bat',
    }

    if script_name not in allowed_scripts:
        return jsonify({
            'success': False,
            'error': f'Script no permitido: {script_name}'
        }), 400

    # Ruta al script
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(project_root, allowed_scripts[script_name])

    if not os.path.exists(script_path):
        return jsonify({
            'success': False,
            'error': f'Script no encontrado: {script_path}'
        }), 404

    try:
        # Determinar si estamos en Windows o WSL
        import platform
        utf8_env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        if platform.system() == 'Windows':
            # Windows nativo
            result = subprocess.run(
                ['cmd', '/c', script_path],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=600,  # 10 minutos máximo
                cwd=project_root,
                env=utf8_env
            )
        else:
            # WSL o Linux - ejecutar .bat a través de cmd.exe
            # Convertir ruta WSL a Windows si es necesario
            win_path = script_path.replace('/mnt/c/', 'C:\\').replace('/', '\\')
            result = subprocess.run(
                ['cmd.exe', '/c', win_path],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=600,
                cwd=project_root,
                env=utf8_env
            )

        return jsonify({
            'success': result.returncode == 0,
            'output': result.stdout,
            'error': result.stderr if result.returncode != 0 else None,
            'return_code': result.returncode
        })

    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'error': 'El script excedió el tiempo límite (10 minutos)'
        }), 408

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/list_scripts')
def list_scripts():
    """Lista los scripts disponibles"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    scripts = []
    for filename in os.listdir(project_root):
        if filename.endswith('.bat') and filename[0].isdigit():
            scripts.append({
                'name': filename.replace('.bat', ''),
                'filename': filename,
                'exists': True
            })

    return jsonify({'scripts': sorted(scripts, key=lambda x: x['name'])})


# ============================================================================
# NUEVOS ENDPOINTS PARA HERRAMIENTAS AVANZADAS
# ============================================================================

@app.route('/train_model', methods=['POST'])
def train_model():
    """Entrena el modelo con opciones personalizadas"""
    data = request.json

    catalog_path = data.get('catalog_path', 'data/elodie/')
    model_type = data.get('model_type', 'decision_tree')
    max_depth = data.get('max_depth', 9)
    test_size = data.get('test_size', 20) / 100.0
    n_estimators = data.get('n_estimators', 100)
    output_path = data.get('output_path', 'models/')

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Validar ruta del catálogo
    full_catalog_path = os.path.join(project_root, catalog_path)
    if not os.path.exists(full_catalog_path):
        return jsonify({
            'success': False,
            'error': f'Catálogo no encontrado: {catalog_path}'
        }), 404

    # Contar archivos en el catálogo
    spectrum_files = [f for f in os.listdir(full_catalog_path) if '_tipo' in f.lower() and f.endswith('.txt')]
    if len(spectrum_files) == 0:
        return jsonify({
            'success': False,
            'error': f'No se encontraron espectros etiquetados en: {catalog_path}'
        }), 400

    try:
        import time
        start_time = time.time()

        # Importar módulo de entrenamiento
        sys.path.insert(0, project_root)

        # Ejecutar entrenamiento usando train_and_validate.py
        # Construir comando Python (el script está en src/)
        train_script = os.path.join(project_root, 'src', 'train_and_validate.py')

        if not os.path.exists(train_script):
            return jsonify({
                'success': False,
                'error': 'Script de entrenamiento no encontrado: src/train_and_validate.py'
            }), 404

        # Ejecutar script de entrenamiento con parámetros
        import subprocess
        cmd = [
            sys.executable, train_script,
            '--catalog', full_catalog_path,
            '--output', os.path.join(project_root, output_path),
            '--model', model_type,
            '--max-depth', str(max_depth),
            '--test-size', str(test_size),
        ]

        if model_type in ['random_forest', 'gradient_boosting']:
            cmd.extend(['--n-estimators', str(n_estimators)])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=600,
            cwd=project_root,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            # Leer métricas del modelo entrenado
            metadata_path = os.path.join(project_root, output_path, 'metadata.json')
            metrics = {}
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metrics = json.load(f)

            return jsonify({
                'success': True,
                'output': result.stdout,
                'elapsed_time': round(elapsed_time, 2),
                'n_samples': metrics.get('n_train', 0) + metrics.get('n_test', 0),
                'accuracy': round(metrics.get('accuracy_test', 0) * 100, 2),
                'model_type': model_type,
                'catalog': catalog_path,
                'n_files': len(spectrum_files)
            })
        else:
            return jsonify({
                'success': False,
                'error': result.stderr or 'Error desconocido durante el entrenamiento',
                'output': result.stdout
            }), 500

    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'error': 'El entrenamiento excedió el tiempo límite (10 minutos)'
        }), 408

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


# ============================================================================
# ENDPOINT STREAMING PARA ENTRENAMIENTO EN TIEMPO REAL
# ============================================================================

@app.route('/train_model_stream', methods=['POST'])
def train_model_stream():
    """Entrena modelo con salida en tiempo real via Server-Sent Events."""
    import json as _json
    import re as _re
    from flask import Response, stream_with_context

    data = request.json or {}
    catalog_path   = data.get('catalog_path', 'data/elodie/')
    model_type     = data.get('model_type', 'decision_tree')
    max_depth      = int(data.get('max_depth', 9))
    test_size      = int(data.get('test_size', 20)) / 100.0
    n_estimators   = int(data.get('n_estimators', 100))
    output_path    = data.get('output_path', 'models/')

    full_catalog = (catalog_path if os.path.isabs(catalog_path)
                    else os.path.join(project_root, catalog_path))
    full_output  = (output_path if os.path.isabs(output_path)
                    else os.path.join(project_root, output_path))

    train_script = os.path.join(project_root, 'src', 'train_and_validate.py')

    def generate():
        import time as _time

        def evt(msg, pct=None, tipo='line'):
            if pct is not None:
                return f"data: {_json.dumps({'type': tipo, 'pct': pct, 'msg': msg})}\n\n"
            return f"data: {_json.dumps({'type': tipo, 'msg': msg})}\n\n"

        yield evt('Verificando archivos...', pct=2)

        if not os.path.exists(train_script):
            yield evt('Script no encontrado: src/train_and_validate.py', tipo='done')
            return

        if not os.path.isdir(full_catalog):
            yield evt(f'Catalogo no encontrado: {full_catalog}', tipo='done')
            return

        yield evt(f'Catalogo OK. Lanzando proceso Python...', pct=4)

        cmd = [
            sys.executable, '-u', train_script,
            '--catalog', full_catalog,
            '--output', full_output,
            '--model', model_type,
            '--max-depth', str(max_depth),
            '--test-size', str(test_size),
        ]
        if model_type in ['random_forest', 'gradient_boosting']:
            cmd.extend(['--n-estimators', str(n_estimators)])

        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'}
        pct = 5

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,          # sin buffer en el lado del padre
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=project_root,
                env=env
            )
        except Exception as e:
            yield f"data: {_json.dumps({'type': 'done', 'success': False, 'msg': f'Error al lanzar proceso: {e}'})}\n\n"
            return

        yield evt(f'Proceso PID {proc.pid} iniciado. Esperando salida...', pct=5)

        last_keepalive = _time.time()

        # Leer línea a línea con keepalive cada 3 segundos
        while True:
            line = proc.stdout.readline()

            # readline() devuelve '' cuando el proceso termina y cierra stdout
            if line == '':
                if proc.poll() is not None:
                    break
                # Proceso vivo pero sin output aún → keepalive
                if _time.time() - last_keepalive >= 3:
                    yield ': keepalive\n\n'
                    last_keepalive = _time.time()
                _time.sleep(0.05)
                continue

            last_keepalive = _time.time()
            line = line.rstrip('\r\n')
            if not line.strip():
                continue

            if 'Archivos encontrados' in line:
                pct = 10
            elif 'Procesados:' in line:
                m = _re.search(r'\((\d+\.?\d*)%\)', line)
                if m:
                    pct = int(10 + float(m.group(1)) * 0.40)
            elif 'Espectros procesados exitosamente' in line:
                pct = 52
            elif any(k in line for k in ['ENTRENAMIENTO', 'RANDOM FOREST', 'GRADIENT BOOSTING']):
                pct = 58
            elif 'Accuracy (entrenamiento)' in line:
                pct = 75
            elif 'Accuracy (prueba)' in line or 'Accuracy en test' in line:
                pct = 82
            elif 'VALIDACION CRUZADA' in line or 'VALIDACIÓN CRUZADA' in line:
                pct = 85
            elif 'Accuracy promedio' in line:
                pct = 90
            elif 'VALIDACION DEL CLASIFICADOR' in line or 'VALIDACIÓN DEL CLASIFICADOR' in line:
                pct = 93
            elif '[OK]' in line and 'guardado' in line.lower():
                pct = 97

            yield evt(line, pct=pct)

        proc.wait()

        if proc.returncode == 0:
            metrics = {}
            metadata_path = os.path.join(full_output, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metrics = _json.load(f)
            n_samples = metrics.get('n_train', 0) + metrics.get('n_test', 0)
            accuracy  = round(metrics.get('accuracy_test', 0) * 100, 2)
            yield f"data: {_json.dumps({'type': 'done', 'success': True, 'accuracy': accuracy, 'n_samples': n_samples})}\n\n"
        else:
            yield f"data: {_json.dumps({'type': 'done', 'success': False, 'msg': f'El proceso termino con codigo {proc.returncode}'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


# ============================================================================
# ENDPOINT STREAMING PARA ENTRENAMIENTO NEURAL EN TIEMPO REAL
# ============================================================================

@app.route('/train_neural_stream', methods=['POST'])
def train_neural_stream():
    """Entrena modelo KNN/CNN con salida en tiempo real via Server-Sent Events."""
    import json as _json
    import re as _re
    from flask import Response, stream_with_context

    data = request.json or {}
    model_type   = data.get('model_type', 'knn')
    catalog_path = data.get('catalog_path', 'data/elodie/')
    test_size    = float(data.get('test_size', 0.2))
    output_path  = data.get('output_path', 'models/')

    full_catalog = (catalog_path if os.path.isabs(catalog_path)
                    else os.path.join(project_root, catalog_path))
    full_output  = (output_path if os.path.isabs(output_path)
                    else os.path.join(project_root, output_path))

    train_script = os.path.join(project_root, 'src', 'train_neural_models.py')

    def generate():
        import time as _time

        def evt(msg, pct=None, tipo='line'):
            payload = {'type': tipo, 'msg': msg}
            if pct is not None:
                payload['pct'] = pct
            return f"data: {_json.dumps(payload)}\n\n"

        yield evt('Verificando archivos...', pct=2)

        if not os.path.exists(train_script):
            yield evt('Script no encontrado: src/train_neural_models.py', tipo='done')
            return
        if not os.path.isdir(full_catalog):
            yield evt(f'Catalogo no encontrado: {full_catalog}', tipo='done')
            return

        yield evt(f'Catalogo OK. Lanzando proceso {model_type.upper()}...', pct=5)

        cmd = [
            sys.executable, '-u', train_script,
            '--model', model_type,
            '--catalog', full_catalog,
            '--output', full_output,
            '--test-size', str(test_size),
        ]
        if model_type == 'knn':
            cmd += ['--n-neighbors', str(data.get('n_neighbors', 5)),
                    '--weights',     data.get('weights', 'distance'),
                    '--metric',      data.get('metric', 'euclidean')]
        else:
            cmd += ['--epochs',        str(data.get('epochs', 50)),
                    '--batch-size',    str(data.get('batch_size', 32)),
                    '--learning-rate', str(data.get('learning_rate', 0.001)),
                    '--dropout',       str(data.get('dropout_rate', 0.3)),
                    '--dense-units',   str(data.get('dense_units', 128))]
            if model_type == 'cnn_2d':
                cmd += ['--image-size', str(data.get('image_size', 64)),
                        '--image-dir',  data.get('image_dir', 'espectros png/'),
                        '--labels-csv', data.get('labels_csv', 'espectros png/clasificacion_estrellas.csv')]

        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'}
        pct = 5

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=project_root,
                env=env
            )
        except Exception as e:
            yield evt(f'Error al lanzar proceso: {e}', tipo='done')
            return

        yield evt(f'Proceso PID {proc.pid} iniciado. Esperando salida...', pct=6)

        last_keepalive = _time.time()

        while True:
            line = proc.stdout.readline()
            if line == '':
                if proc.poll() is not None:
                    break
                if _time.time() - last_keepalive >= 3:
                    yield ': keepalive\n\n'
                    last_keepalive = _time.time()
                _time.sleep(0.05)
                continue

            last_keepalive = _time.time()
            line = line.rstrip('\r\n')
            if not line.strip():
                continue

            # Calcular porcentaje según contenido
            if 'Archivos encontrados' in line:
                pct = 12
            elif 'Procesados:' in line:
                m = _re.search(r'(\d+)/(\d+)', line)
                if m:
                    pct = int(12 + (int(m.group(1)) / max(int(m.group(2)), 1)) * 30)
            elif 'Espectros cargados exitosamente' in line:
                pct = 45
            elif 'ENTRENANDO KNN' in line:
                pct = 55
            elif 'ENTRENANDO CNN' in line:
                pct = 55
            elif 'Entrenamiento completado' in line:
                pct = 85
            elif 'Accuracy en test' in line:
                pct = 90
            elif 'Epocas ejecutadas' in line or 'EarlyStopping' in line.lower():
                pct = 88
            elif '[OK]' in line or 'guardado' in line.lower():
                pct = 95

            yield evt(line, pct=pct)

        proc.wait()

        if proc.returncode == 0:
            metrics = {}
            metadata_path = os.path.join(full_output, f'{model_type}_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metrics = _json.load(f)
            yield f"data: {_json.dumps({'type': 'done', 'success': True, 'accuracy': round(metrics.get('accuracy_test', 0) * 100, 2), 'n_samples': metrics.get('n_samples', 0), 'n_classes': len(metrics.get('classes', []))})}\n\n"
        else:
            yield f"data: {_json.dumps({'type': 'done', 'success': False, 'msg': f'El proceso termino con codigo {proc.returncode}'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


# ============================================================================
# ENDPOINTS PARA REDES NEURONALES (KNN, CNN)
# ============================================================================

@app.route('/train_neural', methods=['POST'])
def train_neural():
    """Entrena modelos KNN o CNN para clasificación espectral"""
    data = request.json

    model_type = data.get('model_type', 'knn')  # 'knn', 'cnn_1d', 'cnn_2d'
    catalog_path = data.get('catalog_path', 'data/elodie/')
    # test_size ya viene como decimal desde JS (ej: 0.2 para 20%)
    test_size = data.get('test_size', 0.2)
    if test_size > 1:  # Si viene como porcentaje, convertir
        test_size = test_size / 100.0
    output_path = data.get('output_path', 'models/')

    # Parámetros KNN
    n_neighbors = data.get('n_neighbors', 5)
    weights = data.get('weights', 'uniform')
    metric = data.get('metric', 'euclidean')

    # Parámetros CNN
    epochs = data.get('epochs', 20)
    batch_size = data.get('batch_size', 32)
    learning_rate = data.get('learning_rate', 0.001)
    dropout_rate = data.get('dropout_rate', 0.3)
    dense_units = data.get('dense_units', 128)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Validar ruta del catálogo
    full_catalog_path = os.path.join(project_root, catalog_path)
    if not os.path.exists(full_catalog_path):
        return jsonify({
            'success': False,
            'error': f'Catálogo no encontrado: {catalog_path}'
        }), 404

    # Contar archivos
    spectrum_files = [f for f in os.listdir(full_catalog_path)
                      if '_tipo' in f.lower() and f.endswith('.txt')]
    if len(spectrum_files) == 0:
        return jsonify({
            'success': False,
            'error': f'No se encontraron espectros etiquetados en: {catalog_path}'
        }), 400

    try:
        import time
        start_time = time.time()

        # Verificar que existe el script de entrenamiento (en src/)
        train_script = os.path.join(project_root, 'src', 'train_neural_models.py')
        if not os.path.exists(train_script):
            return jsonify({
                'success': False,
                'error': 'Script de entrenamiento no encontrado: src/train_neural_models.py'
            }), 404

        # Construir comando
        cmd = [
            sys.executable, train_script,
            '--model', model_type,
            '--catalog', full_catalog_path,
            '--output', os.path.join(project_root, output_path),
            '--test-size', str(test_size),
        ]

        # Agregar parámetros específicos según tipo de modelo
        if model_type == 'knn':
            cmd.extend([
                '--n-neighbors', str(n_neighbors),
                '--weights', weights,
                '--metric', metric,
            ])
        else:  # CNN
            cmd.extend([
                '--epochs', str(epochs),
                '--batch-size', str(batch_size),
                '--learning-rate', str(learning_rate),
                '--dropout', str(dropout_rate),
                '--dense-units', str(dense_units),
            ])

        # Ejecutar entrenamiento
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=1800,  # 30 minutos para CNN
            cwd=project_root,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )

        elapsed_time = time.time() - start_time

        # Log para debug
        print(f"[train_neural] Comando ejecutado: {' '.join(cmd)}")
        print(f"[train_neural] Return code: {result.returncode}")
        if result.stdout:
            print(f"[train_neural] STDOUT (ultimos 1500 chars):")
            print(result.stdout[-1500:])
        if result.stderr:
            print(f"[train_neural] STDERR:")
            print(result.stderr[-2000:])

        if result.returncode == 0:
            # Leer métricas del modelo entrenado
            metadata_path = os.path.join(project_root, output_path, f'{model_type}_metadata.json')
            metrics = {}
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metrics = json.load(f)

            return jsonify({
                'success': True,
                'output': result.stdout,
                'elapsed_time': round(elapsed_time, 2),
                'training_time': f"{int(elapsed_time // 60)}:{int(elapsed_time % 60):02d}",
                'n_samples': metrics.get('n_samples', 0),
                'accuracy': round(metrics.get('accuracy_test', 0) * 100, 2),
                'model_type': model_type,
                'catalog': catalog_path,
                'n_files': len(spectrum_files),
                'classes': metrics.get('classes', []),
                'n_classes': len(metrics.get('classes', []))
            })
        else:
            # Extraer el error más relevante del stderr
            error_msg = result.stderr or 'Error desconocido'
            # Buscar la última línea de error
            if error_msg:
                lines = error_msg.strip().split('\n')
                # Buscar líneas con Error o Exception
                error_lines = [l for l in lines if 'Error' in l or 'Exception' in l or 'error' in l.lower()]
                error_summary = error_lines[-1] if error_lines else lines[-1]
            else:
                error_summary = 'Error desconocido durante el entrenamiento'

            return jsonify({
                'success': False,
                'error': error_summary,
                'stderr': result.stderr[-1500:] if result.stderr else '',
                'stdout': result.stdout[-500:] if result.stdout else ''
            }), 500

    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'error': 'El entrenamiento excedió el tiempo límite (30 minutos)'
        }), 408

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/neural_metrics', methods=['GET'])
def get_neural_metrics():
    """Obtiene métricas de los modelos neuronales disponibles"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')

    metrics = {
        'knn': None,
        'cnn_1d': None,
        'cnn_2d': None
    }

    # Buscar métricas de cada modelo
    for model_type in metrics.keys():
        metadata_path = os.path.join(models_dir, f'{model_type}_metadata.json')
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metrics[model_type] = json.load(f)

    # También buscar modelo KNN genérico
    knn_generic = os.path.join(models_dir, 'knn_metadata.json')
    if os.path.exists(knn_generic) and metrics['knn'] is None:
        import json
        with open(knn_generic, 'r') as f:
            metrics['knn'] = json.load(f)

    # Verificar qué modelos existen
    available_models = []
    if os.path.exists(os.path.join(models_dir, 'knn_model.pkl')):
        available_models.append('knn')
    if os.path.exists(os.path.join(models_dir, 'cnn_model.h5')) or \
       os.path.exists(os.path.join(models_dir, 'cnn_1d_model.h5')):
        available_models.append('cnn_1d')
    if os.path.exists(os.path.join(models_dir, 'cnn_2d_model.h5')):
        available_models.append('cnn_2d')

    # Construir dict 'models' con los campos que espera el frontend JS
    models_frontend = {}
    for model_type in available_models:
        meta = metrics.get(model_type) or {}
        acc_raw = meta.get('accuracy_test')
        models_frontend[model_type] = {
            'tipo':            meta.get('model_type', model_type),
            'clases':          meta.get('classes', []),
            'accuracy':        round(acc_raw * 100, 1) if acc_raw is not None else None,
            'n_neighbors':     meta.get('n_neighbors'),
            'spectrum_length': meta.get('spectrum_length'),
            'n_samples':       meta.get('n_samples'),
            'epochs_trained':  meta.get('epochs_trained'),
        }

    return jsonify({
        'success': True,
        'models': models_frontend,
        'metrics': metrics,
        'available_models': available_models
    })


@app.route('/verify_neural_models', methods=['GET'])
def verify_neural_models():
    """Verifica qué modelos neuronales están disponibles"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')

    status = {
        'knn': {
            'model_exists': os.path.exists(os.path.join(models_dir, 'knn_model.pkl')),
            'metadata_exists': os.path.exists(os.path.join(models_dir, 'knn_metadata.json')),
            'scaler_exists': os.path.exists(os.path.join(models_dir, 'knn_scaler.pkl'))
        },
        'cnn_1d': {
            'model_exists': os.path.exists(os.path.join(models_dir, 'cnn_model.h5')) or
                           os.path.exists(os.path.join(models_dir, 'cnn_1d_model.h5')),
            'metadata_exists': os.path.exists(os.path.join(models_dir, 'cnn_1d_metadata.json')) or
                              os.path.exists(os.path.join(models_dir, 'cnn_metadata.json'))
        },
        'cnn_2d': {
            'model_exists': os.path.exists(os.path.join(models_dir, 'cnn_2d_model.h5')),
            'metadata_exists': os.path.exists(os.path.join(models_dir, 'cnn_2d_metadata.json'))
        }
    }

    return jsonify({
        'success': True,
        'status': status,
        'models_dir': models_dir
    })


@app.route('/test_spectrum_advanced', methods=['POST'])
def test_spectrum_advanced():
    """Prueba un espectro con opciones avanzadas"""
    data = request.json

    spectrum_path = data.get('spectrum_path', '')
    method = data.get('method', 'multi_method')
    detail_level = data.get('detail_level', 'detailed')
    save_plot = data.get('save_plot', True)

    if not spectrum_path:
        return jsonify({'success': False, 'error': 'Ruta de espectro no especificada'}), 400

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Resolver la ruta del espectro
    # 1. Si es ruta absoluta y existe, usarla directamente
    # 2. Si es solo nombre de archivo, buscar en carpetas conocidas
    # 3. Si es ruta relativa, buscar desde la raíz del proyecto

    filename_only = os.path.basename(spectrum_path)

    search_dirs = [
        project_root,
        os.path.join(project_root, 'data', 'elodie'),
        os.path.join(project_root, 'data', 'espectros'),
        os.path.join(project_root, 'elodie'),
        os.path.join(project_root, 'espectros'),
    ]

    possible_paths = [spectrum_path]                                            # Ruta tal cual (absoluta)
    possible_paths += [os.path.join(project_root, spectrum_path)]              # Relativa al proyecto
    possible_paths += [os.path.join(project_root, 'webapp', 'uploads', filename_only)]  # Subida reciente
    possible_paths += [os.path.join(d, filename_only) for d in search_dirs]   # Solo nombre en dirs conocidos

    full_path = None
    for path in possible_paths:
        if os.path.isfile(path):
            full_path = os.path.abspath(path)
            break

    if not full_path:
        return jsonify({
            'success': False,
            'error': (
                f'Archivo no encontrado: {spectrum_path}\n'
                f'Puedes usar:\n'
                f'  - Ruta absoluta: C:\\Users\\...\\espectro.txt  o  E:\\pen\\espectro.txt\n'
                f'  - Ruta relativa al proyecto: data/elodie/HD000108_tipoO6pe.txt\n'
                f'  - Solo el nombre si está en data/elodie/ o data/espectros/: HD000108_tipoO6pe.txt'
            )
        }), 404

    try:
        # Procesar espectro
        use_multi = (method == 'multi_method')
        result = process_spectrum(full_path, os.path.basename(spectrum_path), use_multi_method=use_multi)

        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']}), 500

        # Extraer tipo original del nombre
        original_type = 'N/A'
        filename = os.path.basename(spectrum_path)
        if '_tipo' in filename.lower():
            parts = filename.lower().split('_tipo')
            if len(parts) > 1:
                original_type = parts[1].replace('.txt', '').upper()[:5]

        # Determinar si coincide
        classified_main = result['tipo_clasificado'][0] if result['tipo_clasificado'] else ''
        original_main = original_type[0] if original_type != 'N/A' else ''
        is_match = classified_main == original_main

        # Preparar respuesta
        response = {
            'success': True,
            'filename': filename,
            'tipo_clasificado': result['tipo_clasificado'],
            'subtipo': result['subtipo'],
            'confianza': result.get('confianza'),
            'original_type': original_type,
            'is_match': is_match,
            'n_lineas': result['n_lineas'],
            'rango_lambda': result['rango_lambda'],
            'method_used': method,
            'lineas_detectadas': ([] if detail_level == 'basic'
                                   else result['lineas_detectadas'] if detail_level == 'debug'
                                   else result['lineas_detectadas'][:20])
        }

        if save_plot and 'plot_path' in result:
            response['plot_url'] = f"/plot/{os.path.basename(result['plot_path'])}"

        if detail_level == 'debug' and 'diagnostics' in result:
            response['diagnostics'] = result['diagnostics']

        return jsonify(response)

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/get_metrics')
def get_metrics():
    """Obtiene métricas del modelo actual"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')

    metadata_path = os.path.join(models_dir, 'metadata.json')
    report_path = os.path.join(models_dir, 'validation_report.txt')
    confusion_matrix_path = os.path.join(models_dir, 'confusion_matrix.png')

    if not os.path.exists(metadata_path):
        return jsonify({
            'success': False,
            'error': 'Modelo no encontrado. Ejecuta el entrenamiento primero.'
        }), 404

    try:
        import json

        # Cargar metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Cargar reporte si existe
        report_text = ''
        accuracy_by_type = {}
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report_text = f.read()

            # Parsear accuracy por tipo del reporte
            import re
            pattern = r'(\w):\s+([\d.]+)%\s+\((\d+)/(\d+)\)'
            matches = re.findall(pattern, report_text)
            for match in matches:
                tipo, acc, correct, total = match
                accuracy_by_type[tipo] = {
                    'accuracy': float(acc),
                    'correct': int(correct),
                    'total': int(total)
                }

        # Verificar matriz de confusión
        has_confusion_matrix = os.path.exists(confusion_matrix_path)

        # Top features
        top_features = []
        if 'feature_names' in metadata:
            # Intentar cargar importancias si están disponibles
            # Por ahora, listar los features
            top_features = metadata['feature_names'][:10]

        return jsonify({
            'success': True,
            'accuracy_test': round(metadata.get('accuracy_test', 0) * 100, 2),
            'accuracy_cv_mean': round(metadata.get('accuracy_cv_mean', 0) * 100, 2),
            'accuracy_physical': round(metadata.get('accuracy_physical', 0) * 100, 2),
            'n_train': metadata.get('n_train', 0),
            'n_test': metadata.get('n_test', 0),
            'timestamp': metadata.get('timestamp', 'N/A'),
            'accuracy_by_type': accuracy_by_type,
            'has_confusion_matrix': has_confusion_matrix,
            'top_features': top_features,
            'report_text': report_text
        })

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/list_catalogs')
def list_catalogs():
    """Lista los catálogos de espectros disponibles"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    catalogs = []

    def scan_directory(base_path, prefix=''):
        """Escanea un directorio buscando catálogos de espectros"""
        if not os.path.exists(base_path):
            return
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                # Contar archivos de espectros
                try:
                    spectrum_files = [f for f in os.listdir(item_path) if f.endswith('.txt') and '_tipo' in f.lower()]
                except PermissionError:
                    continue

                if len(spectrum_files) > 0:
                    # Contar tipos
                    types_count = {}
                    for f in spectrum_files:
                        if '_tipo' in f.lower():
                            parts = f.lower().split('_tipo')
                            if len(parts) > 1:
                                tipo = parts[1].replace('.txt', '').upper()
                                main_type = tipo[0] if tipo else '?'
                                types_count[main_type] = types_count.get(main_type, 0) + 1

                    rel_path = prefix + item + '/' if prefix else item + '/'
                    catalogs.append({
                        'name': item,
                        'path': rel_path,
                        'n_files': len(spectrum_files),
                        'types': types_count
                    })

    # Buscar en la raíz del proyecto
    scan_directory(project_root)

    # Buscar en la carpeta data/
    data_path = os.path.join(project_root, 'data')
    scan_directory(data_path, 'data/')

    return jsonify({
        'success': True,
        'catalogs': catalogs
    })


@app.route('/list_catalog_files/<path:catalog>')
def list_catalog_files(catalog):
    """Lista archivos de un catálogo específico con filtro opcional"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    catalog_path = os.path.join(project_root, catalog)

    if not os.path.exists(catalog_path) or not os.path.isdir(catalog_path):
        return jsonify({'success': False, 'error': f'Catálogo no encontrado: {catalog}'}), 404

    # Filtro opcional
    filter_text = request.args.get('filter', '').lower()
    filter_type = request.args.get('type', '').upper()
    limit = int(request.args.get('limit', 100))

    files = []
    for f in sorted(os.listdir(catalog_path)):
        if not f.endswith('.txt'):
            continue

        # Aplicar filtros
        if filter_text and filter_text not in f.lower():
            continue

        # Extraer tipo del nombre
        tipo = ''
        if '_tipo' in f.lower():
            parts = f.lower().split('_tipo')
            if len(parts) > 1:
                tipo = parts[1].replace('.txt', '').upper()

        if filter_type and not tipo.startswith(filter_type):
            continue

        files.append({
            'filename': f,
            'path': f'{catalog}/{f}',
            'tipo': tipo
        })

        if len(files) >= limit:
            break

    return jsonify({
        'success': True,
        'catalog': catalog,
        'files': files,
        'total': len(files)
    })


@app.route('/verify_model')
def verify_model():
    """Verifica el estado del modelo"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')

    checks = {
        'model_file': os.path.exists(os.path.join(models_dir, 'decision_tree.pkl')),
        'metadata': os.path.exists(os.path.join(models_dir, 'metadata.json')),
        'confusion_matrix': os.path.exists(os.path.join(models_dir, 'confusion_matrix.png')),
        'validation_report': os.path.exists(os.path.join(models_dir, 'validation_report.txt')),
        'multi_method_available': MULTI_METHOD_AVAILABLE
    }

    all_ok = all([checks['model_file'], checks['metadata']])

    # Cargar metadata si existe
    model_info = {}
    if checks['metadata']:
        try:
            import json
            with open(os.path.join(models_dir, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
                model_info = {
                    'accuracy': round(metadata.get('accuracy_test', 0) * 100, 2),
                    'n_samples': metadata.get('n_train', 0) + metadata.get('n_test', 0),
                    'timestamp': metadata.get('timestamp', 'N/A')
                }
        except:
            pass

    return jsonify({
        'success': True,
        'status': 'OK' if all_ok else 'INCOMPLETE',
        'checks': checks,
        'model_info': model_info
    })


@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Limpia archivos temporales y resultados"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    cleared = {
        'uploads': 0,
        'results': 0
    }

    # Limpiar uploads
    uploads_dir = app.config['UPLOAD_FOLDER']
    if os.path.exists(uploads_dir):
        for f in os.listdir(uploads_dir):
            try:
                os.remove(os.path.join(uploads_dir, f))
                cleared['uploads'] += 1
            except:
                pass

    # Limpiar results
    results_dir = app.config['RESULTS_FOLDER']
    if os.path.exists(results_dir):
        for f in os.listdir(results_dir):
            try:
                os.remove(os.path.join(results_dir, f))
                cleared['results'] += 1
            except:
                pass

    return jsonify({
        'success': True,
        'cleared': cleared,
        'message': f"Eliminados {cleared['uploads']} uploads y {cleared['results']} resultados"
    })


@app.route('/confusion_matrix')
def get_confusion_matrix():
    """Devuelve la imagen de la matriz de confusión"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(project_root, 'models', 'confusion_matrix.png')

    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/png')
    else:
        return "Matriz de confusión no encontrada", 404


# ============================================================================
# ÁRBOL INTERACTIVO — devuelve datos crudos del espectro para visualización
# ============================================================================

@app.route('/spectrum_raw', methods=['POST'])
def spectrum_raw():
    """
    Carga un espectro y devuelve arrays wavelength/flux normalizados como JSON.
    Usado por el Árbol Interactivo para mostrar el espectro real en cada paso.
    """
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No se recibió archivo'}), 400

    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Formato no válido (.txt, .fits, .fit)'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        wavelengths, flux, error, metadata = load_spectrum_file(filepath)

        if error:
            return jsonify({'success': False, 'error': error}), 400

        # Normalizar al continuo para mostrar correctamente
        flux_norm, _ = normalize_to_continuum(wavelengths, flux)

        # Reducir puntos si el espectro es muy largo (máx 2000 puntos para el SVG)
        n = len(wavelengths)
        if n > 2000:
            step = n // 2000
            wavelengths = wavelengths[::step]
            flux_norm   = flux_norm[::step]

        return jsonify({
            'success': True,
            'wavelength': wavelengths.tolist(),
            'flux': flux_norm.tolist(),
            'wmin': float(wavelengths.min()),
            'wmax': float(wavelengths.max()),
            'filename': filename,
            'n_points': len(wavelengths),
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ENDPOINT: CLASE DE LUMINOSIDAD MK (para el Árbol Interactivo)
# ============================================================================

@app.route('/api/luminosity', methods=['POST'])
def api_luminosity():
    """
    Calcula la clase de luminosidad MK para un espectro ya normalizado.

    Acepta JSON con:
        wavelength   : list[float]  — longitudes de onda en Å
        flux         : list[float]  — flujo normalizado al continuo
        spectral_type: str          — tipo espectral determinado por el árbol
                                      (p. ej. "G2", "B1", "M4")

    Devuelve JSON con:
        success         : bool
        luminosity_class: str   — "Ia", "Ib", "II", "III", "IV" o "V"
        mk_full         : str   — tipo MK completo (p. ej. "G2V", "B1Ib")
        lum_name        : str   — nombre legible de la clase
        indicators      : dict  — razones usadas para el diagnóstico
        error           : str   — presente solo si success == False
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'success': False, 'error': 'Se esperaba JSON'}), 400

        wavelength    = np.asarray(data.get('wavelength', []), dtype=float)
        flux          = np.asarray(data.get('flux',       []), dtype=float)
        spectral_type = str(data.get('spectral_type', '')).strip()

        if len(wavelength) < 10 or len(flux) < 10:
            return jsonify({'success': False, 'error': 'Datos de espectro insuficientes'}), 400

        if not spectral_type:
            return jsonify({'success': False, 'error': 'Se requiere spectral_type'}), 400

        # Medir líneas diagnóstico con las ventanas optimizadas
        measurements = measure_diagnostic_lines(wavelength, flux)

        # Importar módulo de luminosidad
        from luminosity_classification import (
            estimate_luminosity_class,
            combine_spectral_and_luminosity
        )
        from spectral_classification_corrected import compute_spectral_ratios

        lum_class = estimate_luminosity_class(measurements, spectral_type)
        mk_full   = combine_spectral_and_luminosity(spectral_type, lum_class)

        # Nombre legible de la clase
        lum_names = {
            'Ia': 'Supergigante muy luminosa',
            'Ib': 'Supergigante',
            'II': 'Gigante brillante',
            'III': 'Gigante',
            'IV': 'Subgigante',
            'V': 'Secuencia principal (enana)',
        }
        lum_name = lum_names.get(lum_class, lum_class)

        # Calcular razones diagnóstico relevantes para mostrar en UI
        ratios = compute_spectral_ratios(measurements)
        key_ratios = {
            k: round(float(v), 3)
            for k, v in ratios.items()
            if k in ('HeI_HeII', 'SrII_FeI', 'CaI_FeI', 'BaII_FeI',
                     'NaI_CaI', 'TiO_CaH', 'NIII_HeII', 'MgIb_FeI')
        }

        return jsonify(convert_numpy_types({
            'success':          True,
            'luminosity_class': lum_class,
            'mk_full':          mk_full,
            'lum_name':         lum_name,
            'indicators':       key_ratios,
        }))

    except ImportError as e:
        return jsonify({'success': False,
                        'error': f'Módulo de luminosidad no disponible: {e}'}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# MANEJADORES DE ERROR GLOBALES — siempre devuelven JSON, nunca HTML
# ============================================================================

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'success': False, 'error': f'Solicitud incorrecta: {str(e)}'}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': f'Ruta no encontrada: {request.path}'}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'success': False, 'error': f'Método no permitido en {request.path}'}), 405

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'Archivo demasiado grande (máximo 50 MB)'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': f'Error interno del servidor: {str(e)}'}), 500


if __name__ == '__main__':
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    print("="*70)
    print("CLASIFICACION ESPECTRAL - Servidor Web")
    print("="*70)
    print(f"Lineas espectrales configuradas: {len(SPECTRAL_LINES)}")
    print(f"Directorio de subidas: {app.config['UPLOAD_FOLDER']}")
    print(f"Directorio de resultados: {app.config['RESULTS_FOLDER']}")
    print("\nIniciando servidor en: http://localhost:5000")
    print("Presiona Ctrl+C para detener")
    print("="*70)

    app.run(debug=False, host='0.0.0.0', port=5000)
