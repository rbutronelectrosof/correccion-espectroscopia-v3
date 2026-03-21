@echo off
chcp 65001 >nul
echo ========================================================================
echo  INSTALACION - Sistema de Clasificacion Espectral v2.0
echo ========================================================================
echo.

echo [1/4] Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo X ERROR: Python no esta instalado o no esta en PATH
    echo.
    echo Descarga Python desde: https://www.python.org/downloads/
    echo Asegurate de marcar "Add Python to PATH" durante la instalacion
    pause
    exit /b 1
)
python --version
echo OK Python detectado
echo.

echo [2/4] Actualizando pip...
python -m pip install --upgrade pip
echo.

echo [3/4] Instalando dependencias...
echo.
python -m pip install numpy scipy matplotlib pandas astropy scikit-learn seaborn flask werkzeug Pillow joblib

if errorlevel 1 (
    echo.
    echo X ERROR: Hubo un problema instalando los paquetes
    pause
    exit /b 1
)

echo.
echo OK Dependencias basicas instaladas
echo.

echo [4/4] TensorFlow (opcional - para CNN)
echo ========================================================================
echo TensorFlow es necesario solo para clasificacion con CNN.
echo Es una descarga grande (~500MB) y puede tardar varios minutos.
echo.
set /p instalar_tf="Deseas instalar TensorFlow? (s/n): "

if /i "%instalar_tf%"=="s" (
    echo.
    echo Instalando TensorFlow...
    python -m pip install tensorflow
    if errorlevel 1 (
        echo [ADVERTENCIA] No se pudo instalar TensorFlow
        echo Puedes usar KNN y arbol de decision sin TensorFlow
    ) else (
        echo OK TensorFlow instalado
    )
) else (
    echo.
    echo TensorFlow no instalado. Puedes instalarlo mas tarde con:
    echo    pip install tensorflow
)

echo.
echo ========================================================================
echo  INSTALACION COMPLETADA
echo ========================================================================
echo.
echo Paquetes instalados:
echo   - numpy, scipy, matplotlib (procesamiento cientifico)
echo   - pandas (manejo de datos)
echo   - astropy (archivos FITS)
echo   - scikit-learn (machine learning)
echo   - seaborn (visualizacion)
echo   - flask, werkzeug (servidor web)
echo.
echo Siguiente paso: Ejecuta INICIAR.bat
echo.
pause
