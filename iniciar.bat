@echo off
chcp 65001 >nul
echo ========================================================================
echo  WebApp - Sistema de Clasificacion Espectral v3.0
echo ========================================================================
echo.

REM Verificar Python y Flask
python --version >nul 2>&1
if errorlevel 1 (
    echo X ERROR: Python no esta instalado
    echo Ejecuta primero: INSTALAR.bat
    pause
    exit /b 1
)

python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo X ERROR: Las dependencias no estan instaladas
    echo Ejecuta primero: INSTALAR.bat
    pause
    exit /b 1
)

REM Verificar directorio webapp
if not exist "webapp\app.py" (
    echo X ERROR: No se encuentra webapp\app.py
    echo Asegurate de ejecutar este script desde el directorio del proyecto
    pause
    exit /b 1
)

REM Verificar modelos (advertencia si no existen)
if not exist "models\decision_tree.pkl" (
    echo ========================================================================
    echo  ADVERTENCIA: Modelo ML no encontrado
    echo ========================================================================
    echo.
    echo El sistema funcionara con clasificador fisico + template matching.
    echo Para mayor precision, entrena los modelos desde la interfaz web.
    echo.
    timeout /t 3 >nul
)

echo Iniciando servidor web...
echo.

REM Abrir navegador automaticamente despues de 3 segundos
start "" /b cmd /c "ping -n 8 127.0.0.1 >nul && explorer http://localhost:5000"

echo ========================================================================
echo.
echo  Servidor iniciado en: http://localhost:5000
echo.
echo  Tu navegador se abrira automaticamente...
echo.
echo  Para DETENER el servidor: Presiona Ctrl+C
echo.
echo ========================================================================
echo.

cd /d "%~dp0"
python webapp/app.py

echo.
echo Servidor detenido.
pause
