# SpectroClass: Arquitectura Multi-Método para Clasificación Espectral Automática
# SpectroClass: A Multi-Method Architecture for Automatic Spectral Classification

**Roberto Butron**
Facultad de Ciencias Exactas y Naturales, UNCuyo, Padre J. Contreras 1300, M5502JMA, Mendoza, Argentina

Congreso: Evolución estelar, exoplanetas y dinámica de sistemas estelares
Fecha: 22–24 de abril de 2026

---

## Resumen (Español)

Presentamos SpectroClass, un nuevo software de clasificación espectral estelar automática que integra tres métodos independientes con votación ponderada configurable. El sistema procesa espectros en formato TXT y FITS, reconstruyendo la calibración en longitud de onda a partir de la solución WCS almacenada en la cabecera FITS, y normaliza el espectro al continuo mediante un ajuste local iterativo que excluye automáticamente las regiones de absorción.

SpectroClass mide **85 líneas diagnóstico** mediante integración trapezoidal del perfil de absorción en ventanas adaptativas por línea (6–30 Å, diccionario LINE_WINDOWS). La identificación de cada línea se realiza buscando el mínimo de flujo dentro de una ventana centrada en la longitud de onda de laboratorio, con un umbral de tolerancia adaptado al tipo espectral. Se calculan además **21 razones diagnóstico** (función compute_spectral_ratios) que cubren toda la secuencia OBAFGKM: razones de ionización para tipos tempranos (He I/He II, Si III/Si II, N IV/N III), de temperatura para tipos tardíos (Ca II K/Hε, Cr I/Fe I) y de gravedad para luminosidad (Sr II/Fe I, Ba II/Fe I, TiO/CaH).

Los anchos equivalentes y razones alimentan tres clasificadores independientes: (1) un clasificador físico basado en criterios de intensidad de líneas con **árbol de decisión de 26 nodos** (ampliado desde 18), incluyendo 8 nuevos nodos para subtipos O3–O4 (N V + O V), O tardíos (N III), B con O II, F/G con banda CH G, G con tripleta Mg b, K con Na I D, y M tardías con VO y CaH; (2) un árbol interactivo web que replica el flujo de clasificación visual permitiendo intervención del usuario; (3) clasificadores neuronales k-NN y CNN-1D sobre el espectro remuestreado a una grilla uniforme.

La clasificación final se obtiene por votación ponderada, reportando tipo MK, clase de luminosidad (I–V, módulo luminosity_classification.py), subtipo, confianza porcentual y hasta tres alternativas con justificación diagnóstica trazable. SpectroClass se distribuye como aplicación web de código abierto con interfaz de visualización espectral interactiva, exportación a CSV/TXT/PDF y documentación de ayuda integrada en castellano.

Palabras clave: clasificación espectral — sistema MK — anchos equivalentes — redes neuronales — automatización — espectroscopía estelar — clase de luminosidad

---

## Abstract (English)

We present SpectroClass, a new software for automatic stellar spectral classification integrating three independent methods with configurable weighted voting. The system processes TXT and FITS spectra, reconstructing the wavelength calibration from the WCS solution stored in the FITS header, and normalizes the spectrum to the continuum through an iterative local fit that automatically excludes absorption regions.

SpectroClass measures **85 diagnostic lines** through trapezoidal integration of the absorption profile within per-line adaptive windows (6–30 Å, LINE_WINDOWS dictionary). Each line is identified by locating the flux minimum within a window centered on the laboratory wavelength, with a specified tolerance threshold. Additionally, **21 diagnostic ratios** (compute_spectral_ratios function) are computed covering the full OBAFGKM sequence: ionization ratios for early types (He I/He II, Si III/Si II, N IV/N III), temperature ratios for late types (Ca II K/Hε, Cr I/Fe I), and gravity-sensitive ratios for luminosity estimation (Sr II/Fe I, Ba II/Fe I, TiO/CaH).

The resulting equivalent widths and ratios feed three independent classifiers: (1) a physical classifier based on line strength ratios and equivalent width criteria calibrated against MK standard stars, with a **26-node hierarchical decision tree** (expanded from 18), including 8 new nodes for O3–O4 (N V + O V), late O (N III), B with O II lines, F/G with the CH G-band, G with Mg b triplet, K with Na I D, and late M with VO and CaH; (2) an interactive web decision tree replicating the visual classification workflow with user intervention capability; and (3) k-NN and 1D-CNN neural classifiers trained on spectra resampled over a uniform grid.

The final classification is obtained by weighted voting, reporting MK type, **luminosity class** (I–V, luminosity_classification.py module), subtype, percentage confidence, and up to three alternatives with traceable diagnostic justification. SpectroClass is distributed as an open-source web application with an interactive spectral visualization interface, CSV/TXT/PDF export, and integrated help documentation in Spanish.

Keywords: spectral classification — MK system — equivalent widths — neural networks — automation — stellar spectroscopy — luminosity class
