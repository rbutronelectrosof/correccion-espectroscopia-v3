"""
Clasificación de Luminosidad MK (Clase de Luminosidad)
========================================================

Estima la clase de luminosidad MK de una estrella a partir de los anchos
equivalentes (EW) de indicadores espectroscópicos de gravedad superficial.

Clases MK implementadas
-----------------------
  I    Supergigante
  II   Gigante brillante
  III  Gigante
  IV   Subgigante
  V    Secuencia principal (enana)

Fundamento astrofísico
----------------------
La clase de luminosidad refleja la gravedad superficial de la estrella
(log g).  A mayor luminosidad (menor log g):

* Las líneas de presión aumentan su EW por el efecto Stark (p. ej. Balmer
  en estrellas A).
* Las líneas de iones sensibles a la densidad electrónica cambian de
  intensidad relativa (p. ej. Fe II vs Fe I, Sr II vs Fe I).
* En estrellas O/B el perfil P Cygni de He II 4686 revela viento estelar
  y pérdida de masa, propios de supergigantes.

Referencias
-----------
* Gray & Corbally (2009) — *Stellar Spectral Classification*, Princeton UP.
* Jaschek & Jaschek (1987) — *The Classification of Stars*, Cambridge UP.
* Morgan, Keenan & Kellman (1943) — Atlas MK original.
* Díaz et al. (2011) — Calibración Ca I / Fe I para clases G-K.

Uso
---
    from src.luminosity_classification import estimate_luminosity_class

    lum = estimate_luminosity_class(measurements, "G5")
    # → "III"

    mk = combine_spectral_and_luminosity("G5", "III")
    # → "G5III"
"""

from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# Tipo interno: measurements es el dict devuelto por measure_diagnostic_lines
# donde cada valor tiene la forma {"ew": float, "depth": float, ...}
# ---------------------------------------------------------------------------
Measurements = Dict[str, dict]


# ============================================================================
# Función auxiliar: extrae EW con fallback seguro
# ============================================================================

def _ew(measurements: Measurements, key: str, default: float = 0.0) -> float:
    """
    Extrae el ancho equivalente (EW) de una línea de forma segura.

    Acepta tanto el formato interno de SpectroClass
    (``{"ew": float, "depth": float, ...}``) como diccionarios planos
    ``{"HeII_4686": float, ...}`` (formato externo/tests).

    Parameters
    ----------
    measurements : dict
        Diccionario de mediciones.
    key : str
        Nombre de la línea (p. ej. ``"He_II_4686"``).
    default : float
        Valor devuelto si la clave no existe o el EW es None.

    Returns
    -------
    float
        EW en Ångströms.
    """
    entry = measurements.get(key)
    if entry is None:
        return default
    if isinstance(entry, dict):
        return float(entry.get('ew', default) or default)
    # Formato plano: el valor ya es el EW
    return float(entry)


# ============================================================================
# Reglas por grupo espectral
# ============================================================================

def _luminosity_OB(measurements: Measurements) -> str:
    """
    Clase de luminosidad para estrellas O y B.

    Indicador principal: comportamiento de He II 4686 Å (efecto Of).

    Fundamento
    ----------
    * En **supergigantes** (I) el viento estelar intenso invierte la línea
      He II 4686 hacia emisión o rellena la absorción, resultando en EW < 0
      o cercano a cero. Triplete N III 4634-4642 también en emisión.
    * En **gigantes** (III) la absorción He II 4686 es débil (EW 0–0.1 Å).
    * En **enanas** (V) la absorción He II 4686 es clara (EW > 0.3 Å).

    Indicadores secundarios
    -----------------------
    * He II 4686 vs He I 4713 (O tardías O8-O9.7): Gray & Corbally 2009 §3.2
      - Clase V: He II 4686 >> He I 4713
      - Clase III: He II 4686 > He I 4713
      - Clase II: He II 4686 ≈ He I 4713
      - Clase Ib: He II 4686 < He I 4713
      - Clase Ia: He II 4686 en emisión (EW < 0)
    * Si IV 4116 / He I 4121: luminosidad en B0–B0.7; Si IV crece con lum.
    * O II 4070-4076 / He I 4026: luminosidad B1-B4 (crece en supergigantes).
    * Si IV 4089 / He I 4471: más fuerte en supergigantes O tempranas.
    * O II 4591 + 4596: más fuerte en estrellas O-B luminosas.

    Returns
    -------
    str
        "Ia", "Ib", "II", "III", "IV" o "V".
    """
    he2_4686 = _ew(measurements, 'He_II_4686')
    he2_4542 = _ew(measurements, 'He_II_4542')

    # Promedio He II disponible
    he2_vals = [v for v in (he2_4686, he2_4542) if v != 0.0]
    he2_avg  = sum(he2_vals) / len(he2_vals) if he2_vals else 0.0

    # Indicadores secundarios O tempranas
    si4_4089  = _ew(measurements, 'Si_IV_4089')
    o2_4591   = _ew(measurements, 'O_II_4591')
    o2_4596   = _ew(measurements, 'O_II_4596')
    o2_4070   = _ew(measurements, 'O_II_4070')
    o2_4076   = _ew(measurements, 'O_II_4076')
    o2_avg    = (o2_4591 + o2_4596) / 2.0
    o2_blue   = (o2_4070 + o2_4076) / 2.0   # criterio B1-B4

    # He I 4713: criterio clave para O tardías (O8-O9.7)
    he1_4713  = _ew(measurements, 'He_I_4713')

    # Si IV 4116 / He I 4121: criterio B0-B0.7
    si4_4116  = _ew(measurements, 'Si_IV_4116')
    he1_4121  = _ew(measurements, 'He_I_4121')
    si4_he1_121 = si4_4116 / (he1_4121 + 0.01) if he1_4121 > 0.005 else 0.0

    # ── Regla principal: He II 4686 (efecto Of) ──────────────────────────
    if he2_4686 < -0.05:
        # EW negativo → línea en emisión (viento estelar) → supergigante
        if o2_avg > 0.40 or si4_4089 > 0.30 or o2_blue > 0.30:
            return 'Ia'   # supergigante muy luminosa
        return 'Ib'       # supergigante moderada

    if he2_avg < 0.05:
        # Absorción débil o rellena → gigante
        return 'III'

    # ── Regla secundaria: He II 4686 vs He I 4713 (O tardías) ────────────
    if he1_4713 > 0.02:   # He I 4713 medible → O tardía, aplicar criterio
        ratio_he = he2_4686 / (he1_4713 + 0.01)
        if ratio_he < 0.7:
            return 'Ib'   # He II 4686 < He I 4713
        if ratio_he < 1.1:
            return 'II'   # He II 4686 ≈ He I 4713
        if ratio_he < 2.5:
            return 'III'  # He II 4686 > He I 4713 pero no muy fuerte
        if ratio_he < 5.0:
            return 'IV'
        return 'V'

    # ── Regla terciaria: O II 4070-4076 / He I 4026 (B1-B4) ─────────────
    if o2_blue > 0.20:
        if o2_blue > 0.40:
            return 'Ib'
        if o2_blue > 0.30:
            return 'II'
        return 'III'

    # ── Regla cuaternaria: Si IV / He I (B0-B0.7) ────────────────────────
    if si4_he1_121 > 1.5:
        return 'Ib'
    if si4_he1_121 > 0.8:
        return 'III'

    # Absorción moderada-fuerte → distinguir IV de V con indicadores extra
    if o2_avg > 0.20 and si4_4089 > 0.10:
        return 'IV'

    return 'V'


def _luminosity_AF(measurements: Measurements) -> str:
    """
    Clase de luminosidad para estrellas A y F.

    Indicador principal: razón Sr II 4077 / Fe I 4071
    (A tempranas y F tardías, F6-F9).

    Indicador complementario (A tardías y F tempranas, A7-F5):
    blends Fe II / Ti II en 4172-4178 Å y 4395-4400 Å.

    Fundamento
    ----------
    * Sr II 4077 (resonante de estroncio ionizado): más fuerte a menor log g.
      La razón Sr II / Fe I aumenta de V → I.
    * Blends Fe II + Ti II en 4172-4178 y 4395-4400 Å: crecen
      notablemente en supergigantes A tardías y F tempranas (Gray & Corbally
      2009 §4.3). Se evalúan frente a Fe I 4046 (referencia neutral).
    * Balmer: supergigantes A/F tienen Balmer más angostos (efecto Stark
      reducido a baja presión). Indicador cualitativo de último recurso.

    Returns
    -------
    str
        "Ia", "Ib", "II", "III", "IV" o "V".
    """
    sr2_4077  = _ew(measurements, 'Sr_II_4077')
    sr2_4216  = _ew(measurements, 'Sr_II_4216')
    fe1_4071  = _ew(measurements, 'Fe_I_4071')
    fe1_4046  = _ew(measurements, 'Fe_I_4046')
    fe1_4383  = _ew(measurements, 'Fe_I_4383')

    # Blends Fe II / Ti II (luminosidad A7-F5)
    ti2_4178  = _ew(measurements, 'Ti_II_4178')
    ti2_4399  = _ew(measurements, 'Ti_II_4399')
    ti2_4444  = _ew(measurements, 'Ti_II_4444')
    tifeii_avg = (ti2_4178 + ti2_4399) / 2.0

    # Referencia neutral Fe I (la más estable)
    fe1_ref = fe1_4046 if fe1_4046 > 0.005 else (fe1_4383 if fe1_4383 > 0.005 else 0.0)

    # ── Criterio 1: Sr II 4077 / Fe I 4071 (primario para toda A-F) ──────
    sr2_available = sr2_4077 > 0.01 or sr2_4216 > 0.01
    sr2_best      = max(sr2_4077, sr2_4216)
    fe1_available = fe1_4071 > 0.005 or fe1_ref > 0.005
    fe1_best      = fe1_4071 if fe1_4071 > 0.005 else fe1_ref

    if sr2_available and fe1_available:
        ratio_sr = sr2_best / (fe1_best + 0.01)
        if ratio_sr > 6.0:
            return 'Ia'
        if ratio_sr > 4.0:
            return 'Ib'
        if ratio_sr > 2.0:
            return 'III'
        if ratio_sr > 1.0:
            return 'IV'
        return 'V'

    # ── Criterio 2: Blends Fe II/Ti II / Fe I (A tardía-F temprana) ──────
    if fe1_ref > 0.005 and tifeii_avg > 0.005:
        ratio_ti = tifeii_avg / (fe1_ref + 0.01)
        if ratio_ti > 3.0:
            return 'Ia'
        if ratio_ti > 2.0:
            return 'Ib'
        if ratio_ti > 1.2:
            return 'III'
        if ratio_ti > 0.6:
            return 'IV'
        return 'V'

    # ── Indicadores de último recurso ────────────────────────────────────
    mg2_4481 = _ew(measurements, 'Mg_II_4481')
    h_beta   = _ew(measurements, 'H_beta')
    h_gamma  = _ew(measurements, 'H_gamma')
    h_delta  = _ew(measurements, 'H_delta')
    h_avg    = (h_beta + h_gamma + h_delta) / 3.0 if (h_beta + h_gamma + h_delta) > 0 else 0.0

    if h_avg > 8.0:
        return 'V'
    if h_avg > 5.0:
        return 'IV'
    if mg2_4481 > 0.4:
        return 'V'

    return 'V'   # defecto conservador


def _luminosity_GK(measurements: Measurements) -> str:
    """
    Clase de luminosidad para estrellas G y K.

    Indicador principal: razón Y II 4376 / Fe I 4383
    (mejor indicador de luminosidad G-K, independiente de metalicidad).

    Indicador secundario: razón Ca I 4227 / Fe I 4383 (Díaz et al. 2011).

    Fundamento
    ----------
    * Y II 4376 (ítrio ionizado): crece con la luminosidad (menor log g)
      porque los iones se ven favorecidos en atmósferas menos densas.
      Es el criterio más seguro porque no depende de la metalicidad global
      (Gray & Corbally 2009 §5.1, §6.1).
    * Ca I 4227 (resonante de calcio neutro): se debilita a menor log g
      porque la ionización Ca I → Ca II aumenta. La razón Ca I / Fe I
      disminuye de V → I (Díaz et al. 2011).
    * CN λ4215: bandas moleculares con fuerte efecto de luminosidad positivo
      (gigantes y supergigantes tienen CN más intenso).

    Referencias
    -----------
    * Gray & Corbally (2009), §5.1 y §6.1.
    * Díaz et al. (2011), AJ 141, 171.

    Returns
    -------
    str
        "Ia", "Ib", "II", "III", "IV" o "V".
    """
    y2_4376   = _ew(measurements, 'Y_II_4376')
    fe1_4383  = _ew(measurements, 'Fe_I_4383')
    ca1_4227  = _ew(measurements, 'Ca_I_4227')
    cn_4215   = _ew(measurements, 'CN_4215')

    # Si no hay Fe I 4383, usar alternativa
    if fe1_4383 < 0.01:
        fe1_4383 = _ew(measurements, 'Fe_I_4046')

    if fe1_4383 < 0.01 and ca1_4227 < 0.01 and y2_4376 < 0.005:
        return 'V'

    # ── Criterio 1: Y II 4376 / Fe I 4383 (primario) ─────────────────────
    if y2_4376 > 0.005 and fe1_4383 > 0.01:
        ratio_y = y2_4376 / (fe1_4383 + 0.01)
        if ratio_y > 1.2:
            return 'Ia'
        if ratio_y > 0.8:
            return 'Ib'
        if ratio_y > 0.5:
            return 'II'
        if ratio_y > 0.3:
            return 'III'
        if ratio_y > 0.15:
            return 'IV'
        return 'V'

    # ── Criterio 2: Ca I 4227 / Fe I 4383 (secundario) ───────────────────
    if ca1_4227 > 0.01 and fe1_4383 > 0.01:
        ratio_ca = ca1_4227 / (fe1_4383 + 0.01)
        # CN 4215 como voto extra de luminosidad (positivo en gigantes)
        lum_vote = 0
        if cn_4215 > 0.05:
            lum_vote -= 1   # CN fuerte → más luminosa

        # Umbrales empíricos (Díaz et al. 2011; Gray & Corbally 2009)
        if ratio_ca < 0.3:
            base = 'Ia'
        elif ratio_ca < 0.5:
            base = 'Ib'
        elif ratio_ca < 0.8:
            base = 'III'
        elif ratio_ca < 1.5:
            base = 'IV'
        else:
            base = 'V'

        # Aplicar voto CN (puede bajar un nivel de luminosidad)
        if lum_vote < 0:
            escalon = {'V': 'IV', 'IV': 'III', 'III': 'Ib', 'Ib': 'Ia', 'Ia': 'Ia'}
            return escalon.get(base, base)
        return base

    return 'V'


def _luminosity_M(measurements: Measurements) -> str:
    """
    Clase de luminosidad para estrellas M.

    Indicadores (por orden de fiabilidad)
    --------------------------------------
    1. Ca I 4226 — efecto negativo: MUCHO más fuerte en enanas que en gigantes.
    2. Bandas CaH (6908, 6946, 6382 Å) — presentes en enanas, desaparecen
       en gigantes. Son los mejores discriminadores enana/gigante en óptico.
    3. MgH ~4770 Å — blend MgH+TiO; MgH domina en enanas (alta presión).
    4. Ratio TiO / metales — supergigantes M tienen TiO débil para su subtipo.

    Fundamento
    ----------
    * Ca I 4227 y CaH se ven favorecidos por la alta gravedad de las enanas
      (mayor presión electrónica mantiene Ca neutro e hidruro estable).
    * En supergigantes M (Ia-Ib) las bandas TiO son más débiles que en
      gigantes del mismo subtipo porque T_eff es mayor.

    References
    ----------
    * Gray & Corbally (2009) §7.1–7.3.
    * Keenan & McNeil (1989) Perkins Observatory tables.

    Returns
    -------
    str
        "Ia", "Ib", "II", "III" o "V".
    """
    tio_4955  = _ew(measurements, 'TiO_4955')
    tio_5167  = _ew(measurements, 'TiO_5167')
    tio_6651  = _ew(measurements, 'TiO_6651')
    ca1_4227  = _ew(measurements, 'Ca_I_4227')
    fe1_4957  = _ew(measurements, 'Fe_I_4957')

    # Bandas CaH — efecto de luminosidad negativo (fuerte en enanas)
    cah_6908  = _ew(measurements, 'CaH_6908')
    cah_6946  = _ew(measurements, 'CaH_6946')
    cah_6382  = _ew(measurements, 'CaH_6382')
    cah_6750  = _ew(measurements, 'CaH_6750')
    cah_total = cah_6908 + cah_6946 + cah_6382 + cah_6750

    # MgH 4770 — efecto negativo (fuerte en enanas de alta gravedad)
    mgh_4770  = _ew(measurements, 'MgH_4770')

    # TiO total
    tio_total = tio_4955 + tio_5167 + tio_6651

    # ── Criterio 1: CaH total (mejor discriminador enana/gigante) ─────────
    if cah_total > 0.05:
        # CaH presente y medible → enana (o subgigante)
        if cah_total > 0.3 or ca1_4227 > 0.5 or mgh_4770 > 0.2:
            return 'V'
        return 'IV'

    # ── Criterio 2: Ca I 4226 y MgH (efecto negativo) ────────────────────
    if ca1_4227 > 0.5:
        # Ca I muy fuerte → enana M
        return 'V'
    if mgh_4770 > 0.15:
        # MgH visible → enana o subgigante
        return 'V'

    # ── Criterio 3: TiO total vs metales (luminosidad positiva) ──────────
    metals = ca1_4227 + fe1_4957
    if tio_total > 0.5 and metals < 0.25:
        # TiO fuerte + metales débiles → supergigante
        if tio_total > 1.0:
            return 'Ia'
        return 'Ib'

    if tio_total > 0.25:
        return 'III'

    # TiO débil sin CaH significativo → probablemente K tardía o M temprana
    return 'V'


# ============================================================================
# Función principal pública
# ============================================================================

def estimate_luminosity_class(
    measurements: Measurements,
    spectral_type: str
) -> str:
    """
    Estima la clase de luminosidad MK a partir de los anchos equivalentes
    y el tipo espectral ya determinado.

    El tipo espectral es necesario porque cada grupo de temperatura usa
    indicadores de gravedad diferentes (ver sección de reglas más abajo).

    Parameters
    ----------
    measurements : dict
        Diccionario de mediciones de líneas tal como lo devuelve
        ``measure_diagnostic_lines()``.  Cada entrada puede ser:

        * Formato interno SpectroClass: ``{"ew": float, "depth": float}``
        * Formato plano (tests / uso externo): ``{"HeII_4686": 0.32}``

    spectral_type : str
        Tipo espectral de la estrella.  Se usa únicamente la primera letra
        mayúscula (ej. ``"B2V"`` → ``"B"``).  Aceptado en cualquier
        capitalización.

    Returns
    -------
    str
        Clase de luminosidad MK:

        * ``"Ia"``  – supergigante muy luminosa
        * ``"Ib"``  – supergigante moderada
        * ``"II"``  – gigante brillante
        * ``"III"`` – gigante
        * ``"IV"``  – subgigante
        * ``"V"``   – secuencia principal

        Devuelve ``"V"`` si no hay indicadores suficientes.

    Examples
    --------
    >>> from src.luminosity_classification import estimate_luminosity_class
    >>> ew = {"He_II_4686": {"ew": 0.4}}
    >>> estimate_luminosity_class(ew, "O7")
    'V'

    >>> ew = {"He_II_4686": {"ew": -0.2}}
    >>> estimate_luminosity_class(ew, "B1")
    'Ib'

    >>> ew = {"Ca_I_4227": {"ew": 0.7}, "Fe_I_4383": {"ew": 0.6}}
    >>> estimate_luminosity_class(ew, "G5")
    'IV'
    """
    if not spectral_type:
        return 'V'

    # Tomar sólo la letra del grupo de temperatura (ignorar subtipo y
    # posible clase de luminosidad ya presente en la cadena)
    letter = spectral_type.strip()[0].upper()

    if letter in ('O', 'B'):
        return _luminosity_OB(measurements)

    if letter in ('A', 'F'):
        return _luminosity_AF(measurements)

    if letter in ('G', 'K'):
        return _luminosity_GK(measurements)

    if letter == 'M':
        return _luminosity_M(measurements)

    # Tipos no convencionales (W, L, T, C, S, etc.) → sin clasificación
    return 'V'


# ============================================================================
# Función combinadora (helper público)
# ============================================================================

def combine_spectral_and_luminosity(
    spectral_type: str,
    luminosity_class: str
) -> str:
    """
    Combina el tipo espectral y la clase de luminosidad en la notación
    estándar MK.

    Parameters
    ----------
    spectral_type : str
        Tipo espectral, posiblemente ya con sufijo de emisión
        (ej. ``"B2e"``, ``"O7"``, ``"G5"``).
    luminosity_class : str
        Clase de luminosidad devuelta por ``estimate_luminosity_class``
        (ej. ``"V"``, ``"III"``, ``"Ia"``).

    Returns
    -------
    str
        Clasificación MK completa (ej. ``"B2eIa"``, ``"G5III"``,
        ``"O7V"``).

    Examples
    --------
    >>> combine_spectral_and_luminosity("G2", "V")
    'G2V'
    >>> combine_spectral_and_luminosity("K2", "I")
    'K2I'
    >>> combine_spectral_and_luminosity("B2e", "Ia")
    'B2eIa'
    """
    spectral_type   = spectral_type.strip()
    luminosity_class = luminosity_class.strip()

    # Evitar duplicar la clase si ya está añadida (caso re-llamada)
    roman = ('Ia', 'Ib', 'II', 'III', 'IV', 'V')
    for r in roman:
        if spectral_type.endswith(r):
            return spectral_type   # ya tiene clase, no duplicar

    return spectral_type + luminosity_class
