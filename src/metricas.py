"""
Metricas e inferencia estadistica, implementadas sin dependencias externas.

Se evita numpy/scipy a proposito: el enunciado restringe las bibliotecas y
estas funciones son cortas. Programarlas obliga a justificar la eleccion del
estimador, que es parte de lo que se evalua.
"""

import math

Z_95 = 1.959963984540054   # cuantil normal bilateral al 95%


def intervalo_wilson(exitos, ensayos, z=Z_95):
    """Intervalo de confianza de Wilson para una proporcion.

    Se usa Wilson y no la aproximacion normal (Wald) porque el regimen de
    interes de este trabajo son tasas de exito cercanas a 1, donde Wald
    produce intervalos que se salen de [0,1] y subestima la incertidumbre.
    Wilson permanece dentro del rango y mantiene cobertura con n moderado.
    """
    if ensayos == 0:
        return 0.0, 1.0
    p = exitos / ensayos
    denominador = 1 + z ** 2 / ensayos
    centro = (p + z ** 2 / (2 * ensayos)) / denominador
    margen = (
        z
        / denominador
        * math.sqrt(p * (1 - p) / ensayos + z ** 2 / (4 * ensayos ** 2))
    )
    return max(0.0, centro - margen), min(1.0, centro + margen)


def mediana(valores):
    """Mediana de una lista no vacia."""
    v = sorted(valores)
    n = len(v)
    if n == 0:
        return None
    medio = n // 2
    return v[medio] if n % 2 else (v[medio - 1] + v[medio]) / 2


def resumen_censurado(valores, censura=None):
    """Resumen de una variable con datos censurados (p. ej. generacion del
    primer acierto, indefinida en las corridas que nunca aciertan).

    Se reporta la mediana de los casos observados y la fraccion censurada.
    NO se reporta la media: con censura a la derecha la media muestral de
    los casos observados es un estimador sesgado hacia abajo.
    """
    observados = [v for v in valores if v is not censura and v is not None]
    return {
        "n": len(valores),
        "observados": len(observados),
        "fraccion_censurada": 1 - len(observados) / len(valores) if valores else 0.0,
        "mediana": mediana(observados),
        "minimo": min(observados) if observados else None,
    }
