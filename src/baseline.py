"""
Linea base: busqueda aleatoria uniforme con presupuesto de evaluaciones fijo.

Justificacion. Un algoritmo genetico sin referencia no se puede juzgar: una
tasa de exito del 95% no dice nada si muestrear al azar la misma cantidad de
soluciones ya alcanza el 95%. Esta linea base fija el nivel que el AG debe
superar para justificar su complejidad.

Ademas sirve como prueba del arnes experimental. Como el muestreo es
uniforme sobre un espacio finito con un unico optimo, la probabilidad de
exito tiene forma cerrada:

    P(exito | k evaluaciones) = 1 - (1 - 1/|Omega|)^k

Si la tasa empirica simulada no cae dentro del intervalo de confianza de esa
prediccion analitica, hay un error en el codigo experimental, no un
resultado. Es un test, no un adorno.
"""

import random

from instancia import N_PROYECTOS, calcular_beneficio, calcular_costo, PRESUPUESTO


def probabilidad_exito_analitica(evaluaciones, n_optimos=1, espacio=2 ** N_PROYECTOS):
    """P(encontrar el optimo) tras k muestras uniformes con reemplazo."""
    return 1.0 - (1.0 - n_optimos / espacio) ** evaluaciones


def busqueda_aleatoria(evaluaciones, semilla):
    """Muestrea k individuos uniformes y devuelve el mejor factible.

    Devuelve (beneficio, costo, individuo, evaluacion_del_primer_optimo).
    El ultimo campo es None si nunca se alcanzo el optimo; se usa para
    comparar esfuerzo contra el AG en la misma moneda: evaluaciones.
    """
    rng = random.Random(semilla)
    mejor = (-1, 0, None)
    primer_optimo = None

    for k in range(1, evaluaciones + 1):
        individuo = [rng.randint(0, 1) for _ in range(N_PROYECTOS)]
        costo = calcular_costo(individuo)
        if costo > PRESUPUESTO:
            continue                      # se descarta: no es solucion valida
        beneficio = calcular_beneficio(individuo)
        if beneficio > mejor[0]:
            mejor = (beneficio, costo, individuo)
            if primer_optimo is None and beneficio == 100:
                primer_optimo = k

    return mejor[0], mejor[1], mejor[2], primer_optimo


if __name__ == "__main__":
    from exacto import optimo_por_enumeracion
    from metricas import intervalo_wilson

    b_optimo = optimo_por_enumeracion()[0]
    semillas = 200

    print("LINEA BASE: BUSQUEDA ALEATORIA UNIFORME")
    print(f"{'k evals':>8} {'analitica':>10} {'empirica':>10} {'IC Wilson 95%':>20} {'coherente':>10}")

    for evaluaciones in (500, 2000, 10000):
        exitos = sum(
            1
            for s in range(semillas)
            if busqueda_aleatoria(evaluaciones, s)[0] == b_optimo
        )
        p_emp = exitos / semillas
        p_teo = probabilidad_exito_analitica(evaluaciones)
        lo, hi = intervalo_wilson(exitos, semillas)
        coherente = lo <= p_teo <= hi
        print(
            f"{evaluaciones:>8} {p_teo:>10.4f} {p_emp:>10.4f} "
            f"   [{lo:.4f}, {hi:.4f}] {str(coherente):>10}"
        )
