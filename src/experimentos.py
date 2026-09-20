"""
Arnes experimental: factorial con presupuesto igualado, barrido de lambda,
configuraciones literales del enunciado y lineas base.

Todas las comparaciones usan el MISMO conjunto de semillas (diseno pareado,
decision D6), lo que permite contrastar configuraciones con la prueba exacta
de McNemar sobre los pares discordantes en lugar de tratar las muestras como
independientes.
"""

import json
import math
import os
import time

from baseline import busqueda_aleatoria, probabilidad_exito_analitica
from exacto import optimo_por_enumeracion, solucion_voraz
from genetico import ejecutar_algoritmo_genetico
from metricas import intervalo_wilson, resumen_censurado

SEMILLAS = list(range(100))            # decision D4
PRESUPUESTO_FIJO = 2000                # decision D6
B_OPTIMO = optimo_por_enumeracion()[0]
DIR_RESULTADOS = os.path.join(os.path.dirname(__file__), "..", "resultados")


# --------------------------------------------------------------------------
# Inferencia
# --------------------------------------------------------------------------

def _binomial_acumulada(k, n, p=0.5):
    """P(X <= k) para X ~ Binomial(n, p), sin dependencias externas."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def mcnemar_exacto(solo_a, solo_b):
    """Prueba exacta de McNemar para dos algoritmos sobre las mismas semillas.

    solo_a = semillas donde A acierta y B no; solo_b = al reves. Los casos en
    que ambos aciertan o ambos fallan no aportan informacion sobre la
    diferencia y se excluyen (pares concordantes).

    Bajo H0 (ambos algoritmos igual de buenos), cada par discordante es una
    moneda equilibrada. Devuelve el p-valor bilateral.
    """
    n = solo_a + solo_b
    if n == 0:
        return 1.0
    k = min(solo_a, solo_b)
    return min(1.0, 2 * _binomial_acumulada(k, n))


def holm(p_valores):
    """Correccion de Holm-Bonferroni para comparaciones multiples.

    Controla la tasa de error por familia sin la perdida de potencia de
    Bonferroni simple. Devuelve los p-valores ajustados en el orden original.
    """
    indexados = sorted(enumerate(p_valores), key=lambda t: t[1])
    m = len(p_valores)
    ajustados = [0.0] * m
    anterior = 0.0
    for rango, (idx, p) in enumerate(indexados):
        valor = min(1.0, (m - rango) * p)
        anterior = max(anterior, valor)      # monotonia
        ajustados[idx] = anterior
    return ajustados


# --------------------------------------------------------------------------
# Ejecucion de una celda experimental
# --------------------------------------------------------------------------

def correr_celda(etiqueta, semillas=SEMILLAS, historial=False, **config):
    """Ejecuta el AG con una configuracion sobre todas las semillas."""
    inicio = time.time()
    exitos_por_semilla, beneficios = [], []
    generaciones_acierto, evaluaciones_acierto = [], []
    infactibles = 0

    for s in semillas:
        r = ejecutar_algoritmo_genetico(
            semilla=s,
            beneficio_optimo=B_OPTIMO,
            registrar_historial=historial,
            **config,
        )
        acierto = r["mejor_factible"]["beneficio"] == B_OPTIMO
        exitos_por_semilla.append(acierto)
        beneficios.append(r["mejor_factible"]["beneficio"])
        generaciones_acierto.append(r["generacion_primer_optimo"])
        evaluaciones_acierto.append(r["evaluaciones_primer_optimo"])
        if not r["mejor_por_aptitud"]["factible"]:
            infactibles += 1

    n = len(semillas)
    exitos = sum(exitos_por_semilla)
    lo, hi = intervalo_wilson(exitos, n)

    return {
        "etiqueta": etiqueta,
        "config": config,
        "evaluaciones": config["n_poblacion"] * config["n_generaciones"],
        "n_semillas": n,
        "exitos": exitos,
        "tasa_exito": exitos / n,
        "ic_wilson": [lo, hi],
        "beneficio_medio": sum(beneficios) / n,
        "beneficio_minimo": min(beneficios),
        "tasa_infactible": infactibles / n,
        "primer_acierto_generacion": resumen_censurado(generaciones_acierto),
        "primer_acierto_evaluaciones": resumen_censurado(evaluaciones_acierto),
        "exitos_por_semilla": exitos_por_semilla,
        "segundos": time.time() - inicio,
    }


def correr_baseline(etiqueta, evaluaciones, semillas=SEMILLAS):
    """Linea base de busqueda aleatoria con presupuesto igualado."""
    exitos_por_semilla = []
    for s in semillas:
        b, _, _, _ = busqueda_aleatoria(evaluaciones, 10_000 + s)
        exitos_por_semilla.append(b == B_OPTIMO)
    n = len(semillas)
    exitos = sum(exitos_por_semilla)
    lo, hi = intervalo_wilson(exitos, n)
    return {
        "etiqueta": etiqueta,
        "evaluaciones": evaluaciones,
        "n_semillas": n,
        "exitos": exitos,
        "tasa_exito": exitos / n,
        "ic_wilson": [lo, hi],
        "tasa_exito_analitica": probabilidad_exito_analitica(evaluaciones),
        "exitos_por_semilla": exitos_por_semilla,
    }


def comparar_pareado(a, b):
    """Compara dos conjuntos de resultados sobre las mismas semillas."""
    ea, eb = a["exitos_por_semilla"], b["exitos_por_semilla"]
    solo_a = sum(1 for x, y in zip(ea, eb) if x and not y)
    solo_b = sum(1 for x, y in zip(ea, eb) if y and not x)
    return {
        "a": a["etiqueta"],
        "b": b["etiqueta"],
        "solo_a": solo_a,
        "solo_b": solo_b,
        "diferencia": a["tasa_exito"] - b["tasa_exito"],
        "p_mcnemar": mcnemar_exacto(solo_a, solo_b),
    }


# --------------------------------------------------------------------------
# Experimentos
# --------------------------------------------------------------------------

BASE = {"p_cruce": 0.80, "tam_torneo": 3, "n_elite": 1, "lambda_penalizacion": 5.0}


def experimento_factorial():
    """3x3: poblacion x mutacion, con presupuesto fijo de 2000 evaluaciones.

    Las generaciones se despejan como PRESUPUESTO_FIJO / N para que el
    tamano de poblacion quede aislado del cómputo disponible (D6).
    """
    celdas = []
    for n_pob in (10, 20, 50):
        for p_mut in (0.01, 0.05, 0.10):
            celdas.append(
                correr_celda(
                    f"N={n_pob}, pm={p_mut}",
                    n_poblacion=n_pob,
                    n_generaciones=PRESUPUESTO_FIJO // n_pob,
                    p_mutacion=p_mut,
                    **BASE,
                )
            )
    return celdas


def experimento_enunciado():
    """Configuraciones A, B y C tal como las define el enunciado."""
    definiciones = [
        ("A (enunciado)", 10, 50, 0.01),
        ("B (enunciado)", 20, 100, 0.05),
        ("C (enunciado)", 50, 200, 0.10),
    ]
    return [
        correr_celda(e, n_poblacion=n, n_generaciones=g, p_mutacion=pm, **BASE)
        for e, n, g, pm in definiciones
    ]


def experimento_lambda():
    """Barrido de la penalizacion con N=20, G=100 (presupuesto fijo)."""
    valores = [0.0, 0.5, 1.0, 1.5, 1.9, 2.1, 3.0, 5.0, 10.0, 28.0]
    base_sin_lambda = {k: v for k, v in BASE.items() if k != "lambda_penalizacion"}
    return [
        correr_celda(
            f"lambda={lam}",
            n_poblacion=20,
            n_generaciones=100,
            p_mutacion=0.05,
            lambda_penalizacion=lam,
            **base_sin_lambda,
        )
        for lam in valores
    ]


def main():
    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    inicio = time.time()

    factorial = experimento_factorial()
    enunciado = experimento_enunciado()
    barrido = experimento_lambda()
    bases = [correr_baseline(f"aleatoria k={k}", k) for k in (500, 2000, 10000)]
    voraz = solucion_voraz()

    # Comparaciones pareadas AG vs busqueda aleatoria al mismo presupuesto.
    emparejados = {500: bases[0], 2000: bases[1], 10000: bases[2]}
    comparaciones = [
        comparar_pareado(c, emparejados[c["evaluaciones"]])
        for c in factorial + enunciado
        if c["evaluaciones"] in emparejados
    ]
    for comp, p_aj in zip(comparaciones, holm([c["p_mcnemar"] for c in comparaciones])):
        comp["p_holm"] = p_aj

    resultados = {
        "b_optimo": B_OPTIMO,
        "n_semillas": len(SEMILLAS),
        "voraz": {"beneficio": voraz[0], "costo": voraz[1], "cromosoma": voraz[2]},
        "factorial": factorial,
        "enunciado": enunciado,
        "barrido_lambda": barrido,
        "baselines": bases,
        "comparaciones": comparaciones,
        "segundos_total": time.time() - inicio,
    }

    ruta = os.path.join(DIR_RESULTADOS, "resultados.json")
    with open(ruta, "w") as f:
        json.dump(resultados, f, indent=2)
    print(f"Resultados guardados en {ruta} ({resultados['segundos_total']:.1f} s)")
    return resultados


if __name__ == "__main__":
    main()
