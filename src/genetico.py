"""
Algoritmo genetico para la seleccion optima de proyectos.

Contiene las nueve funciones exigidas por el enunciado, con los nombres
literales solicitados. calcular_costo y calcular_beneficio se reexportan
desde instancia.py para que las nueve esten disponibles en este modulo sin
duplicar la definicion de los datos.
"""

import random

from instancia import (
    N_PROYECTOS,
    PRESUPUESTO,
    calcular_beneficio,
    calcular_costo,
    proyectos_seleccionados,
)

# Configuracion inicial del enunciado.
CONFIG_ENUNCIADO = {
    "n_poblacion": 20,
    "n_generaciones": 100,
    "p_cruce": 0.80,
    "p_mutacion": 0.05,
    "tam_torneo": 3,
    "n_elite": 1,
    "lambda_penalizacion": 5.0,
}


# --------------------------------------------------------------------------
# Representacion y poblacion inicial
# --------------------------------------------------------------------------

def generar_individuo(rng):
    """Cromosoma binario de 10 genes, cada gen equiprobable.

    El muestreo uniforme sobre {0,1}^10 implica un costo esperado de
    sum(c_i)/2 = 47.5, apenas por debajo del presupuesto: cerca de la mitad
    de la poblacion inicial sera infactible. Esto es deliberado, no un
    defecto: sesgar la inicializacion hacia soluciones factibles reduciria
    la diversidad inicial y ocultaria el efecto de la penalizacion, que es
    justamente uno de los objetos de estudio.
    """
    return [rng.randint(0, 1) for _ in range(N_PROYECTOS)]


def generar_poblacion(n_poblacion, rng):
    """Poblacion inicial de tamano configurable."""
    return [generar_individuo(rng) for _ in range(n_poblacion)]


# --------------------------------------------------------------------------
# Funcion de aptitud
# --------------------------------------------------------------------------

def calcular_aptitud(individuo, lambda_penalizacion=5.0):
    """f_lambda(X) = B(X) - lambda * max(0, C(X) - W).

    Penalizacion lineal exterior. El umbral exacto de esta instancia es
    lambda* = 1.9: por debajo, el optimo global del paisaje penalizado es
    infactible (ver exacto.lambda_critico_exacto). Una cota suficiente
    general y demostrable es lambda >= max(b_i) = 28.
    """
    costo = calcular_costo(individuo)
    beneficio = calcular_beneficio(individuo)
    if costo <= PRESUPUESTO:
        return float(beneficio)
    return float(beneficio) - lambda_penalizacion * (costo - PRESUPUESTO)


# --------------------------------------------------------------------------
# Operadores geneticos
# --------------------------------------------------------------------------

def seleccionar_padre(poblacion, aptitudes, tam_torneo, rng):
    """Seleccion por torneo sobre aptitudes ya calculadas.

    Recibe las aptitudes en lugar de recalcularlas: el torneo no debe
    consumir presupuesto de evaluaciones, o el numero real de evaluaciones
    dejaria de ser n_poblacion * n_generaciones y las comparaciones entre
    configuraciones con presupuesto igualado quedarian invalidadas.

    El muestreo es sin reemplazo dentro del torneo (rng.sample), de modo que
    los tam_torneo participantes son siempre distintos.
    """
    indices = rng.sample(range(len(poblacion)), tam_torneo)
    ganador = max(indices, key=lambda i: aptitudes[i])
    return poblacion[ganador][:]


def cruzar(padre_1, padre_2, p_cruce, rng):
    """Cruce de un punto.

    El punto de corte se toma en {1, ..., n-1}. Se excluyen 0 y n porque
    producirian descendientes identicos a los padres, lo que haria que la
    probabilidad de cruce p_cruce dejara de medir recombinacion efectiva.

    Si no ocurre cruce, los descendientes son copias de los padres.
    """
    if rng.random() >= p_cruce:
        return padre_1[:], padre_2[:], None

    punto = rng.randint(1, N_PROYECTOS - 1)
    hijo_1 = padre_1[:punto] + padre_2[punto:]
    hijo_2 = padre_2[:punto] + padre_1[punto:]
    return hijo_1, hijo_2, punto


def mutar(individuo, p_mutacion, rng):
    """Mutacion binaria gen a gen.

    Cada gen se invierte de forma independiente con probabilidad p_mutacion,
    tal como especifica el enunciado. El numero esperado de genes mutados
    por individuo es n * p_mutacion: 0.1 para p_m = 0.01, 0.5 para 0.05 y
    1.0 para 0.10. Con p_m = 0.01 el 90% de los individuos pasa intacto.

    Modifica y devuelve el mismo individuo.
    """
    for i in range(len(individuo)):
        if rng.random() < p_mutacion:
            individuo[i] ^= 1
    return individuo


# --------------------------------------------------------------------------
# Diversidad (decision D10)
# --------------------------------------------------------------------------

def _distancia_hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def medir_diversidad(poblacion, mejor):
    """Diversidad de la poblacion en dos numeros.

    - individuos_distintos: cardinal del conjunto de cromosomas unicos.
    - hamming_media: distancia de Hamming media al mejor individuo.

    Permite responder la pregunta 5f del enunciado con una magnitud medible
    en lugar de con una impresion visual de la grafica: la convergencia
    prematura se manifiesta como colapso de ambas cantidades en las
    primeras generaciones.
    """
    distintos = len({tuple(ind) for ind in poblacion})
    hamming = sum(_distancia_hamming(ind, mejor) for ind in poblacion) / len(poblacion)
    return distintos, hamming


# --------------------------------------------------------------------------
# Algoritmo completo
# --------------------------------------------------------------------------

def ejecutar_algoritmo_genetico(
    n_poblacion=20,
    n_generaciones=100,
    p_cruce=0.80,
    p_mutacion=0.05,
    tam_torneo=3,
    n_elite=1,
    lambda_penalizacion=5.0,
    semilla=0,
    beneficio_optimo=None,
    registrar_historial=True,
):
    """Ejecuta el ciclo completo y devuelve el registro de la corrida.

    Modelo generacional con elitismo, segun el enunciado. Se devuelven dos
    soluciones distintas (decision D7):

    - mejor_por_aptitud: el individuo de mayor f_lambda, que con lambda por
      debajo del umbral critico puede ser INFACTIBLE. Es lo que el enunciado
      pide devolver y lo que hace visible el efecto de la penalizacion.
    - mejor_factible: el mejor individuo factible visto en toda la corrida.

    Reportar ambos es lo que permite medir la tasa de infactibilidad del
    resultado en funcion de lambda.
    """
    rng = random.Random(semilla)

    poblacion = generar_poblacion(n_poblacion, rng)
    evaluaciones = 0

    mejor_global = None          # (aptitud, individuo)
    mejor_factible = None        # (beneficio, costo, individuo)
    generacion_primer_optimo = None
    evaluaciones_primer_optimo = None
    historial = []

    for generacion in range(n_generaciones):
        # --- Evaluacion: una sola vez por generacion (ver seleccionar_padre)
        aptitudes = [calcular_aptitud(ind, lambda_penalizacion) for ind in poblacion]
        evaluaciones += n_poblacion

        idx_mejor = max(range(n_poblacion), key=lambda i: aptitudes[i])
        mejor_individuo = poblacion[idx_mejor]
        mejor_aptitud = aptitudes[idx_mejor]

        if mejor_global is None or mejor_aptitud > mejor_global[0]:
            mejor_global = (mejor_aptitud, mejor_individuo[:])

        # Mejor factible visto, independientemente de la aptitud penalizada.
        for ind, _ in zip(poblacion, aptitudes):
            costo = calcular_costo(ind)
            if costo <= PRESUPUESTO:
                beneficio = calcular_beneficio(ind)
                if mejor_factible is None or beneficio > mejor_factible[0]:
                    mejor_factible = (beneficio, costo, ind[:])

        # Primer acierto: primera generacion en que aparece el optimo global.
        if (
            beneficio_optimo is not None
            and generacion_primer_optimo is None
            and mejor_factible is not None
            and mejor_factible[0] == beneficio_optimo
        ):
            generacion_primer_optimo = generacion
            evaluaciones_primer_optimo = evaluaciones

        if registrar_historial:
            distintos, hamming = medir_diversidad(poblacion, mejor_individuo)
            historial.append(
                {
                    "generacion": generacion,
                    "mejor_aptitud": mejor_aptitud,
                    "aptitud_promedio": sum(aptitudes) / n_poblacion,
                    "mejor_beneficio": calcular_beneficio(mejor_individuo),
                    "mejor_costo": calcular_costo(mejor_individuo),
                    "mejor_cromosoma": "".join(str(g) for g in mejor_individuo),
                    "individuos_distintos": distintos,
                    "hamming_media": hamming,
                }
            )

        # --- Nueva generacion: elitismo + descendencia
        orden = sorted(range(n_poblacion), key=lambda i: -aptitudes[i])
        nueva = [poblacion[i][:] for i in orden[:n_elite]]

        while len(nueva) < n_poblacion:
            padre_1 = seleccionar_padre(poblacion, aptitudes, tam_torneo, rng)
            padre_2 = seleccionar_padre(poblacion, aptitudes, tam_torneo, rng)
            hijo_1, hijo_2, _ = cruzar(padre_1, padre_2, p_cruce, rng)
            for hijo in (hijo_1, hijo_2):
                if len(nueva) < n_poblacion:
                    nueva.append(mutar(hijo, p_mutacion, rng))

        poblacion = nueva

    aptitud_final, individuo_final = mejor_global
    return {
        "mejor_por_aptitud": {
            "cromosoma": individuo_final,
            "aptitud": aptitud_final,
            "beneficio": calcular_beneficio(individuo_final),
            "costo": calcular_costo(individuo_final),
            "factible": calcular_costo(individuo_final) <= PRESUPUESTO,
            "proyectos": proyectos_seleccionados(individuo_final),
        },
        "mejor_factible": {
            "cromosoma": mejor_factible[2],
            "beneficio": mejor_factible[0],
            "costo": mejor_factible[1],
            "proyectos": proyectos_seleccionados(mejor_factible[2]),
        }
        if mejor_factible
        else None,
        "generacion_primer_optimo": generacion_primer_optimo,
        "evaluaciones_primer_optimo": evaluaciones_primer_optimo,
        "evaluaciones": evaluaciones,
        "historial": historial,
    }
