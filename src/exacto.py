"""
Verdad de terreno: resolucion exacta de la instancia por tres metodos
independientes, mas la caracterizacion del paisaje de busqueda.

Motivacion. Todas las metricas del trabajo (tasa de exito, generacion del
primer acierto, tasa de infactibilidad) se definen respecto al optimo B*.
Un error en B* invalidaria el analisis completo, asi que se calcula por
enumeracion exhaustiva y se verifica de forma cruzada con programacion
dinamica, que recorre el espacio con una logica distinta.

El algoritmo voraz NO se usa como verdad de terreno: en la mochila 0-1 no
garantiza optimalidad. Se calcula para comprobar empiricamente si en ESTA
instancia coincide con el optimo, que es una de las preguntas del trabajo.
"""

from instancia import (
    BENEFICIOS,
    COSTOS,
    NOMBRES,
    N_PROYECTOS,
    PRESUPUESTO,
    calcular_beneficio,
    calcular_costo,
    mascara_a_individuo,
    proyectos_seleccionados,
)


def enumerar_espacio():
    """Evalua las 2^n soluciones del espacio.

    Devuelve dos listas de tuplas (beneficio, costo, individuo): las
    factibles y las infactibles. Solo es viable porque n = 10.
    """
    factibles, infactibles = [], []
    for mascara in range(2 ** N_PROYECTOS):
        individuo = mascara_a_individuo(mascara)
        costo = calcular_costo(individuo)
        beneficio = calcular_beneficio(individuo)
        registro = (beneficio, costo, individuo)
        if costo <= PRESUPUESTO:
            factibles.append(registro)
        else:
            infactibles.append(registro)
    return factibles, infactibles


def optimo_por_enumeracion():
    """Optimo global por fuerza bruta. Devuelve (B*, C*, individuo)."""
    factibles, _ = enumerar_espacio()
    return max(factibles, key=lambda r: r[0])


def optimo_por_programacion_dinamica():
    """Optimo por PD en O(n*W); verificacion cruzada de la enumeracion.

    tabla[i][w] = beneficio maximo usando los primeros i proyectos con
    presupuesto w. La reconstruccion recupera el individuo elegido.
    """
    n, W = N_PROYECTOS, PRESUPUESTO
    tabla = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        c_i, b_i = COSTOS[i - 1], BENEFICIOS[i - 1]
        for w in range(W + 1):
            # Opcion 1: no incluir el proyecto i.
            mejor = tabla[i - 1][w]
            # Opcion 2: incluirlo, si cabe en el presupuesto restante.
            if c_i <= w:
                mejor = max(mejor, tabla[i - 1][w - c_i] + b_i)
            tabla[i][w] = mejor

    # Reconstruccion hacia atras: el proyecto i se incluyo si la celda
    # cambio respecto a la fila anterior.
    individuo = [0] * n
    w = W
    for i in range(n, 0, -1):
        if tabla[i][w] != tabla[i - 1][w]:
            individuo[i - 1] = 1
            w -= COSTOS[i - 1]

    return tabla[n][W], calcular_costo(individuo), individuo


def solucion_voraz():
    """Heuristica voraz por ratio beneficio/costo, en O(n log n).

    Ordena los proyectos por b_i/c_i descendente y toma cada uno mientras
    quepa. Es la heuristica clasica de la mochila fraccionaria; en la
    version 0-1 no es optima en general.
    """
    orden = sorted(range(N_PROYECTOS), key=lambda i: -BENEFICIOS[i] / COSTOS[i])
    individuo = [0] * N_PROYECTOS
    costo = 0
    for i in orden:
        if costo + COSTOS[i] <= PRESUPUESTO:
            individuo[i] = 1
            costo += COSTOS[i]
    return calcular_beneficio(individuo), costo, individuo


def lambda_critico_exacto():
    """Umbral exacto de penalizacion para esta instancia.

    La aptitud penalizada es f_lambda(X) = B(X) - lambda*max(0, C(X)-W).
    El optimo global de f_lambda es factible si y solo si ninguna solucion
    infactible supera a B*, es decir si

        lambda > max_{X infactible} (B(X) - B*) / (C(X) - W).

    Ese maximo se calcula aqui de forma exacta porque disponemos de todas
    las soluciones infactibles.
    """
    factibles, infactibles = enumerar_espacio()
    b_optimo = max(b for b, _, _ in factibles)

    peor = max(
        ((b - b_optimo) / (c - PRESUPUESTO), b, c, ind) for b, c, ind in infactibles
    )
    ratio, b, c, ind = peor
    return ratio, (b, c, ind)


def cota_suficiente_general():
    """Cota lambda >= max(b_i) que garantiza factibilidad del optimo penalizado.

    Con costos enteros positivos, quitar un proyecto i de una solucion
    infactible con exceso E cambia la aptitud en -b_i + lambda*min(E, c_i).
    Como c_i >= 1 y E >= 1, se tiene min(E, c_i) >= 1, luego el cambio es
    >= lambda - b_i >= 0 cuando lambda >= max(b_i). Por induccion se llega
    a una solucion factible sin empeorar la aptitud.
    """
    return max(BENEFICIOS)


def caracterizar_paisaje():
    """Estadisticos del espacio de busqueda usados en el informe."""
    factibles, infactibles = enumerar_espacio()
    beneficios = sorted((b for b, _, _ in factibles), reverse=True)
    b_optimo = beneficios[0]
    ratios = [BENEFICIOS[i] / COSTOS[i] for i in range(N_PROYECTOS)]
    return {
        "total": 2 ** N_PROYECTOS,
        "factibles": len(factibles),
        "infactibles": len(infactibles),
        "fraccion_factible": len(factibles) / 2 ** N_PROYECTOS,
        "b_optimo": b_optimo,
        "n_optimos": sum(1 for b in beneficios if b == b_optimo),
        "a_menos_de_1": sum(1 for b in beneficios if b >= b_optimo - 1),
        "a_menos_de_3": sum(1 for b in beneficios if b >= b_optimo - 3),
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
    }


def _formatear(etiqueta, resultado):
    beneficio, costo, individuo = resultado
    proyectos = ", ".join(proyectos_seleccionados(individuo))
    return (
        f"  {etiqueta:<24} B={beneficio:3d}  C={costo:3d}  "
        f"X={''.join(str(g) for g in individuo)}  [{proyectos}]"
    )


if __name__ == "__main__":
    print("VERDAD DE TERRENO")
    enum = optimo_por_enumeracion()
    pd = optimo_por_programacion_dinamica()
    voraz = solucion_voraz()
    print(_formatear("Enumeracion (2^n)", enum))
    print(_formatear("Prog. dinamica O(nW)", pd))
    print(_formatear("Voraz por ratio", voraz))

    assert enum[0] == pd[0], "Enumeracion y PD discrepan: hay un error"
    print("\n  [OK] Enumeracion y programacion dinamica coinciden en B* =", enum[0])
    print("  Voraz alcanza el optimo:", voraz[0] == enum[0])

    print("\nPAISAJE DE BUSQUEDA")
    p = caracterizar_paisaje()
    print(f"  Espacio total                 : {p['total']}")
    print(f"  Soluciones factibles          : {p['factibles']} ({p['fraccion_factible']:.1%})")
    print(f"  Optimos globales (B = B*)     : {p['n_optimos']}")
    print(f"  Soluciones a <=1 del optimo   : {p['a_menos_de_1']}")
    print(f"  Soluciones a <=3 del optimo   : {p['a_menos_de_3']}")
    print(f"  Ratios b_i/c_i                : [{p['ratio_min']:.3f}, {p['ratio_max']:.3f}]")

    print("\nPENALIZACION")
    lam, (b, c, ind) = lambda_critico_exacto()
    print(f"  lambda* exacto de la instancia: {lam:.4f}")
    print(f"    alcanzado en B={b}, C={c}, [{', '.join(proyectos_seleccionados(ind))}]")
    print(f"  Cota suficiente general       : lambda >= max(b_i) = {cota_suficiente_general()}")
    print(f"  Valor sugerido por el enunciado: lambda = 5")
