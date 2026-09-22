"""
Traza manual exigida por el punto 3 del enunciado.

Decision: en lugar de inventar un ejemplo a mano, se reproduce paso a paso una
iteracion real del algoritmo con una semilla fija. Asi el ejemplo del informe es
verificable ejecutando el codigo, y cualquier discrepancia entre lo narrado y lo
implementado sale a la luz.

Sobre la eleccion de semilla: la semilla 42 que sugiere el enunciado produce una
traza degenerada (los dos padres comparten el sufijo, el cruce devuelve copias
identicas y no muta ningun gen), inservible como ejemplo didactico. Se usa la
semilla 26, escogida entre las primeras 500 por cumplir tres criterios fijados de
antemano: cruce con efecto real sobre ambos descendientes, exactamente una
mutacion (legible), y un padre factible y otro infactible, para que la
penalizacion se vea en accion. La eleccion se declara en el informe; el caso
degenerado de la semilla 42 se comenta aparte porque ilustra por que el cruce
entre padres parecidos deja de aportar (mecanismo de la convergencia prematura).
"""

import random

from genetico import calcular_aptitud, cruzar, generar_poblacion, mutar
from instancia import calcular_beneficio, calcular_costo, proyectos_seleccionados

SEMILLA = 26
LAMBDA = 5.0
TAM_TORNEO = 3


def crom(x):
    return "".join(str(g) for g in x)


def describir(etiqueta, ind):
    return (f"{etiqueta:<28} {crom(ind)}  C={calcular_costo(ind):>3}  "
            f"B={calcular_beneficio(ind):>3}  f={calcular_aptitud(ind, LAMBDA):>7.1f}  "
            f"{'factible' if calcular_costo(ind) <= 50 else 'INFACTIBLE'}")


def main():
    rng = random.Random(SEMILLA)
    poblacion = generar_poblacion(20, rng)

    print(f"POBLACION INICIAL (semilla {SEMILLA}, N=20)")
    print(f"{'#':>3}  {'cromosoma':<12}{'C':>5}{'B':>5}{'aptitud':>10}  {'validez':<11} proyectos")
    for i, ind in enumerate(poblacion):
        c, b = calcular_costo(ind), calcular_beneficio(ind)
        f = calcular_aptitud(ind, LAMBDA)
        val = "valida" if c <= 50 else "no valida"
        print(f"{i:>3}  {crom(ind):<12}{c:>5}{b:>5}{f:>10.1f}  {val:<11} "
              f"{', '.join(proyectos_seleccionados(ind))}")

    aptitudes = [calcular_aptitud(ind, LAMBDA) for ind in poblacion]
    print(f"\nAptitud promedio de la poblacion: {sum(aptitudes)/len(aptitudes):.2f}")

    print("\n" + "=" * 78)
    print("PASO 1. SELECCION POR TORNEO (tam = 3)")
    padres = []
    for n in (1, 2):
        indices = rng.sample(range(len(poblacion)), TAM_TORNEO)
        print(f"\n  Torneo {n}: participantes {indices}")
        for i in indices:
            print(f"    individuo {i:>2}  {crom(poblacion[i])}  aptitud {aptitudes[i]:>7.1f}")
        ganador = max(indices, key=lambda i: aptitudes[i])
        print(f"    -> gana el individuo {ganador} (mayor aptitud)")
        padres.append(poblacion[ganador][:])

    print("\n  " + describir("Padre 1", padres[0]))
    print("  " + describir("Padre 2", padres[1]))

    print("\n" + "=" * 78)
    print("PASO 2. CRUCE DE UN PUNTO (p_cruce = 0,80)")
    estado = rng.getstate()
    sorteo = rng.random()
    rng.setstate(estado)
    hijo_1, hijo_2, punto = cruzar(padres[0], padres[1], 0.80, rng)
    print(f"\n  Sorteo de cruce: {sorteo:.4f} < 0,80  ->  se aplica cruce"
          if punto else f"\n  Sorteo de cruce: {sorteo:.4f} >= 0,80  ->  sin cruce")
    if punto:
        print(f"  Punto de corte: k = {punto}\n")
        print(f"    P1 = {crom(padres[0])[:punto]} | {crom(padres[0])[punto:]}")
        print(f"    P2 = {crom(padres[1])[:punto]} | {crom(padres[1])[punto:]}")
        print(f"    H1 = {crom(hijo_1)[:punto]} | {crom(hijo_1)[punto:]}")
        print(f"    H2 = {crom(hijo_2)[:punto]} | {crom(hijo_2)[punto:]}")
    print("\n  " + describir("Hijo 1 antes de mutar", hijo_1))
    print("  " + describir("Hijo 2 antes de mutar", hijo_2))

    print("\n" + "=" * 78)
    print("PASO 3. MUTACION BINARIA (p_m = 0,05 por gen)")
    finales = []
    for n, hijo in enumerate((hijo_1, hijo_2), start=1):
        antes = crom(hijo)
        copia = hijo[:]
        mutar(copia, 0.05, rng)
        cambios = [i + 1 for i in range(10) if antes[i] != crom(copia)[i]]
        print(f"\n  Hijo {n}: {antes}  ->  {crom(copia)}")
        print(f"    genes mutados: {cambios if cambios else 'ninguno'}")
        finales.append(copia)

    print("\n  " + describir("Descendiente final 1", finales[0]))
    print("  " + describir("Descendiente final 2", finales[1]))

    print("\n" + "=" * 78)
    print("RESUMEN PARA EL INFORME")
    print("  " + describir("Padre 1", padres[0]))
    print("  " + describir("Padre 2", padres[1]))
    print(f"  {'Punto de cruce':<28} k = {punto}")
    print("  " + describir("Hijo 1 (antes de mutar)", hijo_1))
    print("  " + describir("Hijo 2 (antes de mutar)", hijo_2))
    print("  " + describir("Hijo 1 (final)", finales[0]))
    print("  " + describir("Hijo 2 (final)", finales[1]))


if __name__ == "__main__":
    main()
