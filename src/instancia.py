"""
Instancia del problema de seleccion optima de proyectos (mochila 0-1).

Datos tomados del enunciado del taller. Este modulo es la unica fuente
de verdad de los datos: cualquier otro modulo los importa desde aqui,
nunca los redefine.
"""

# Costo c_i y beneficio b_i de cada proyecto P1..P10 (indices 0..9).
COSTOS = [12, 7, 11, 8, 9, 14, 6, 10, 5, 13]
BENEFICIOS = [24, 13, 23, 15, 16, 28, 11, 19, 9, 25]

N_PROYECTOS = len(COSTOS)      # n = 10
PRESUPUESTO = 50               # W

# Nombres para reportar soluciones de forma legible.
NOMBRES = [f"P{i + 1}" for i in range(N_PROYECTOS)]


def calcular_costo(individuo):
    """C(X) = suma de c_i * x_i."""
    return sum(COSTOS[i] * individuo[i] for i in range(N_PROYECTOS))


def calcular_beneficio(individuo):
    """B(X) = suma de b_i * x_i."""
    return sum(BENEFICIOS[i] * individuo[i] for i in range(N_PROYECTOS))


def es_factible(individuo):
    """Una solucion es factible si C(X) <= W."""
    return calcular_costo(individuo) <= PRESUPUESTO


def proyectos_seleccionados(individuo):
    """Lista de nombres de los proyectos con x_i = 1."""
    return [NOMBRES[i] for i in range(N_PROYECTOS) if individuo[i] == 1]


def mascara_a_individuo(mascara):
    """Convierte un entero de 10 bits en una lista de genes [x_1..x_10].

    Se usa solo para enumerar el espacio completo; el resto del programa
    trabaja siempre con listas de 0/1.
    """
    return [(mascara >> i) & 1 for i in range(N_PROYECTOS)]
