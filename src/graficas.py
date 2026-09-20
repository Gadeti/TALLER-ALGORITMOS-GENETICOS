"""
Figuras del informe.

Decisiones de visualizacion:
- Las curvas de aptitud se agregan sobre las 100 semillas (mediana y rango
  intercuartilico) en lugar de mostrar una corrida suelta: una trayectoria
  aislada de un algoritmo estocastico es una anecdota, no evidencia.
- Nunca se usa doble eje vertical. Cuando dos magnitudes tienen escalas
  distintas (individuos distintos y distancia de Hamming) se usan paneles
  separados que comparten el eje de generaciones.
- La identidad de cada serie se codifica con color Y con marcador distinto,
  de modo que las figuras siguen siendo legibles impresas en escala de grises.
- Formato PDF vectorial para incrustar en LaTeX sin perdida de nitidez.
"""

import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exacto import optimo_por_enumeracion
from genetico import ejecutar_algoritmo_genetico
from metricas import intervalo_wilson

AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_SUAVE = "#0b0b0b", "#52514e"
DIR = os.path.join(os.path.dirname(__file__), "..", "resultados")
B_OPT = optimo_por_enumeracion()[0]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.edgecolor": "#b9b8b2",
    "axes.labelcolor": TINTA,
    "axes.titlesize": 10,
    "text.color": TINTA,
    "xtick.color": TINTA_SUAVE,
    "ytick.color": TINTA_SUAVE,
    "grid.color": "#e6e5e0",
    "figure.dpi": 140,
})


def _limpiar(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def _guardar(fig, nombre):
    os.makedirs(DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(DIR, f"{nombre}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {nombre}.pdf / .png")


def _corridas_con_historial(n_pob=20, n_gen=100, p_mut=0.05, lam=5.0, semillas=100):
    return [
        ejecutar_algoritmo_genetico(
            n_poblacion=n_pob, n_generaciones=n_gen, p_cruce=0.80, p_mutacion=p_mut,
            tam_torneo=3, n_elite=1, lambda_penalizacion=lam,
            semilla=s, beneficio_optimo=B_OPT, registrar_historial=True,
        )
        for s in range(semillas)
    ]


def _percentiles(series_por_generacion, q):
    """Cuantil q de cada generacion, sin numpy."""
    salida = []
    for valores in series_por_generacion:
        v = sorted(valores)
        pos = q * (len(v) - 1)
        bajo, alto = int(pos), min(int(pos) + 1, len(v) - 1)
        salida.append(v[bajo] + (v[alto] - v[bajo]) * (pos - bajo))
    return salida


def _por_generacion(corridas, clave):
    n_gen = len(corridas[0]["historial"])
    return [[r["historial"][g][clave] for r in corridas] for g in range(n_gen)]


# --------------------------------------------------------------------------

def figura_aptitud(corridas):
    """Mejor aptitud y aptitud promedio por generacion (exigida por el enunciado)."""
    mejor = _por_generacion(corridas, "mejor_aptitud")
    promedio = _por_generacion(corridas, "aptitud_promedio")
    gens = range(len(mejor))

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for datos, color, marcador, etiqueta in (
        (mejor, AZUL, "o", "Mejor aptitud"),
        (promedio, NARANJA, "s", "Aptitud promedio"),
    ):
        ax.fill_between(gens, _percentiles(datos, 0.25), _percentiles(datos, 0.75),
                        color=color, alpha=0.18, linewidth=0)
        med = _percentiles(datos, 0.5)
        ax.plot(gens, med, color=color, linewidth=2, label=etiqueta)
        paso = max(1, len(med) // 8)
        ax.plot(list(gens)[::paso], med[::paso], color=color, marker=marcador,
                markersize=5, linestyle="none", markeredgecolor="white",
                markeredgewidth=1.2)

    ax.axhline(B_OPT, color=TINTA_SUAVE, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(f"$B^* = {B_OPT}$", xy=(len(mejor) * 0.62, B_OPT + 1.5),
                color=TINTA_SUAVE, fontsize=8.5)
    ax.set_xlabel("Generación")
    ax.set_ylabel("Aptitud")
    ax.set_title("Evolución de la aptitud (mediana y rango intercuartílico, 100 semillas)")
    ax.legend(frameon=False, loc="lower right")
    _limpiar(ax)
    _guardar(fig, "fig1_aptitud")


def figura_diversidad(corridas):
    """Colapso de la diversidad: dos paneles, nunca doble eje."""
    distintos = _por_generacion(corridas, "individuos_distintos")
    hamming = _por_generacion(corridas, "hamming_media")
    gens = range(len(distintos))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.4, 4.4), sharex=True)
    for ax, datos, color, ylab, titulo in (
        (a1, distintos, AZUL, "Individuos distintos",
         "Diversidad de la población (población de 20)"),
        (a2, hamming, AQUA, "Distancia de Hamming media", None),
    ):
        ax.fill_between(gens, _percentiles(datos, 0.25), _percentiles(datos, 0.75),
                        color=color, alpha=0.18, linewidth=0)
        ax.plot(gens, _percentiles(datos, 0.5), color=color, linewidth=2)
        ax.set_ylabel(ylab)
        if titulo:
            ax.set_title(titulo)
        _limpiar(ax)

    a1.annotate("colapso antes de la generación 20", xy=(20, 9), xytext=(38, 15),
                fontsize=8.5, color=TINTA_SUAVE,
                arrowprops=dict(arrowstyle="->", color=TINTA_SUAVE, linewidth=0.9))
    a2.set_xlabel("Generación")
    _guardar(fig, "fig2_diversidad")


def figura_lambda(datos):
    """Tasa de exito frente a la penalizacion, con IC de Wilson."""
    celdas = datos["barrido_lambda"]
    lams = [c["config"]["lambda_penalizacion"] for c in celdas]
    tasas = [c["tasa_exito"] for c in celdas]
    bajo = [t - c["ic_wilson"][0] for t, c in zip(tasas, celdas)]
    alto = [c["ic_wilson"][1] - t for t, c in zip(tasas, celdas)]
    x = range(len(lams))

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.errorbar(x, tasas, yerr=[bajo, alto], color=AZUL, linewidth=2,
                marker="o", markersize=6, markeredgecolor="white",
                markeredgewidth=1.2, capsize=3, ecolor="#9fb9dd")

    i_critico = lams.index(1.9)
    ax.axvline(i_critico, color=NARANJA, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(r"$\lambda^* = 1{,}9$", xy=(i_critico + 0.12, 0.08),
                color=NARANJA, fontsize=9)
    i_enunciado = lams.index(5.0)
    ax.annotate("valor del enunciado", xy=(i_enunciado, tasas[i_enunciado] - 0.03),
                xytext=(i_enunciado + 0.35, 0.62), fontsize=8.5, color=TINTA_SUAVE,
                ha="left",
                arrowprops=dict(arrowstyle="->", color=TINTA_SUAVE, linewidth=0.9,
                                connectionstyle="arc3,rad=0.2"))

    for xi, t in zip(x, tasas):
        ax.annotate(f"{t:.2f}", xy=(xi, t), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=TINTA_SUAVE,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              edgecolor="none", alpha=0.85))

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{l:g}" for l in lams])
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel(r"Coeficiente de penalización $\lambda$")
    ax.set_ylabel("Tasa de éxito")
    ax.set_title("Efecto de la penalización (N=20, G=100, 100 semillas)")
    _limpiar(ax)
    _guardar(fig, "fig3_lambda")


def figura_factorial(datos):
    """Interaccion poblacion x mutacion a presupuesto igualado."""
    celdas = {c["etiqueta"]: c for c in datos["factorial"]}
    mutaciones = [0.01, 0.05, 0.10]
    x = range(len(mutaciones))

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for n_pob, color, marcador in ((10, AZUL, "o"), (20, NARANJA, "s"), (50, AQUA, "^")):
        tasas, bajo, alto = [], [], []
        for pm in mutaciones:
            c = celdas[f"N={n_pob}, pm={pm:g}"]
            tasas.append(c["tasa_exito"])
            bajo.append(c["tasa_exito"] - c["ic_wilson"][0])
            alto.append(c["ic_wilson"][1] - c["tasa_exito"])
        ax.errorbar(x, tasas, yerr=[bajo, alto], color=color, linewidth=2,
                    marker=marcador, markersize=6, markeredgecolor="white",
                    markeredgewidth=1.2, capsize=3, ecolor=color, elinewidth=0.9,
                    alpha=0.95, label=f"N = {n_pob}")
        ax.annotate(f"N={n_pob}", xy=(2, tasas[-1]), xytext=(6, -2),
                    textcoords="offset points", fontsize=8.5, color=color)

    aleatoria = next(b for b in datos["baselines"] if b["evaluaciones"] == 2000)
    ax.axhline(aleatoria["tasa_exito"], color=TINTA_SUAVE, linewidth=1.2,
               linestyle=(0, (4, 3)))
    ax.annotate(f"búsqueda aleatoria ({aleatoria['tasa_exito']:.2f})",
                xy=(0.02, aleatoria["tasa_exito"] + 0.03), fontsize=8.5,
                color=TINTA_SUAVE)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{m:g}" for m in mutaciones])
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel(r"Probabilidad de mutación $p_m$")
    ax.set_ylabel("Tasa de éxito")
    ax.set_title("Factorial a presupuesto igualado (2000 evaluaciones)")
    ax.legend(frameon=False, loc="upper left")
    _limpiar(ax)
    _guardar(fig, "fig4_factorial")


def figura_comparacion(datos):
    """AG frente a busqueda aleatoria al mismo presupuesto de evaluaciones."""
    celdas = datos["factorial"] + datos["enunciado"]
    celdas = sorted(celdas, key=lambda c: c["tasa_exito"])
    bases = {b["evaluaciones"]: b["tasa_exito"] for b in datos["baselines"]}

    etiquetas = [c["etiqueta"].replace(" (enunciado)", "*") for c in celdas]
    y = range(len(celdas))
    ancho = 0.38

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.barh([i + ancho / 2 for i in y], [c["tasa_exito"] for c in celdas],
            height=ancho, color=AZUL, label="Algoritmo genético")
    ax.barh([i - ancho / 2 for i in y], [bases[c["evaluaciones"]] for c in celdas],
            height=ancho, color=NARANJA, label="Búsqueda aleatoria")

    for i, c in enumerate(celdas):
        ax.annotate(f"{c['tasa_exito']:.2f}", xy=(c["tasa_exito"], i + ancho / 2),
                    xytext=(4, -2.5), textcoords="offset points", fontsize=7.5,
                    color=TINTA_SUAVE)
        ax.annotate(f"{bases[c['evaluaciones']]:.2f}",
                    xy=(bases[c["evaluaciones"]], i - ancho / 2),
                    xytext=(4, -2.5), textcoords="offset points", fontsize=7.5,
                    color=TINTA_SUAVE)

    ax.set_yticks(list(y))
    ax.set_yticklabels(etiquetas, fontsize=8)
    ax.set_xlim(0, 1.16)
    ax.set_xlabel("Tasa de éxito (mismo presupuesto de evaluaciones)")
    ax.set_title("El AG nunca supera a la búsqueda aleatoria\n"
                 "(* configuraciones del enunciado)", fontsize=9.5)
    ax.legend(frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.10))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    _guardar(fig, "fig5_comparacion")


if __name__ == "__main__":
    with open(os.path.join(DIR, "resultados.json")) as f:
        datos = json.load(f)
    print("Generando figuras:")
    corridas = _corridas_con_historial()
    figura_aptitud(corridas)
    figura_diversidad(corridas)
    figura_lambda(datos)
    figura_factorial(datos)
    figura_comparacion(datos)
