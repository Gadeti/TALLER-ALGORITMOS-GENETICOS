# Taller: diseño e implementación de un algoritmo genético

Selección óptima de proyectos bajo restricción presupuestal (mochila 0-1,
n = 10 proyectos, presupuesto W = 50).

## Requisitos

- Python 3.9 o superior
- `matplotlib` (solo para generar las figuras)

```bash
pip install matplotlib
```

No se usa ninguna biblioteca que implemente algoritmos genéticos. El intervalo
de confianza de Wilson, la prueba exacta de McNemar y la corrección de Holm
están implementados en `src/metricas.py` y `src/experimentos.py` sin
dependencias externas.

## Ejecución

Todos los scripts se ejecutan desde `src/`.

```bash
cd src

python3 exacto.py              # verdad de terreno: enumeración, PD, voraz, lambda*
python3 baseline.py            # línea base aleatoria + autoprueba analítica
python3 verificar_genetico.py  # 9 pruebas de verificación del AG
python3 traza_manual.py        # traza completa de una iteración (punto 3)
python3 experimentos.py        # experimentos completos -> resultados/resultados.json
python3 graficas.py            # figuras -> resultados/fig*.pdf y fig*.png
```

`experimentos.py` tarda unos 45 segundos (2200 ejecuciones). `graficas.py`
requiere que `experimentos.py` se haya ejecutado antes.

Para reproducir todo de cero:

```bash
cd src && python3 exacto.py && python3 verificar_genetico.py \
  && python3 experimentos.py && python3 graficas.py
```

## Informe

```bash
cd informe && pdflatex informe.tex && pdflatex informe.tex
```

Se compila dos veces para resolver las referencias cruzadas. Requiere
`babel`, `booktabs`, `amsmath`, `geometry`, `microtype`, `hyperref` y
`caption`.

## Estructura

```
src/
  instancia.py           datos del problema; C(X), B(X), factibilidad
  exacto.py              enumeración, programación dinámica, voraz, lambda*
  metricas.py            intervalo de Wilson, resumen con censura
  baseline.py            búsqueda aleatoria con presupuesto igualado
  genetico.py            las nueve funciones exigidas + medición de diversidad
  verificar_genetico.py  pruebas de verificación
  traza_manual.py        traza paso a paso de una iteración (punto 3)
  experimentos.py        factorial, barrido de lambda, A/B/C, McNemar, Holm
  graficas.py            figuras del informe
resultados/
  resultados.json        salida completa de los experimentos
  hallazgos_lambda.txt   atractores dominantes por régimen de lambda
  fig1..fig5 .pdf/.png   figuras
informe/
  informe.tex            informe en LaTeX
  informe.pdf            informe compilado
```

## Las nueve funciones exigidas

Todas están en `src/genetico.py`. `calcular_costo` y `calcular_beneficio` se
definen en `src/instancia.py` y se reexportan desde `genetico.py` para no
duplicar la definición de los datos.

| Función | Ubicación |
|---|---|
| `generar_individuo()` | `genetico.py` |
| `generar_poblacion()` | `genetico.py` |
| `calcular_costo()` | `instancia.py`, reexportada en `genetico.py` |
| `calcular_beneficio()` | `instancia.py`, reexportada en `genetico.py` |
| `calcular_aptitud()` | `genetico.py` |
| `seleccionar_padre()` | `genetico.py` |
| `cruzar()` | `genetico.py` |
| `mutar()` | `genetico.py` |
| `ejecutar_algoritmo_genetico()` | `genetico.py` |

## Resultados principales

- **Óptimo global:** B* = 100, costo 50 exacto, `1010010001` = {P1, P3, P6, P10}.
  Único. Verificado por enumeración y por programación dinámica de forma
  independiente.
- **La heurística voraz por razón b/c alcanza ese óptimo** en diez operaciones.
- **Umbral exacto de penalización:** λ* = 1.9. Por debajo, el óptimo del paisaje
  penalizado es infactible. El λ = 5 sugerido rinde 0.31 de tasa de éxito frente
  a 0.65 con λ = 2.1.
- **El algoritmo genético no supera a la búsqueda aleatoria** con el mismo
  presupuesto de evaluaciones en ninguna de las doce configuraciones evaluadas
  (McNemar pareado con corrección de Holm, 100 semillas): empata en tres y es
  significativamente peor en ocho.

## Nota sobre reproducibilidad

Cada ejecución usa una instancia `random.Random(semilla)` propia en lugar del
estado global de `random`. Los resultados son por tanto reproducibles
exactamente, y el orden en que se ejecuten los scripts no altera ninguna salida.
