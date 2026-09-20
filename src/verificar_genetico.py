"""Pruebas de verificacion del modulo genetico (decision: verificar cada parte)."""
from exacto import optimo_por_enumeracion
from genetico import CONFIG_ENUNCIADO, ejecutar_algoritmo_genetico

B_OPT = optimo_por_enumeracion()[0]
fallos = []

def check(nombre, condicion, detalle=""):
    print(f"  [{'OK ' if condicion else 'FALLA'}] {nombre} {detalle}")
    if not condicion:
        fallos.append(nombre)

print("VERIFICACION DEL MODULO GENETICO\n")

# 1. Reproducibilidad
r1 = ejecutar_algoritmo_genetico(**CONFIG_ENUNCIADO, semilla=7, beneficio_optimo=B_OPT)
r2 = ejecutar_algoritmo_genetico(**CONFIG_ENUNCIADO, semilla=7, beneficio_optimo=B_OPT)
r3 = ejecutar_algoritmo_genetico(**CONFIG_ENUNCIADO, semilla=8, beneficio_optimo=B_OPT)
check("misma semilla -> mismo resultado",
      r1["mejor_por_aptitud"] == r2["mejor_por_aptitud"])
check("semillas distintas -> trayectorias distintas",
      r1["historial"] != r3["historial"])

# 2. Presupuesto de evaluaciones exacto
for N, G in ((10, 200), (20, 100), (50, 40)):
    r = ejecutar_algoritmo_genetico(n_poblacion=N, n_generaciones=G, semilla=0)
    check(f"evaluaciones N={N} G={G}", r["evaluaciones"] == N * G,
          f"= {r['evaluaciones']} (esperado {N*G})")

# 3. Elitismo: la mejor aptitud nunca decrece
hist = r1["historial"]
monotona = all(hist[i+1]["mejor_aptitud"] >= hist[i]["mejor_aptitud"] for i in range(len(hist)-1))
check("elitismo -> mejor aptitud monotona no decreciente", monotona)

# 4. El AG nunca supera el optimo real (deteccion de bug en la aptitud)
peor = []
for s in range(50):
    r = ejecutar_algoritmo_genetico(**CONFIG_ENUNCIADO, semilla=s, beneficio_optimo=B_OPT)
    mf = r["mejor_factible"]
    peor.append(mf["beneficio"])
    if mf["costo"] > 50: fallos.append("factible viola presupuesto")
check("ningun mejor factible supera B*=100", max(peor) <= B_OPT, f"max={max(peor)}")
check("todos los 'mejor factible' respetan C<=50", "factible viola presupuesto" not in fallos)

# 5. Coherencia de la generacion del primer acierto
r = ejecutar_algoritmo_genetico(**CONFIG_ENUNCIADO, semilla=1, beneficio_optimo=B_OPT)
g = r["generacion_primer_optimo"]
check("generacion del primer acierto coherente con el historial",
      g is None or r["historial"][g]["mejor_beneficio"] <= B_OPT, f"g={g}")

# 6. Comportamiento con la configuracion del enunciado (50 semillas)
exitos = sum(1 for b in peor if b == B_OPT)
print(f"\n  Config del enunciado, 50 semillas: tasa de exito = {exitos/50:.1%}")
print(f"  Beneficio minimo obtenido en 50 corridas = {min(peor)}")

# 7. lambda bajo -> el 'mejor por aptitud' se vuelve infactible
for lam in (0.0, 1.0, 1.9, 2.1, 5.0):
    cfg = dict(CONFIG_ENUNCIADO); cfg["lambda_penalizacion"] = lam
    infact = sum(1 for s in range(50)
                 if not ejecutar_algoritmo_genetico(**cfg, semilla=s)["mejor_por_aptitud"]["factible"])
    print(f"  lambda={lam:>4}: devuelve solucion infactible en {infact:>2}/50 corridas")

print("\n" + ("TODAS LAS PRUEBAS PASAN" if not fallos else f"FALLOS: {set(fallos)}"))
