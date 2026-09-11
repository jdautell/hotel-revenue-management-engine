"""
Capa 5: revenue management de red (multinoche).

EL PROBLEMA QUE RESUELVE
------------------------
EMSR-b controla UNA noche. Pero una reserva de tres noches no consume una
habitacion: consume una habitacion en cada una de tres fechas, y esas fechas
tienen demanda distinta.

Ejemplo concreto de lo que EMSR-b por noche no puede ver: llega una solicitud de
jueves a sabado. El jueves esta vacio y el sabado casi lleno. Si se cotiza con
el piso del jueves, se regala la habitacion del sabado. Si se cotiza con el del
sabado, se rechaza demanda buena para el jueves. El piso correcto no es el de
ninguna noche: es la SUMA de los costos de oportunidad de las tres.

LA FORMULACION
--------------
Programa lineal deterministico (DLP), la formulacion canonica de network revenue
management (Talluri & van Ryzin, "The Theory and Practice of Revenue
Management", cap. 3):

    max  sum_j  r_j * x_j
    s.a. sum_j  a_ij * x_j <= c_i     para cada noche i
         0 <= x_j <= E[D_j]

    j        = producto (fecha de llegada, duracion, clase tarifaria)
    r_j      = ingreso del producto = tarifa_clase * duracion
    a_ij     = 1 si el producto j ocupa la noche i
    c_i      = habitaciones disponibles la noche i
    E[D_j]   = demanda esperada del producto j

Los **precios sombra** (variables duales de las restricciones de capacidad) son
los bid prices por fecha. El dual de la noche i responde exactamente la pregunta
del revenue manager: cuanto ingreso adicional generaria una habitacion mas esa
noche.

LA REGLA DE CONTROL
-------------------
Una estancia que llega el dia a y dura L noches se acepta si

    ingreso total  >=  suma de los bid prices de las noches a .. a+L-1

que en tarifa por noche es

    ADR minimo  =  (suma de bid prices) / L

Esto es *additive bid price control*, el heuristico estandar de red. Dos limitaciones conocidas, y las dos se manejan explicitamente:

  - El DLP es **deterministico**: sustituye la demanda por su media e ignora la
    variabilidad. Bajo capacidad muy apretada eso produce duales extremos, mas
    altos que cualquier tarifa individual. No es un error de calculo -una
    habitacion mas esa noche si vale eso, porque habilita estancias largas
    completas- pero si es una senal de que el modelo esta operando en un
    regimen donde conviene revisarlo a mano.
  - El dual **vale cero** cuando la restriccion no esta activa. Correcto como
    medida de desplazamiento, inservible como piso de tarifa por si solo. Por
    eso existe `combined_floor`, que lo combina con EMSR-b.

El remedio estandar para ambas es volver a resolver conforme entra demanda
(*re-solving*): `solve` se recalcula con la capacidad restante del momento.

RELACION CON EMSR-b
-------------------
No se reemplaza: se complementan. EMSR-b sigue dando el control por clase dentro
de una noche; el DLP da el costo de oportunidad correcto cuando la estancia
cruza varias fechas. Cuando la estancia es de una sola noche, ambos deberian
dar ordenes de magnitud parecidos, y `compare_single_vs_network` lo verifica.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class NetworkSolution:
    bid_prices: np.ndarray          # costo de oportunidad por noche
    revenue: float                  # ingreso optimo del plan
    allocation: np.ndarray          # x_j: cuanto vender de cada producto
    products: list[tuple[int, int, int]]  # (llegada, duracion, clase)
    nights: int


def build_products(
    horizon: int,
    fares: list[float],
    demand_by_date: list[list[float]],
    los_probs: dict[int, float],
) -> tuple[list[tuple[int, int, int]], np.ndarray, np.ndarray]:
    """
    Productos del problema de red.

    demand_by_date[d][k] = demanda esperada de la clase k con llegada el dia d.
    los_probs[L]         = probabilidad de que la estancia dure L noches.

    La demanda de un producto se reparte proporcionalmente: se verifico en los
    datos que la distribucion de duracion es practicamente igual entre clases
    (duracion media 2.89 a 2.95 noches en las cuatro), asi que repartir por una
    sola distribucion no introduce sesgo apreciable.
    """
    products: list[tuple[int, int, int]] = []
    revenues: list[float] = []
    demands: list[float] = []

    for d in range(horizon):
        for k, fare in enumerate(fares):
            for los, p_los in los_probs.items():
                if p_los <= 0:
                    continue
                products.append((d, los, k))
                revenues.append(fare * los)
                demands.append(demand_by_date[d][k] * p_los)

    return products, np.array(revenues), np.array(demands)


def solve(
    horizon: int,
    fares: list[float],
    demand_by_date: list[list[float]],
    los_probs: dict[int, float],
    capacity_by_night: list[float],
) -> NetworkSolution:
    """
    Resuelve el DLP y devuelve los bid prices por noche.

    `capacity_by_night` debe cubrir el horizonte mas la estancia mas larga: una
    llegada el ultimo dia del horizonte todavia ocupa noches posteriores.
    """
    max_los = max(los_probs)
    n_nights = horizon + max_los - 1
    if len(capacity_by_night) < n_nights:
        raise ValueError(
            f"se necesitan {n_nights} noches de capacidad, llegaron {len(capacity_by_night)}"
        )

    products, revenues, demands = build_products(horizon, fares, demand_by_date, los_probs)

    # A[i, j] = 1 si el producto j ocupa la noche i
    A = np.zeros((n_nights, len(products)))
    for j, (arrival, los, _) in enumerate(products):
        A[arrival : arrival + los, j] = 1.0

    res = linprog(
        c=-revenues,                       # linprog minimiza
        A_ub=A,
        b_ub=np.asarray(capacity_by_night[:n_nights], dtype=float),
        bounds=[(0.0, float(d)) for d in demands],
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"el DLP no convergio: {res.message}")

    # Los duales vienen negativos por la convencion de minimizacion de HiGHS.
    bid = np.maximum(0.0, -np.asarray(res.ineqlin.marginals, dtype=float))

    return NetworkSolution(
        bid_prices=bid,
        revenue=float(-res.fun),
        allocation=np.asarray(res.x, dtype=float),
        products=products,
        nights=n_nights,
    )


def stay_floor(solution: NetworkSolution, arrival: int, los: int) -> float:
    """Costo de desplazamiento total de una estancia: suma de sus bid prices."""
    return float(np.sum(solution.bid_prices[arrival : arrival + los]))


def min_acceptable_adr(solution: NetworkSolution, arrival: int, los: int) -> float:
    """Tarifa por noche minima que cubre el desplazamiento que la estancia causa."""
    return stay_floor(solution, arrival, los) / max(1, los)


def combined_floor(
    solution: NetworkSolution, arrival: int, los: int, emsrb_floor: float
) -> float:
    """
    El piso que se usa en la practica: el mayor de los dos controles.

    Por que hacen falta los dos:

    - El **dual del DLP vale cero** cuando la restriccion de capacidad de esa
      noche no esta activa. Eso es correcto (una habitacion mas no genera
      ingreso adicional si sobran) pero como piso de tarifa es inservible:
      diria que se puede vender a cero. El DLP mide DESPLAZAMIENTO, no valor.
    - EMSR-b sigue protegiendo la mezcla de clases dentro de cada noche, que es
      justo lo que el DLP deterministico no ve porque ignora la variabilidad.

    Se toma el maximo: la tarifa tiene que cubrir a la vez el desplazamiento que
    causa en la red y el control de clases de la noche de llegada.
    """
    return max(emsrb_floor, min_acceptable_adr(solution, arrival, los))


def compare_single_vs_network(
    solution: NetworkSolution, arrival: int, los: int, single_night_floor: float
) -> dict:
    """
    Contrasta el piso por noche (EMSR-b sobre una sola fecha) contra el de red.

    La diferencia es el error que se comete al cotizar una estancia multinoche
    mirando una sola de sus fechas.
    """
    net = min_acceptable_adr(solution, arrival, los)
    return {
        "noches": los,
        "piso_una_noche": round(single_night_floor, 2),
        "piso_red_por_noche": round(net, 2),
        "diferencia_pct": round((net / single_night_floor - 1) * 100, 1)
        if single_night_floor > 0
        else float("inf"),
        "bid_prices_de_la_estancia": [
            round(float(v), 2) for v in solution.bid_prices[arrival : arrival + los]
        ],
    }
