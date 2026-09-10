"""
Capa 4: el motor de decision.

Tres piezas, y cada una es matematica estandar de revenue management, no
heuristica inventada.

--------------------------------------------------------------------------
1. EMSR-b  ->  precio sombra (bid price)
--------------------------------------------------------------------------
Belobaba (1989), la generalizacion multiclase de la regla de Littlewood (1972).
Es el estandar de la industria para control de inventario a nivel de leg/noche.

Para clases 1..n ordenadas de tarifa mayor a menor, el nivel de proteccion y_j
de las clases 1..j frente a la clase j+1 sale de:

    P( S_j > y_j ) = p_{j+1} / p_bar_j

    S_j     = demanda agregada de las clases 1..j ~ Normal(sum mu, sqrt(sum sigma^2))
    p_bar_j = tarifa promedio ponderada por demanda de las clases 1..j

Con S_j normal:  y_j = mu_S + sigma_S * z,  z = Phi^-1(1 - p_{j+1}/p_bar_j)

Interpretacion: guardar y_j habitaciones para las clases altas solo conviene si
la probabilidad de venderlas caro supera la razon de tarifas. Con capacidad
restante x, la clase j+1 esta abierta si x > y_j, y el **bid price** es la
tarifa mas baja aun abierta. Ese es el costo de oportunidad real de la ultima
habitacion: no una formula inventada, sino el resultado de la regla.

--------------------------------------------------------------------------
2. Demanda logistica calibrada a la elasticidad supuesta
--------------------------------------------------------------------------
La disposicion a pagar se modela logistica con mediana en la tarifa vigente del
segmento (dato observado) y escala s:

    P(reserva | p) = 1 / (1 + exp((p - m) / s))

La curva se calibra con DOS condiciones que el analista elige y puede discutir:

    1. a la tarifa vigente se convierte P0 de las solicitudes que llegan
    2. la elasticidad local en ese punto vale epsilon

    epsilon(p) = -(p / s) * (1 - P(p))
      =>  s = -p_ref * (1 - P0) / epsilon
      =>  m = p_ref + s * ln(P0 / (1 - P0))

Asi el supuesto entra de forma transparente y verificable, y la curva SI tiene
optimo interior.

Nota tecnica: no se usa demanda de elasticidad constante, p^epsilon, a
proposito. Con elasticidad constante el ingreso esperado es monotono en el
precio y el optimo siempre cae en un extremo del intervalo; no es un optimo
economico sino un artefacto de la forma funcional.

--------------------------------------------------------------------------
3. Ingreso esperado ajustado por cancelacion
--------------------------------------------------------------------------
    R(p) = p * noches * P(reserva | p) * [ (1 - pc(p)) + pc(p) * retencion ]

pc(p) sale del modelo XGBoost entrenado con cancelaciones reales, **evaluado al
precio candidato** (la tarifa es una de sus variables). `retencion` vale 1.0 en
tarifas no reembolsables, donde cancelar no destruye el ingreso, y 0.0 en las
reembolsables.

Se maximiza sujeto a  p >= bid_price(x)  de EMSR-b.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

from data import CATEGORICAL_FEATURES


# ==========================================================================
# 1. EMSR-b
# ==========================================================================
def emsrb_protection_levels(
    fares: list[float], mus: list[float], sigmas: list[float]
) -> list[float]:
    """
    Niveles de proteccion EMSR-b. `fares` de mayor a menor.

    Devuelve y[j] = habitaciones protegidas para las clases 0..j frente a la
    clase j+1. Longitud: n-1.
    """
    y: list[float] = []
    for j in range(len(fares) - 1):
        mu_s = float(np.sum(mus[: j + 1]))
        sigma_s = float(np.sqrt(np.sum(np.square(sigmas[: j + 1]))))
        if mu_s <= 0:
            y.append(0.0)
            continue
        p_bar = float(np.dot(fares[: j + 1], mus[: j + 1]) / mu_s)
        ratio = fares[j + 1] / p_bar
        if ratio >= 1.0:  # la clase de abajo no es mas barata: nada que proteger
            y.append(0.0)
            continue
        z = norm.ppf(1.0 - ratio)
        y.append(max(0.0, mu_s + sigma_s * z))
    return y


def open_classes(remaining: float, protections: list[float]) -> list[int]:
    """Clases abiertas con `remaining` habitaciones libres. La 0 siempre abre."""
    openx = [0]
    for k in range(1, len(protections) + 1):
        if remaining > protections[k - 1]:
            openx.append(k)
    return openx


def bid_price(remaining: float, fares: list[float], protections: list[float]) -> float:
    """
    Precio sombra: la tarifa mas baja que el control de inventario aun acepta.
    Es el piso para cualquier tarifa que se cotice.
    """
    return min(fares[k] for k in open_classes(remaining, protections))


# ==========================================================================
# 2. Demanda
# ==========================================================================
def logistic_params(
    reference_price: float, elasticity: float, base_conversion: float = 0.5
) -> tuple[float, float]:
    """
    Parametros (m, s) de la disposicion a pagar logistica, calibrados con dos
    condiciones que el usuario elige explicitamente:

      1. A la tarifa vigente se convierte `base_conversion` de las solicitudes.
      2. La elasticidad local en ese punto es `elasticity`.

    De P(p) = 1 / (1 + exp((p - m) / s)):

        epsilon(p) = -(p / s) * (1 - P(p))
        =>  s = -reference_price * (1 - P0) / epsilon
        =>  m = reference_price + s * ln(P0 / (1 - P0))
    """
    p0 = float(np.clip(base_conversion, 0.01, 0.99))
    s = float(-reference_price * (1.0 - p0) / elasticity)
    m = float(reference_price + s * np.log(p0 / (1.0 - p0)))
    return m, s


def prob_booking(
    price: np.ndarray | float,
    reference_price: float,
    elasticity: float,
    base_conversion: float = 0.5,
):
    """P(reserva | precio) con disposicion a pagar logistica."""
    m, s = logistic_params(reference_price, elasticity, base_conversion)
    return 1.0 / (1.0 + np.exp((np.asarray(price, dtype=float) - m) / s))


# ==========================================================================
# 3. Sobreventa
# ==========================================================================
def overbooking_limit(
    capacity: int, cancel_rate: float, tolerance: float = 0.05, max_factor: float = 1.5
) -> int:
    """
    Mayor autorizacion A tal que P(presentados > capacidad) <= tolerancia,
    con presentados ~ Binomial(A, 1 - tasa_cancelacion).

    Es el calculo estandar de sobreventa por riesgo. Con 28% de cancelaciones y
    5% de tolerancia, un hotel de 230 habitaciones puede autorizar bastante mas
    de 230 sin que la probabilidad de dejar gente fuera pase del 5%.
    """
    show_rate = max(1e-6, 1.0 - cancel_rate)
    best = capacity
    for a in range(capacity, int(capacity * max_factor) + 1):
        if binom.sf(capacity, a, show_rate) <= tolerance:
            best = a
        else:
            break
    return best


# ==========================================================================
# 4. Optimizacion de tarifa
# ==========================================================================
@dataclass
class Booking:
    """Una solicitud de reserva concreta."""

    hotel: str
    market_segment: str
    lead_time: int
    total_nights: int
    stays_in_weekend_nights: int
    adults: int
    children: int = 0
    babies: int = 0
    is_repeated_guest: int = 0
    previous_cancellations: int = 0
    previous_bookings_not_canceled: int = 0
    booking_changes: int = 0
    days_in_waiting_list: int = 0
    required_car_parking_spaces: int = 0
    total_of_special_requests: int = 0
    deposit_type: str = "No Deposit"
    customer_type: str = "Transient"
    distribution_channel: str = "TA/TO"
    meal: str = "BB"
    reserved_room_type: str = "A"
    country_grouped: str = "PRT"
    arrival_month: int = 7
    arrival_week: int = 28
    arrival_dow: int = 5
    arrival_year: int = 2017
    has_agent: int = 1
    has_company: int = 0


@dataclass
class Result:
    price_grid: np.ndarray
    revenue: np.ndarray
    p_booking: np.ndarray
    p_cancel: np.ndarray
    optimal_price: float
    optimal_revenue: float
    bid_price: float
    reference_price: float
    open_classes: list[int] = field(default_factory=list)


def _feature_row(b: Booking, price: float, segment_median: float) -> dict:
    return {
        "lead_time": b.lead_time,
        "stays_in_weekend_nights": b.stays_in_weekend_nights,
        "stays_in_week_nights": max(0, b.total_nights - b.stays_in_weekend_nights),
        "total_nights": b.total_nights,
        "adults": b.adults,
        "children": b.children,
        "babies": b.babies,
        "total_guests": b.adults + b.children + b.babies,
        "previous_cancellations": b.previous_cancellations,
        "previous_bookings_not_canceled": b.previous_bookings_not_canceled,
        "booking_changes": b.booking_changes,
        "days_in_waiting_list": b.days_in_waiting_list,
        "required_car_parking_spaces": b.required_car_parking_spaces,
        "total_of_special_requests": b.total_of_special_requests,
        "adr": price,
        "adr_ratio_segment": price / segment_median,
        "is_repeated_guest": b.is_repeated_guest,
        "has_agent": b.has_agent,
        "has_company": b.has_company,
        "arrival_month": b.arrival_month,
        "arrival_week": b.arrival_week,
        "arrival_dow": b.arrival_dow,
        "arrival_year": b.arrival_year,
        "hotel": b.hotel,
        "market_segment": b.market_segment,
        "distribution_channel": b.distribution_channel,
        "deposit_type": b.deposit_type,
        "customer_type": b.customer_type,
        "meal": b.meal,
        "reserved_room_type": b.reserved_room_type,
        "country_grouped": b.country_grouped,
    }


def cancel_probabilities(
    bundle: dict, booking: Booking, prices: np.ndarray, segment_median: float
) -> np.ndarray:
    """
    P(cancelacion) evaluada en CADA precio candidato.

    Esta es la diferencia de fondo con la version original del proyecto: ahi el
    modelo se llamaba siempre con una tarifa fija de $150, asi que su salida era
    constante respecto al precio y se cancelaba al derivar. El modelo no influia
    en nada.
    """
    model, columns = bundle["model"], bundle["columns"]
    rows = [_feature_row(booking, float(p), segment_median) for p in prices]
    X = pd.DataFrame(rows)[columns]
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return model.predict_proba(X)[:, 1]


def optimize(
    bundle: dict,
    booking: Booking,
    reference_price: float,
    elasticity: float,
    remaining_capacity: float,
    fares: list[float],
    mus: list[float],
    sigmas: list[float],
    segment_median: float | None = None,
    base_conversion: float = 0.5,
    price_span: tuple[float, float] = (0.4, 2.5),
    n_grid: int = 220,
) -> Result:
    """
    Tarifa que maximiza el ingreso neto esperado, con el bid price de EMSR-b
    como piso.
    """
    segment_median = segment_median or reference_price
    protections = emsrb_protection_levels(fares, mus, sigmas)
    floor = bid_price(remaining_capacity, fares, protections)

    lo = max(floor, reference_price * price_span[0])
    hi = max(lo * 1.05, reference_price * price_span[1])
    grid = np.linspace(lo, hi, n_grid)

    p_book = np.asarray(
        prob_booking(grid, reference_price, elasticity, base_conversion), dtype=float
    )
    p_cancel = cancel_probabilities(bundle, booking, grid, segment_median)

    # Una tarifa no reembolsable conserva el ingreso aunque el huesped cancele.
    retention = 1.0 if booking.deposit_type == "Non Refund" else 0.0
    realized = (1.0 - p_cancel) + p_cancel * retention

    revenue = grid * booking.total_nights * p_book * realized
    k = int(np.argmax(revenue))

    return Result(
        price_grid=grid,
        revenue=revenue,
        p_booking=p_book,
        p_cancel=p_cancel,
        optimal_price=float(grid[k]),
        optimal_revenue=float(revenue[k]),
        bid_price=float(floor),
        reference_price=float(reference_price),
        open_classes=open_classes(remaining_capacity, protections),
    )


def sensitivity(
    bundle: dict,
    booking: Booking,
    reference_price: float,
    remaining_capacity: float,
    fares: list[float],
    mus: list[float],
    sigmas: list[float],
    elasticities: list[float],
    segment_median: float | None = None,
    base_conversion: float = 0.5,
) -> pd.DataFrame:
    """
    Como se mueve la recomendacion con el supuesto de elasticidad.

    La elasticidad no se pudo estimar con estos datos (ver
    `estimate_elasticity.py`), asi que mostrar este cuadro no es un extra: es
    parte de reportar el resultado con honestidad.
    """
    rows = []
    for eps in elasticities:
        r = optimize(
            bundle, booking, reference_price, eps, remaining_capacity,
            fares, mus, sigmas, segment_median, base_conversion,
        )
        rows.append(
            {
                "elasticidad": eps,
                "adr_optimo": round(r.optimal_price, 2),
                "vs_tarifa_vigente": f"{(r.optimal_price / reference_price - 1) * 100:+.1f}%",
                "p_reserva": round(float(np.interp(r.optimal_price, r.price_grid, r.p_booking)), 4),
                "p_cancelacion": round(float(np.interp(r.optimal_price, r.price_grid, r.p_cancel)), 4),
                "ingreso_esperado": round(r.optimal_revenue, 2),
            }
        )
    return pd.DataFrame(rows)
