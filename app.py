"""
Accor Revenue Optimizer — dashboard.

Cada numero que se muestra dice de donde sale: dato observado, modelo entrenado
o supuesto del analista. Esa separacion es el punto del proyecto.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from optimizer import (  # noqa: E402
    Booking,
    bid_price,
    emsrb_protection_levels,
    logistic_params,
    open_classes,
    optimize,
    overbooking_limit,
    sensitivity,
)

st.set_page_config(page_title="Accor Revenue Optimizer", layout="wide",
                   initial_sidebar_state="expanded")

SEASON_LABEL = {"alta": "Temporada alta", "media": "Temporada media", "baja": "Temporada baja"}
SEGMENTS = ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups"]
CHANNEL_BY_SEGMENT = {
    "Online TA": "TA/TO", "Offline TA/TO": "TA/TO", "Direct": "Direct",
    "Corporate": "Corporate", "Groups": "TA/TO",
}


@st.cache_resource
def load_artifacts():
    bundle = joblib.load(ROOT / "models" / "cancellation_model.joblib")
    calib = json.loads((ROOT / "models" / "demand_calibration.json").read_text(encoding="utf-8"))
    metrics = json.loads((ROOT / "reports" / "cancellation_metrics.json").read_text(encoding="utf-8"))
    elasticity = json.loads((ROOT / "reports" / "elasticity_analysis.json").read_text(encoding="utf-8"))
    return bundle, calib, metrics, elasticity


bundle, CAL, METRICS, ELAST = load_artifacts()

st.title("Motor de Revenue Management")
st.caption(
    "Control de inventario con EMSR-b · ingreso ajustado por riesgo de cancelación · "
    "curva de demanda con supuesto explícito"
)

# ==========================================================================
# Sidebar
# ==========================================================================
sb = st.sidebar
sb.header("Inventario")
hotel = sb.selectbox("Hotel", list(CAL["hoteles"].keys()))
H = CAL["hoteles"][hotel]
capacity = H["capacidad_estimada"]

month = sb.slider("Mes de llegada", 1, 12, 7)
season = H["temporada_por_mes"][str(month)]
is_weekend = 1 if sb.radio("Día de llegada", ["Entre semana", "Viernes o sábado"],
                           index=1, horizontal=True) == "Viernes o sábado" else 0
remaining = sb.slider("Habitaciones libres para esa noche", 1, capacity, min(60, capacity))

sb.markdown("---")
sb.header("Solicitud de reserva")
segment = sb.selectbox("Segmento de mercado", SEGMENTS)
lead_time = sb.slider("Anticipación (días)", 0, 365, 45)
nights = sb.number_input("Noches", 1, 21, 3)
weekend_nights = sb.number_input("De ellas, en fin de semana", 0, int(nights),
                                 min(2, int(nights)))
adults = sb.number_input("Adultos", 1, 6, 2)
children = sb.number_input("Menores", 0, 6, 0)
deposit = sb.selectbox("Política de tarifa", ["Reembolsable", "No reembolsable"])

sb.markdown("---")
sb.header("Perfil del huésped")
repeated = sb.checkbox("Huésped frecuente")
prev_cancels = sb.number_input("Cancelaciones previas", 0, 10, 0)
special_req = sb.slider("Peticiones especiales", 0, 5, 1)
parking = sb.checkbox("Solicita estacionamiento")

sb.markdown("---")
sb.header("Supuestos del analista")
sb.caption("Estos dos NO se estimaron con los datos. Ver la pestaña de metodología.")
elasticity = sb.slider("Elasticidad precio-demanda", -2.5, -0.3, -1.2, 0.1)
base_conv = sb.slider("Conversión a la tarifa vigente", 0.20, 0.95, 0.70, 0.05)

# ==========================================================================
# Parametros derivados de datos observados
# ==========================================================================
fares = [c["tarifa"] for c in H["clases"]]
dem = H["demanda_por_clase"].get(season, {}).get(str(is_weekend), {})
mus = [dem.get(str(i), {}).get("mu", 1.0) for i in range(len(fares))]
sigmas = [dem.get(str(i), {}).get("sigma", 1.0) for i in range(len(fares))]

medians = bundle["medians"]
row = medians[
    (medians["hotel"] == hotel)
    & (medians["market_segment"] == segment)
    & (medians["arrival_month"] == month)
]
reference_price = float(row["adr_segment_median"].iloc[0]) if len(row) else float(np.median(fares))

booking = Booking(
    hotel=hotel,
    market_segment=segment,
    lead_time=int(lead_time),
    total_nights=int(nights),
    stays_in_weekend_nights=int(weekend_nights),
    adults=int(adults),
    children=int(children),
    is_repeated_guest=int(repeated),
    previous_cancellations=int(prev_cancels),
    previous_bookings_not_canceled=3 if repeated else 0,
    required_car_parking_spaces=int(parking),
    total_of_special_requests=int(special_req),
    deposit_type="Non Refund" if deposit == "No reembolsable" else "No Deposit",
    distribution_channel=CHANNEL_BY_SEGMENT[segment],
    customer_type="Transient",
    arrival_month=int(month),
    arrival_week=int(min(52, max(1, round(month * 4.33)))),
    arrival_dow=5 if is_weekend else 2,
    arrival_year=2017,
)

res = optimize(
    bundle, booking, reference_price, elasticity, remaining,
    fares, mus, sigmas, segment_median=reference_price, base_conversion=base_conv,
)
k = int(np.argmax(res.revenue))
p_book_opt = float(res.p_booking[k])
p_cancel_opt = float(res.p_cancel[k])
protections = emsrb_protection_levels(fares, mus, sigmas)
opened = open_classes(remaining, protections)

# ==========================================================================
# Resultado
# ==========================================================================
st.subheader(f"{hotel} · {SEASON_LABEL[season]} · {'fin de semana' if is_weekend else 'entre semana'}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tarifa vigente del segmento", f"${reference_price:,.2f}",
          help="Mediana observada de ADR para este hotel, segmento y mes. Dato, no supuesto.")
c2.metric("Bid price (EMSR-b)", f"${res.bid_price:,.2f}",
          help="Tarifa más baja que el control de inventario acepta con esta capacidad restante.")
c3.metric("ADR recomendado", f"${res.optimal_price:,.2f}",
          f"{(res.optimal_price / reference_price - 1) * 100:+.1f}% vs vigente")
c4.metric("P(reserva) · P(cancelación)", f"{p_book_opt:.0%} · {p_cancel_opt:.0%}")
c5.metric("Ingreso neto esperado", f"${res.optimal_revenue:,.2f}",
          help="Por solicitud recibida: ADR × noches × P(reserva) × ingreso que se retiene.")

if res.optimal_price <= res.bid_price * 1.001:
    st.warning(
        f"La restricción de inventario manda: con {remaining} habitaciones libres, EMSR-b "
        f"cierra las clases por debajo de ${res.bid_price:,.2f}. Vender más barato desplazaría "
        "demanda de tarifa alta que todavía se espera."
    )

tab1, tab2, tab3, tab4 = st.tabs(
    ["Curva de ingreso", "Control de inventario", "Sensibilidad", "Metodología y datos"]
)

# ---------------------------------------------------------------- curva
with tab1:
    curve = pd.DataFrame({
        "ADR": res.price_grid,
        "Ingreso neto esperado": res.revenue,
        "P(reserva)": res.p_booking,
        "P(cancelación)": res.p_cancel,
    })

    base = alt.Chart(curve).encode(x=alt.X("ADR:Q", title="Tarifa ofrecida (ADR)"))
    area = base.mark_area(opacity=0.25).encode(
        y=alt.Y("Ingreso neto esperado:Q", title="Ingreso neto esperado por solicitud")
    )
    line = base.mark_line(strokeWidth=2.5).encode(y="Ingreso neto esperado:Q")
    opt = alt.Chart(pd.DataFrame({"x": [res.optimal_price]})).mark_rule(
        strokeWidth=2.5, color="#d62728"
    ).encode(x="x:Q")
    bidline = alt.Chart(pd.DataFrame({"x": [res.bid_price]})).mark_rule(
        strokeDash=[6, 4], color="#888"
    ).encode(x="x:Q")
    refline = alt.Chart(pd.DataFrame({"x": [reference_price]})).mark_rule(
        strokeDash=[2, 3], color="#2ca02c"
    ).encode(x="x:Q")

    st.altair_chart((area + line + bidline + refline + opt).properties(height=320),
                    width="stretch")
    st.caption(
        f"Rojo: ADR óptimo (${res.optimal_price:,.2f}) · gris punteado: bid price EMSR-b "
        f"(${res.bid_price:,.2f}) · verde punteado: tarifa vigente (${reference_price:,.2f})"
    )

    probs = curve.melt("ADR", ["P(reserva)", "P(cancelación)"], "curva", "valor")
    st.altair_chart(
        alt.Chart(probs).mark_line(strokeWidth=2).encode(
            x=alt.X("ADR:Q", title="Tarifa ofrecida (ADR)"),
            y=alt.Y("valor:Q", title="Probabilidad", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("curva:N", title=None),
        ).properties(height=240),
        width="stretch",
    )
    st.caption(
        "P(reserva) es el supuesto de demanda; P(cancelación) sale del modelo XGBoost "
        "evaluado **en cada precio candidato**, porque la tarifa es una de sus variables."
    )

# ------------------------------------------------------- inventario
with tab2:
    st.markdown("#### Niveles de protección EMSR-b")
    inv = pd.DataFrame({
        "Clase": [c["nombre"] for c in H["clases"]],
        "Tarifa": [f"${f:,.2f}" for f in fares],
        "Demanda esperada (μ)": [f"{m:.1f}" for m in mus],
        "Desviación (σ)": [f"{s:.1f}" for s in sigmas],
        "Estado": ["ABIERTA" if i in opened else "cerrada" for i in range(len(fares))],
    })
    inv["Protección acumulada"] = [f"{p:.1f} hab." for p in protections] + ["—"]
    st.dataframe(inv, width="stretch", hide_index=True)
    st.caption(
        "μ y σ son la media y desviación de reservas por fecha de llegada en esta temporada "
        "y tipo de día, medidas sobre reservas que sí se materializaron."
    )

    steps = sorted({1, 5, 10, 20, 30, 45, 60, 90, 120, capacity} & set(range(1, capacity + 1)))
    ladder = pd.DataFrame({
        "Habitaciones libres": steps,
        "Bid price": [bid_price(x, fares, protections) for x in steps],
    })
    st.altair_chart(
        alt.Chart(ladder).mark_line(point=True, strokeWidth=2.5, interpolate="step-after").encode(
            x=alt.X("Habitaciones libres:Q"),
            y=alt.Y("Bid price:Q", title="Bid price ($)"),
        ).properties(height=260),
        width="stretch",
    )
    st.caption("Al vaciarse el inventario el piso sube por escalones: se cierran las clases baratas.")

    st.markdown("#### Sobreventa")
    tol = st.select_slider("Tolerancia a dejar huéspedes fuera", [0.01, 0.02, 0.05, 0.10], 0.05)
    base_rate = METRICS["prueba_out_of_time_sin_calibrar"]["tasa_base"]
    auth = overbooking_limit(capacity, base_rate, tol)
    o1, o2, o3 = st.columns(3)
    o1.metric("Capacidad física", f"{capacity}")
    o2.metric("Autorización recomendada", f"{auth}", f"+{auth - capacity} habitaciones")
    o3.metric("Tasa de cancelación observada", f"{base_rate:.1%}")
    st.caption(
        "Presentados ~ Binomial(autorización, 1 − tasa de cancelación). Se autoriza el máximo "
        f"que mantiene P(presentados > capacidad) ≤ {tol:.0%}."
    )

# ------------------------------------------------------ sensibilidad
with tab3:
    st.markdown("#### La recomendación depende de un supuesto que no se pudo estimar")
    st.dataframe(
        sensitivity(bundle, booking, reference_price, remaining, fares, mus, sigmas,
                    [-0.4, -0.8, -1.2, -1.5, -2.0, -2.5], reference_price, base_conv),
        width="stretch", hide_index=True,
    )
    m, s = logistic_params(reference_price, elasticity, base_conv)
    st.caption(
        f"Curva actual: disposición a pagar logística con mediana ${m:,.2f} y escala {s:,.2f}, "
        f"calibrada para dar conversión {base_conv:.0%} y elasticidad {elasticity:.1f} "
        f"en la tarifa vigente."
    )
    st.info(
        "Si la recomendación cambia mucho entre −0.8 y −1.5, la decisión no está sostenida por "
        "los datos sino por el supuesto. Es la primera cosa que habría que resolver con un test "
        "A/B de tarifas antes de llevar esto a producción."
    )

# ------------------------------------------------------- metodologia
with tab4:
    st.markdown("#### De dónde sale cada número")
    st.dataframe(pd.DataFrame([
        {"Componente": "Capacidad del hotel", "Origen": "Dato observado",
         "Detalle": f"Percentil 99 de ocupación diaria reconstruida → {capacity} habitaciones"},
        {"Componente": "Escalera tarifaria", "Origen": "Dato observado",
         "Detalle": "Cuartiles de ADR de reservas materializadas, por hotel"},
        {"Componente": "Demanda por clase (μ, σ)", "Origen": "Dato observado",
         "Detalle": "Reservas por fecha de llegada, por temporada y tipo de día"},
        {"Componente": "Tarifa vigente del segmento", "Origen": "Dato observado",
         "Detalle": "Mediana de ADR por hotel × segmento × mes"},
        {"Componente": "P(cancelación)", "Origen": "Modelo entrenado",
         "Detalle": f"XGBoost sobre 117k reservas · AUC out-of-time "
                    f"{METRICS['prueba_out_of_time_sin_calibrar']['roc_auc']:.4f}"},
        {"Componente": "Niveles de protección", "Origen": "Fórmula estándar",
         "Detalle": "EMSR-b (Belobaba 1989), generaliza Littlewood (1972)"},
        {"Componente": "Sobreventa", "Origen": "Fórmula estándar",
         "Detalle": "Cola binomial de presentados contra capacidad"},
        {"Componente": "Elasticidad precio", "Origen": "⚠️ SUPUESTO",
         "Detalle": "No se puede estimar con estos datos. Ver abajo."},
        {"Componente": "Conversión base", "Origen": "⚠️ SUPUESTO",
         "Detalle": "El dataset no registra solicitudes que no convirtieron"},
    ]), width="stretch", hide_index=True)

    st.markdown("#### Por qué la elasticidad es un supuesto y no una estimación")
    st.markdown(
        "El dataset solo contiene reservas **concretadas**: no hay búsquedas sin conversión ni "
        "tarifas rechazadas. Además el hotel sube precios justo cuando la demanda está fuerte, "
        "así que precio y cantidad se determinan a la vez."
    )
    esp = ELAST["mco_por_especificacion"]
    st.dataframe(pd.DataFrame([
        {"Especificación": k.replace("_", " "), "Elasticidad": f"{v['elasticidad']:+.3f}",
         "IC 95%": f"[{v['ic95'][0]:+.3f}, {v['ic95'][1]:+.3f}]", "R²": f"{v['r2']:.3f}"}
        for k, v in esp.items()
    ]), width="stretch", hide_index=True)
    iv = ELAST["variable_instrumental"]
    st.error(
        f"Todas dan elasticidad **positiva**, lo que es imposible para un bien normal: es sesgo "
        f"de simultaneidad de manual. Se intentó corregir con variable instrumental "
        f"(precio del mismo segmento en el otro hotel): el 2SLS da "
        f"{iv['iv_2sls']['elasticidad']:+.3f}, pero su IC cruza el cero y la primera etapa sale "
        f"negativa ({iv['primera_etapa_coef']:+.3f}), señal de sustitución entre los dos hoteles. "
        f"Eso rompe la restricción de exclusión: **el instrumento no es válido**."
    )

    st.markdown("#### El artefacto de `deposit_type`")
    art = ELAST and METRICS["artefacto_deposit_type"]
    st.dataframe(pd.DataFrame([
        {"Política": k, "Reservas": f"{v['n']:,}", "Tasa de cancelación": f"{v['tasa_cancelacion']:.1%}"}
        for k, v in art.items()
    ]), width="stretch", hide_index=True)
    st.markdown(
        "Las no reembolsables aparecen canceladas el **99.4%** de las veces. No es comportamiento: "
        "es cómo quedó registrado el no-show, y **el hotel ya cobró**. Un modelo que no lo detecta "
        "aprende «no reembolsable = cancela» y concluye que hay que dejar de venderlas, que es "
        "exactamente al revés. Aquí se excluyen del clasificador y se tratan en la ecuación "
        "económica con retención de ingreso del 100%."
    )

    st.markdown("#### Métricas del modelo de cancelación")
    t = METRICS["prueba_out_of_time_sin_calibrar"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC AUC", f"{t['roc_auc']:.4f}")
    m2.metric("PR AUC", f"{t['pr_auc']:.4f}")
    m3.metric("Brier", f"{t['brier']:.4f}", f"vs {METRICS['baseline_tasa_base']['brier']:.4f} base",
              delta_color="inverse")
    m4.metric("Reservas de prueba", f"{t['n']:,}")
    cal_df = pd.DataFrame({
        "Predicho": t["calibracion"]["predicho"],
        "Observado": t["calibracion"]["observado"],
    })
    st.altair_chart(
        (alt.Chart(cal_df).mark_point(size=80, filled=True).encode(
            x=alt.X("Predicho:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Observado:Q", scale=alt.Scale(domain=[0, 1])))
         + alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
             strokeDash=[4, 4], color="#999").encode(x="x:Q", y="y:Q")
         ).properties(height=280),
        width="stretch",
    )
    st.caption(
        "Partición **temporal** por fecha de reserva (no aleatoria): se entrena con lo reservado "
        "antes del 1 de marzo de 2017 y se evalúa con lo posterior. El modelo sobreestima en el "
        "decil más alto, efecto de que la tasa base cayó de 38.5% a 28.2% entre ambos periodos."
    )

st.markdown("---")
st.caption(
    "Datos: Antonio, Almeida & Nunes (2019), *Hotel booking demand datasets*, Data in Brief 22, "
    "41-49. Dos hoteles portugueses, 2015-2017. Proyecto de demostración: las cifras describen "
    "esos hoteles, no a Accor."
)
