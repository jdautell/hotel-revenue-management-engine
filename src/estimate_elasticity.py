"""
Capa 2: elasticidad precio-demanda.

Este archivo existe para documentar un resultado NEGATIVO, y es a proposito.

La pregunta: se puede estimar la elasticidad precio de la demanda con este
dataset? La respuesta corta es NO, y saber por que vale mas que inventar un
numero.

EL PROBLEMA DE FONDO
--------------------
El dataset solo contiene reservas que SI se concretaron. No hay busquedas sin
conversion, ni tarifas ofrecidas y rechazadas, ni datos de rate shopping.
Estimar elasticidad exige observar demanda que NO compro a un precio dado, y eso
aqui no existe.

Ademas hay simultaneidad: el hotel sube la tarifa justamente cuando la demanda
esta fuerte. Precio y cantidad se determinan a la vez, asi que la correlacion
observada mezcla la curva de demanda con la de oferta.

QUE SE HIZO
-----------
1. Panel llegada x hotel x segmento (cantidad = reservas, precio = ADR mediano),
   estimacion log-log con controles cada vez mas ricos.
2. Correccion de endogeneidad con variable instrumental estilo Hausman: el
   precio del mismo segmento en el OTRO hotel, misma fecha.

QUE SALIO
---------
Las estimaciones por MCO dan elasticidad POSITIVA: subir el precio "aumenta" la
demanda. Es economicamente imposible en un bien normal y es el sesgo de
simultaneidad de manual. Los controles lo reducen (+0.475 -> +0.184) pero no lo
eliminan.

La variable instrumental empuja la estimacion hacia cero (-0.021) pero su
intervalo de confianza cruza el cero y, sobre todo, **la primera etapa sale
negativa**. Si el precio del otro hotel mueve la demanda de este en contra, hay
sustitucion entre ambos; y si hay sustitucion, el instrumento afecta la demanda
por un canal directo y se rompe la restriccion de exclusion. El instrumento no
sirve, y forzarlo seria peor que no usarlo.

CONSECUENCIA PARA EL OPTIMIZADOR
--------------------------------
La elasticidad NO se presenta como parametro estimado. Entra como **supuesto
explicito y ajustable**, anclado al rango publicado para hoteleria
(aproximadamente -0.4 a -1.5; mas elastico en ocio, menos en corporativo de
ultimo minuto), y el dashboard obliga a mirar el analisis de sensibilidad.

Es lo que hace la industria cuando no tiene datos experimentales. La forma
correcta de obtenerlos es un test A/B de tarifas o datos de rate shopping; sin
eso, cualquier elasticidad "estimada" con estos datos es un numero inventado con
pasos intermedios.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from data import build_dataset

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# Rango de referencia para elasticidad de demanda hotelera.
# Es el ancla del supuesto, no un resultado de este analisis.
ELASTICITY_PRIOR = {"min": -1.5, "default": -1.2, "max": -0.4}

CONTROLS = "C(hotel) + C(market_segment) + C(week) + C(dow) + C(year)"


def build_panel(df: pd.DataFrame, min_bookings: int = 3) -> pd.DataFrame:
    """Panel llegada x hotel x segmento: cantidad vendida y tarifa vigente."""
    panel = (
        df.groupby(["hotel", "market_segment", "arrival_date"])
        .agg(q=("adr", "size"), price=("adr", "median"))
        .reset_index()
    )
    panel = panel[panel["q"] >= min_bookings].copy()
    panel["lq"] = np.log(panel["q"])
    panel["lp"] = np.log(panel["price"])
    panel["dow"] = panel["arrival_date"].dt.dayofweek.astype(str)
    panel["year"] = panel["arrival_date"].dt.year.astype(str)
    panel["month"] = panel["arrival_date"].dt.month.astype(str)
    panel["week"] = panel["arrival_date"].dt.isocalendar().week.astype(int).astype(str)
    return panel


def ols(panel: pd.DataFrame, formula: str, term: str = "lp") -> dict:
    """MCO con errores agrupados por fecha de llegada."""
    m = smf.ols(formula, data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["arrival_date"]}
    )
    lo, hi = m.conf_int().loc[term]
    return {
        "elasticidad": round(float(m.params[term]), 4),
        "ic95": [round(float(lo), 4), round(float(hi), 4)],
        "r2": round(float(m.rsquared), 4),
        "n": int(m.nobs),
    }


def iv_two_stage(panel: pd.DataFrame) -> dict:
    """
    2SLS con instrumento tipo Hausman: precio del mismo segmento en el otro
    hotel, misma fecha. Se reporta la primera etapa porque es justo lo que
    revela que el instrumento no es valido.
    """
    other = panel[["hotel", "market_segment", "arrival_date", "lp"]].copy()
    other["hotel"] = other["hotel"].map(
        {"City Hotel": "Resort Hotel", "Resort Hotel": "City Hotel"}
    )
    other = other.rename(columns={"lp": "lp_other"})
    p = panel.merge(other, on=["hotel", "market_segment", "arrival_date"], how="inner")

    first = smf.ols(f"lp ~ lp_other + {CONTROLS}", data=p).fit(
        cov_type="cluster", cov_kwds={"groups": p["arrival_date"]}
    )
    f_stat = float((first.params["lp_other"] / first.bse["lp_other"]) ** 2)
    p["lp_hat"] = first.fittedvalues

    return {
        "n": int(len(p)),
        "primera_etapa_coef": round(float(first.params["lp_other"]), 4),
        "primera_etapa_F": round(f_stat, 1),
        "ols_misma_muestra": ols(p, f"lq ~ lp + {CONTROLS}"),
        "iv_2sls": ols(p, f"lq ~ lp_hat + {CONTROLS}", term="lp_hat"),
        "veredicto": (
            "INSTRUMENTO INVALIDO. La primera etapa es negativa: el precio del "
            "otro hotel se mueve en contra, senal de sustitucion entre ambos. Si "
            "hay sustitucion, el instrumento afecta la demanda por un canal "
            "directo y se rompe la restriccion de exclusion. El IC95% del 2SLS "
            "ademas cruza el cero."
        ),
    }


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    train, test, _ = build_dataset(ROOT / "data" / "hotel_bookings.csv")
    panel = build_panel(pd.concat([train, test]))

    especificaciones = {
        "1_sin_controles": ols(panel, "lq ~ lp"),
        "2_hotel_y_segmento": ols(panel, "lq ~ lp + C(hotel) + C(market_segment)"),
        "3_mas_mes_dow_anio": ols(
            panel,
            "lq ~ lp + C(hotel) + C(market_segment) + C(month) + C(dow) + C(year)",
        ),
        "4_semana_del_anio": ols(panel, f"lq ~ lp + {CONTROLS}"),
    }

    out = {
        "pregunta": "Se puede estimar la elasticidad precio con este dataset?",
        "respuesta": "No. Ver 'conclusion'.",
        "n_celdas_panel": int(len(panel)),
        "mco_por_especificacion": especificaciones,
        "variable_instrumental": iv_two_stage(panel),
        "supuesto_usado_en_el_optimizador": ELASTICITY_PRIOR,
        "conclusion": (
            "Todas las especificaciones por MCO dan elasticidad POSITIVA, "
            "imposible para un bien normal, y evidencian sesgo de simultaneidad: "
            "el hotel sube tarifas cuando la demanda esta fuerte. Los controles "
            "reducen el sesgo pero no lo eliminan, y el unico instrumento "
            "disponible falla la restriccion de exclusion. Por eso la elasticidad "
            "entra al optimizador como SUPUESTO explicito con analisis de "
            "sensibilidad, no como parametro estimado. Obtenerla de verdad "
            "requiere test A/B de tarifas o datos de rate shopping."
        ),
    }

    (REPORTS / "elasticity_analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Panel: {out['n_celdas_panel']:,} celdas (llegada x hotel x segmento)\n")
    print("Elasticidad por MCO (todas con el signo equivocado):")
    for k, v in especificaciones.items():
        print(
            f"  {k:<22} {v['elasticidad']:+.3f}  "
            f"IC95% [{v['ic95'][0]:+.3f}, {v['ic95'][1]:+.3f}]  R2={v['r2']:.3f}"
        )
    iv = out["variable_instrumental"]
    print(f"\nVariable instrumental (n={iv['n']:,}):")
    print(f"  primera etapa: coef={iv['primera_etapa_coef']:+.3f}  F={iv['primera_etapa_F']}")
    print(
        f"  MCO {iv['ols_misma_muestra']['elasticidad']:+.3f}  ->  "
        f"2SLS {iv['iv_2sls']['elasticidad']:+.3f} "
        f"IC95% [{iv['iv_2sls']['ic95'][0]:+.3f}, {iv['iv_2sls']['ic95'][1]:+.3f}]"
    )
    print(f"\n  {iv['veredicto']}")
    print(f"\nSupuesto que usa el optimizador: {ELASTICITY_PRIOR}")


if __name__ == "__main__":
    main()
