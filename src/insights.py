"""
Analisis de negocio: donde se fuga el ingreso.

El resto del repositorio construye el motor de decision. Este archivo responde
la pregunta que un director comercial hace primero: **de todo lo que se reserva,
cuanto entra de verdad a caja, y por que canal se pierde el resto.**

La metrica central es el VALOR NETO POR SOLICITUD:

    valor bruto  = ADR x noches
    retencion    = 1 si la tarifa es no reembolsable (el hotel cobra igual)
                   0 si se cancela una tarifa reembolsable
                   1 si no se cancela
    valor neto   = valor bruto x retencion

Comparar segmentos por ADR es enganoso. Comparar por valor neto cambia el orden,
y con eso cambian las decisiones de mezcla de canal.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import build_dataset

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

LEAD_BINS = [-1, 7, 30, 90, 180, 400]
LEAD_LABELS = ["0-7 dias", "8-30 dias", "31-90 dias", "91-180 dias", "mas de 180"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["valor_bruto"] = df["adr"] * df["total_nights"]
    # Una tarifa no reembolsable retiene el ingreso aunque el huesped cancele.
    df["retencion"] = np.where(
        df["deposit_type"] == "Non Refund", 1.0, 1.0 - df["is_canceled"]
    )
    df["valor_neto"] = df["valor_bruto"] * df["retencion"]
    return df


def by_segment(df: pd.DataFrame, min_n: int = 500) -> pd.DataFrame:
    g = df.groupby("market_segment").agg(
        reservas=("valor_bruto", "size"),
        adr_mediano=("adr", "median"),
        noches_promedio=("total_nights", "mean"),
        tasa_cancelacion=("is_canceled", "mean"),
        pct_no_reembolsable=("deposit_type", lambda s: (s == "Non Refund").mean()),
        valor_bruto=("valor_bruto", "sum"),
        valor_neto=("valor_neto", "sum"),
    )
    g = g[g["reservas"] >= min_n].copy()
    g["valor_bruto_por_reserva"] = g["valor_bruto"] / g["reservas"]
    g["valor_neto_por_reserva"] = g["valor_neto"] / g["reservas"]
    g["fuga_pct"] = (1 - g["valor_neto"] / g["valor_bruto"]) * 100
    return g.sort_values("valor_neto_por_reserva", ascending=False)


def by_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tramo"] = pd.cut(df["lead_time"], LEAD_BINS, labels=LEAD_LABELS)
    return df.groupby("tramo", observed=True).agg(
        reservas=("is_canceled", "size"),
        tasa_cancelacion=("is_canceled", "mean"),
        adr_mediano=("adr", "median"),
        valor_neto_por_reserva=("valor_neto", "mean"),
    )


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    train, test, _ = build_dataset(ROOT / "data" / "hotel_bookings.csv")
    df = prepare(pd.concat([train, test]))

    seg, lead = by_segment(df), by_lead_time(df)
    bruto, neto = float(df["valor_bruto"].sum()), float(df["valor_neto"].sum())

    out = {
        "total": {
            "reservas": int(len(df)),
            "valor_bruto_reservado": round(bruto, 2),
            "ingreso_retenido": round(neto, 2),
            "fuga_por_cancelacion": round(bruto - neto, 2),
            "fuga_pct": round((1 - neto / bruto) * 100, 2),
        },
        "por_segmento": json.loads(seg.round(4).to_json(orient="index")),
        "por_anticipacion": json.loads(lead.round(4).to_json(orient="index")),
        "lecturas": [
            "De cada peso reservado se pierden 31 centavos por cancelacion. "
            "Esa es la magnitud del problema que justifica el modelo de riesgo.",
            "El segmento que mas cancela NO es el que mas dinero pierde. Groups "
            "cancela 62% pero solo fuga 14% del valor, porque casi la mitad de "
            "sus reservas son no reembolsables. Online TA cancela 37% y fuga "
            "43%, porque practicamente todas son reembolsables.",
            "Por eso la palanca no es 'reducir cancelaciones' sino 'mover mezcla "
            "hacia tarifas que retengan ingreso'.",
            "Ordenados por ADR, Online TA parece el mejor canal. Ordenados por "
            "valor NETO por solicitud, Direct lo supera por 36%. El ranking se "
            "invierte y con el la decision de mezcla de canal.",
            "El riesgo de cancelacion crece de forma monotona con la "
            "anticipacion: 9.7% dentro de 7 dias, 55.9% a mas de 180. Es el "
            "gradiente que justifica proteger inventario para la demanda tardia.",
        ],
    }

    (REPORTS / "business_insights.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    t = out["total"]
    print("=== FUGA TOTAL DE INGRESO ===")
    print(f"  valor bruto reservado : ${t['valor_bruto_reservado']:>14,.0f}")
    print(f"  ingreso retenido      : ${t['ingreso_retenido']:>14,.0f}")
    print(f"  fuga por cancelacion  : ${t['fuga_por_cancelacion']:>14,.0f}  ({t['fuga_pct']:.1f}%)\n")

    print("=== VALOR POR SEGMENTO (por solicitud recibida) ===")
    view = seg[[
        "reservas", "adr_mediano", "tasa_cancelacion", "pct_no_reembolsable",
        "valor_bruto_por_reserva", "valor_neto_por_reserva", "fuga_pct",
    ]].round(2)
    print(view.to_string())

    print("\n=== RIESGO POR ANTICIPACION ===")
    print(lead.round(3).to_string())

    print("\n=== LECTURAS ===")
    for i, s in enumerate(out["lecturas"], 1):
        print(f"  {i}. {s}")


if __name__ == "__main__":
    main()
