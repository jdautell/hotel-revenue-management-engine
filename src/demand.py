"""
Capa 3: estructura tarifaria y pronostico de demanda por clase.

Todo lo que se calcula aqui sale de datos observados. No hay supuestos.

Que produce:
  1. CAPACIDAD de cada hotel, estimada de la ocupacion real. El dataset no trae
     el numero de habitaciones, asi que se expande cada reserva materializada a
     noches-habitacion, se cuenta la ocupacion diaria y se toma el percentil 99.
     City Hotel ~ 230 habitaciones, Resort Hotel ~ 190.
  2. CLASES TARIFARIAS por cuartiles de ADR dentro de cada hotel, con su tarifa
     representativa. Es la escalera de tarifas (Y/B/M/Q en la nomenclatura
     clasica) que necesita EMSR-b.
  3. DEMANDA POR CLASE: media y desviacion estandar de reservas por fecha de
     llegada, segmentada por temporada y por fin de semana. EMSR-b necesita
     exactamente eso: mu_k y sigma_k por clase.

La demanda se cuenta sobre reservas NO canceladas, que es la demanda que
realmente consume inventario.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import build_dataset

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

N_CLASSES = 4
CLASS_NAMES = ["Y (alta)", "B (media-alta)", "M (media-baja)", "Q (baja)"]

# Ventana con cobertura completa de los dos hoteles.
WINDOW = ("2015-09-01", "2017-07-31")


def occupancy_series(df: pd.DataFrame) -> pd.Series:
    """Noches-habitacion ocupadas por dia, expandiendo cada estancia."""
    stays = df[df["is_canceled"] == 0]
    spans = [
        pd.date_range(a, periods=int(n))
        for a, n in zip(stays["arrival_date"].values, stays["total_nights"].values)
    ]
    if not spans:
        return pd.Series(dtype=int)
    occ = pd.Series(np.concatenate([s.values for s in spans])).value_counts().sort_index()
    lo, hi = pd.Timestamp(WINDOW[0]), pd.Timestamp(WINDOW[1])
    return occ[(occ.index >= lo) & (occ.index <= hi)]


def estimate_capacity(df: pd.DataFrame) -> int:
    """Percentil 99 de la ocupacion diaria, redondeado a la decena."""
    occ = occupancy_series(df)
    return int(np.ceil(occ.quantile(0.99) / 10) * 10)


def season_tier(monthly_demand: pd.Series) -> dict[int, str]:
    """Clasifica los 12 meses en alta / media / baja segun demanda observada."""
    ranked = monthly_demand.rank(pct=True)
    return {
        int(m): ("alta" if r > 2 / 3 else "media" if r > 1 / 3 else "baja")
        for m, r in ranked.items()
    }


def build_calibration(csv_path: str | Path) -> dict:
    train, test, _ = build_dataset(csv_path)
    df = pd.concat([train, test])
    df["is_weekend"] = df["arrival_date"].dt.dayofweek.isin([4, 5]).astype(int)

    out: dict = {
        "fuente": "hotel_bookings.csv (Antonio, Almeida & Nunes 2019)",
        "ventana": {"desde": WINDOW[0], "hasta": WINDOW[1]},
        "hoteles": {},
    }

    for hotel, g in df.groupby("hotel"):
        capacity = estimate_capacity(g)
        occ = occupancy_series(g)

        # Clases tarifarias por cuartiles de ADR de las reservas materializadas.
        sold = g[g["is_canceled"] == 0].copy()
        edges = np.array(sold["adr"].quantile([0, 0.25, 0.50, 0.75, 1.0]), dtype=float)
        edges[0], edges[-1] = 0.0, np.inf
        sold["fare_class"] = pd.cut(
            sold["adr"], bins=edges, labels=range(N_CLASSES), include_lowest=True
        ).astype(int)

        # La clase 0 debe ser la MAS CARA: se invierte el orden de los cuartiles.
        sold["fare_class"] = (N_CLASSES - 1) - sold["fare_class"]

        fares = (
            sold.groupby("fare_class")["adr"]
            .agg(["mean", "median", "size"])
            .sort_index()
        )

        # Temporada por demanda mensual observada.
        monthly = sold.groupby(sold["arrival_date"].dt.month).size()
        tiers = season_tier(monthly)
        sold["season"] = sold["arrival_date"].dt.month.map(tiers)

        # Demanda por clase y por fecha de llegada -> mu_k y sigma_k.
        by_date = (
            sold.groupby(["season", "is_weekend", "fare_class", "arrival_date"])
            .size()
            .rename("q")
            .reset_index()
        )
        stats = (
            by_date.groupby(["season", "is_weekend", "fare_class"])["q"]
            .agg(["mean", "std", "size"])
            .fillna(0.0)
        )

        demanda: dict = {}
        for (season, wknd, cls), row in stats.iterrows():
            demanda.setdefault(season, {}).setdefault(str(int(wknd)), {})[str(int(cls))] = {
                "mu": round(float(row["mean"]), 2),
                "sigma": round(float(max(row["std"], 0.5)), 2),
                "n_fechas": int(row["size"]),
            }

        out["hoteles"][hotel] = {
            "capacidad_estimada": capacity,
            "ocupacion_diaria": {
                "media": round(float(occ.mean()), 1),
                "p50": float(occ.median()),
                "p95": float(occ.quantile(0.95)),
                "max": int(occ.max()),
            },
            "clases": [
                {
                    "id": int(c),
                    "nombre": CLASS_NAMES[int(c)],
                    "tarifa": round(float(fares.loc[c, "mean"]), 2),
                    "adr_mediano": round(float(fares.loc[c, "median"]), 2),
                    "n_reservas": int(fares.loc[c, "size"]),
                }
                for c in sorted(fares.index)
            ],
            "temporada_por_mes": {str(k): v for k, v in tiers.items()},
            "demanda_por_clase": demanda,
        }

    return out


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    cal = build_calibration(ROOT / "data" / "hotel_bookings.csv")
    (MODELS / "demand_calibration.json").write_text(
        json.dumps(cal, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for hotel, h in cal["hoteles"].items():
        print(f"\n=== {hotel} ===")
        print(f"  capacidad estimada: {h['capacidad_estimada']} habitaciones "
              f"(ocupacion media {h['ocupacion_diaria']['media']}, max {h['ocupacion_diaria']['max']})")
        print("  escalera tarifaria:")
        for c in h["clases"]:
            print(f"    {c['id']}  {c['nombre']:<16} ${c['tarifa']:>7.2f}   n={c['n_reservas']:,}")
        alta = h["demanda_por_clase"].get("alta", {}).get("1", {})
        if alta:
            print("  demanda en temporada alta, fin de semana (mu / sigma por clase):")
            for k in sorted(alta):
                print(f"    clase {k}: mu={alta[k]['mu']:.2f}  sigma={alta[k]['sigma']:.2f}")


if __name__ == "__main__":
    main()
