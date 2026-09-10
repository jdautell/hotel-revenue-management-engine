"""
Carga y preparacion del dataset Hotel Booking Demand.

Fuente: Antonio, N., de Almeida, A., Nunes, L. (2019). "Hotel booking demand
datasets". Data in Brief 22, 41-49. https://doi.org/10.1016/j.dib.2018.11.126

Dos decisiones de fondo, y las dos importan para que el modelo sea creible:

1. SOLO VARIABLES CONOCIDAS AL MOMENTO DE RESERVAR.
   El dataset trae columnas que solo existen DESPUES de que la reserva se
   resolvio (`reservation_status`, `reservation_status_date`,
   `assigned_room_type`). Entrenar con ellas da un AUC de 0.99 que se cae en
   produccion. Se descartan explicitamente.

2. PARTICION TEMPORAL, NO ALEATORIA.
   Un modelo de revenue management se entrena con el pasado y se aplica al
   futuro. Partir al azar mezcla reservas de agosto de 2017 en el
   entrenamiento para predecir julio de 2017: eso infla las metricas. Aqui se
   parte por FECHA DE RESERVA (llegada - lead time), que es el momento en que
   el modelo realmente tendria que decidir.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Columnas que solo se conocen despues del desenlace de la reserva.
LEAKAGE_COLS = ["reservation_status", "reservation_status_date", "assigned_room_type"]

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}

NUMERIC_FEATURES = [
    "lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "total_nights",
    "adults", "children", "babies", "total_guests",
    "previous_cancellations", "previous_bookings_not_canceled",
    "booking_changes", "days_in_waiting_list",
    "required_car_parking_spaces", "total_of_special_requests",
    "adr", "adr_ratio_segment",
    "is_repeated_guest", "has_agent", "has_company",
    "arrival_month", "arrival_week", "arrival_dow", "arrival_year",
]

# `deposit_type` NO es variable del modelo, y es deliberado. Ver el docstring de
# train_cancellation.py: las tarifas no reembolsables se excluyen del
# entrenamiento por un artefacto de registro, asi que la columna se queda casi
# sin variacion. Ademas conceptualmente no es una prediccion de comportamiento
# sino una PALANCA COMERCIAL: define si cancelar destruye o no el ingreso, y por
# eso entra en la ecuacion economica del optimizador, no en el clasificador.
CATEGORICAL_FEATURES = [
    "hotel", "market_segment", "distribution_channel",
    "customer_type", "meal", "reserved_room_type", "country_grouped",
]

TARGET = "is_canceled"


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Lee el CSV crudo tal cual viene de la fuente."""
    return pd.read_csv(csv_path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza minima y justificada. Cada filtro se explica."""
    df = df.copy()

    # Nulos documentados por los autores del dataset.
    df["children"] = df["children"].fillna(0)
    df["country"] = df["country"].fillna("UNK")
    df["has_agent"] = df["agent"].notna().astype(int)
    df["has_company"] = df["company"].notna().astype(int)

    # Reservas sin huespedes: filas corruptas, no son reservas reales.
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df = df[df["total_guests"] > 0]

    # Estancias de cero noches: no generan ingreso por habitacion.
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df = df[df["total_nights"] > 0]

    # ADR no positivo (cortesias, errores) y el unico outlier de $5,400.
    df = df[(df["adr"] > 0) & (df["adr"] < 1000)]

    return df.reset_index(drop=True)


def add_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Fecha de llegada y fecha de reserva (llegada - lead time)."""
    df = df.copy()
    df["arrival_month"] = df["arrival_date_month"].map(MONTHS)
    df["arrival_date"] = pd.to_datetime(
        dict(
            year=df["arrival_date_year"],
            month=df["arrival_month"],
            day=df["arrival_date_day_of_month"],
        )
    )
    df["booking_date"] = df["arrival_date"] - pd.to_timedelta(df["lead_time"], unit="D")
    df["arrival_week"] = df["arrival_date_week_number"]
    df["arrival_dow"] = df["arrival_date"].dt.dayofweek
    df["arrival_year"] = df["arrival_date_year"]
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Variables derivadas.

    `adr_ratio_segment` es la mas importante: cuanto se aparta la tarifa de esta
    reserva de la tarifa tipica de su mismo mercado, hotel y mes. Un ADR de $120
    no significa lo mismo en agosto que en noviembre. Se calcula SOLO con la
    mediana del periodo de entrenamiento para no filtrar informacion del futuro
    (ver `fit_segment_medians`).
    """
    df = df.copy()
    df["country_grouped"] = df["country"].where(
        df["country"].isin(df["country"].value_counts().head(20).index), "OTHER"
    )
    return df


def fit_segment_medians(train: pd.DataFrame) -> pd.DataFrame:
    """Mediana de ADR por hotel x segmento x mes, calculada solo con el train."""
    return (
        train.groupby(["hotel", "market_segment", "arrival_month"], observed=True)["adr"]
        .median()
        .rename("adr_segment_median")
        .reset_index()
    )


def apply_segment_medians(df: pd.DataFrame, medians: pd.DataFrame) -> pd.DataFrame:
    """Aplica las medianas del train a cualquier particion (train o test)."""
    df = df.merge(medians, on=["hotel", "market_segment", "arrival_month"], how="left")
    fallback = medians["adr_segment_median"].median()
    df["adr_segment_median"] = df["adr_segment_median"].fillna(fallback)
    df["adr_ratio_segment"] = df["adr"] / df["adr_segment_median"]
    return df


def temporal_split(df: pd.DataFrame, cutoff: str = "2017-03-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parte por FECHA DE RESERVA, no por fecha de llegada.

    El modelo, puesto en produccion el 1 de marzo de 2017, solo habria visto
    reservas hechas antes de esa fecha. Todo lo reservado despues es futuro
    genuino. Esto da metricas mas bajas que una particion aleatoria, y son las
    metricas honestas.
    """
    cut = pd.Timestamp(cutoff)
    train = df[df["booking_date"] < cut].copy()
    test = df[df["booking_date"] >= cut].copy()
    return train, test


def build_dataset(csv_path: str | Path, cutoff: str = "2017-03-01"):
    """Pipeline completo. Devuelve (train, test) listos para modelar."""
    df = load_raw(csv_path)
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
    df = clean(df)
    df = add_dates(df)
    df = add_features(df)

    train, test = temporal_split(df, cutoff)
    medians = fit_segment_medians(train)
    train = apply_segment_medians(train, medians)
    test = apply_segment_medians(test, medians)
    return train, test, medians


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de variables con las categoricas como `category` (XGBoost las lee nativo)."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")
    return X
