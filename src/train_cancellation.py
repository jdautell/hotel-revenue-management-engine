"""
Modelo 1: riesgo de cancelacion.

Es la unica capa del proyecto que se entrena con etiquetas observadas de verdad:
`is_canceled`, el desenlace real de ~117,000 reservas.

Por que importa en revenue management: una reserva de $200 con 60% de riesgo de
cancelar vale menos que una de $150 con 5%. Los RMS reales descuentan el ingreso
por riesgo antes de decidir. Sin esta capa se optimiza ingreso que nunca entra
a caja.

--------------------------------------------------------------------------
EL ARTEFACTO DE `deposit_type` (lo mas importante de este archivo)
--------------------------------------------------------------------------
Un XGBoost entrenado sobre el dataset completo le asigna a `deposit_type` el
62% de la importancia. Motivo:

    deposit_type      n        tasa de cancelacion
    No Deposit        102,650  28.7%
    Non Refund         14,586  99.4%     <-- ojo
    Refundable            162  22.2%

Las no reembolsables "cancelan" el 99.4% de las veces. Eso no es
comportamiento: es como quedo registrado. En una tarifa no reembolsable el
huesped que no llega queda marcado como cancelacion, pero **el hotel ya cobro**.

Si no se detecta, el modelo aprende "no reembolsable = cancela" y el optimizador
concluye que hay que dejar de vender tarifas no reembolsables. Es exactamente al
reves de la practica correcta: la no reembolsable es la que asegura el ingreso.

Aqui se resuelve separando lo que de verdad importa, que no es la cancelacion
sino la PERDIDA DE INGRESO:

  - Tarifas reembolsables -> cancelar destruye el ingreso. Se modelan.
  - Tarifas no reembolsables -> el ingreso queda asegurado. No entran al modelo;
    en el optimizador se tratan con retencion de ingreso ~100%.

Resultado de quitar el artefacto: el AUC out-of-time queda en 0.8375, contra
0.8412 del modelo contaminado, y el error maximo de calibracion MEJORA. O sea
que no se pago nada por la honestidad. Ademas el modelo pasa a explicarse por
comportamiento del huesped -estacionamiento, cancelaciones previas, segmento,
peticiones especiales- en vez de por una peculiaridad de registro.

La columna `deposit_type` tampoco entra como variable del modelo: se usa en la
ecuacion economica del optimizador, donde le corresponde. Es una palanca
comercial (define si cancelar destruye el ingreso), no una prediccion de
comportamiento.
--------------------------------------------------------------------------
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from xgboost import XGBClassifier

from data import TARGET, build_dataset, feature_frame

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    enable_categorical=True,
    tree_method="hist",
    eval_metric="logloss",
    early_stopping_rounds=50,
    random_state=42,
)


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    return {
        "n": int(len(y)),
        "tasa_base": round(float(np.mean(y)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, p)), 4),
        "calibracion": {
            "predicho": [round(float(v), 4) for v in mean_pred],
            "observado": [round(float(v), 4) for v in frac_pos],
            "error_max": round(float(np.max(np.abs(mean_pred - frac_pos))), 4),
        },
    }


def deposit_artifact_table(df: pd.DataFrame) -> dict:
    g = df.groupby("deposit_type")["is_canceled"].agg(["size", "mean"])
    return {k: {"n": int(v["size"]), "tasa_cancelacion": round(float(v["mean"]), 4)}
            for k, v in g.iterrows()}


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    train_all, test_all, medians = build_dataset(ROOT / "data" / "hotel_bookings.csv")
    artifact = deposit_artifact_table(pd.concat([train_all, test_all]))

    # Solo tarifas donde cancelar destruye ingreso.
    train = train_all[train_all["deposit_type"] != "Non Refund"].copy()
    test = test_all[test_all["deposit_type"] != "Non Refund"].copy()

    # Validacion tambien temporal: ultimo 15% del train por fecha de reserva.
    train = train.sort_values("booking_date")
    cut = int(len(train) * 0.85)
    fit_df, val_df = train.iloc[:cut], train.iloc[cut:]

    X_fit, y_fit = feature_frame(fit_df), fit_df[TARGET].to_numpy()
    X_val, y_val = feature_frame(val_df), val_df[TARGET].to_numpy()
    X_test, y_test = feature_frame(test), test[TARGET].to_numpy()

    model = XGBClassifier(**PARAMS)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    p_val_raw = model.predict_proba(X_val)[:, 1]
    p_test_raw = model.predict_proba(X_test)[:, 1]

    # Calibracion isotonica ajustada en validacion (nunca en prueba).
    # El optimizador multiplica por (1 - p), asi que "0.30" tiene que significar
    # 30 de cada 100, no solo que el orden sea correcto.
    #
    # RESULTADO: no mejora, y se documenta en vez de esconderse. La
    # descalibracion no viene del modelo sino de un cambio de regimen: la tasa
    # de cancelacion cae de 38.5% en entrenamiento a 28.2% en el periodo de
    # prueba. Una isotonica ajustada sobre datos del regimen viejo no arregla
    # eso. Tambien se probo recalibrar con los primeros 30 dias del periodo de
    # prueba, simulando produccion, y empeoro (Brier 0.1482 -> 0.1509): 3,393
    # observaciones no alcanzan para una isotonica estable.
    #
    # DECISION: se publican las probabilidades sin calibrar, que son las mejor
    # calibradas de las tres opciones medidas. El sesgo residual esta en el
    # decil mas alto (predice 0.81, se observa 0.69) y se anota como limitacion
    # conocida. En produccion se recalibraria con una ventana movil de varios
    # meses, no de uno.
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(p_val_raw, y_val)
    p_test_cal = calibrator.predict(p_test_raw)

    baseline = np.full_like(y_test, y_fit.mean(), dtype=float)
    metrics = {
        "corte_temporal": "2017-03-01 (fecha de reserva = llegada - lead time)",
        "excluidas": "deposit_type == 'Non Refund' (artefacto de registro, ver docstring)",
        "artefacto_deposit_type": artifact,
        "n_train": int(len(fit_df)),
        "n_val": int(len(val_df)),
        "mejor_iteracion": int(model.best_iteration),
        "validacion": evaluate(y_val, p_val_raw),
        "prueba_out_of_time_sin_calibrar": evaluate(y_test, p_test_raw),
        "prueba_out_of_time_calibrado": evaluate(y_test, p_test_cal),
        "decision_calibracion": (
            "Se publican las probabilidades SIN calibrar. La isotonica no mejora "
            "porque la descalibracion viene de un cambio de tasa base "
            "(38.5% -> 28.2%), no del modelo. Recalibrar con 30 dias del periodo "
            "de prueba tambien empeoro (Brier 0.1482 -> 0.1509)."
        ),
        "baseline_tasa_base": {
            "brier": round(float(brier_score_loss(y_test, baseline)), 4),
            "log_loss": round(float(log_loss(y_test, baseline)), 4),
        },
        "top_variables": {
            k: round(float(v), 4)
            for k, v in pd.Series(model.feature_importances_, index=X_fit.columns)
            .sort_values(ascending=False)
            .head(15)
            .items()
        },
    }

    joblib.dump(
        {
            "model": model,
            # Se guarda por trazabilidad del experimento; el optimizador NO lo usa.
            "calibrator_no_usado": calibrator,
            "medians": medians,
            "columns": list(X_fit.columns),
            "tasa_base_entrenamiento": float(y_fit.mean()),
        },
        MODELS / "cancellation_model.joblib",
    )
    (REPORTS / "cancellation_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    raw, cal = metrics["prueba_out_of_time_sin_calibrar"], metrics["prueba_out_of_time_calibrado"]
    print(f"Prueba out-of-time: n={raw['n']:,}  tasa base={raw['tasa_base']:.4f}")
    print(f"  AUC          {raw['roc_auc']:.4f}")
    print(f"  PR-AUC       {raw['pr_auc']:.4f}")
    print(f"  Brier        {raw['brier']:.4f} -> {cal['brier']:.4f} calibrado "
          f"(baseline {metrics['baseline_tasa_base']['brier']:.4f})")
    print(f"  Error max de calibracion  {raw['calibracion']['error_max']:.4f} -> "
          f"{cal['calibracion']['error_max']:.4f}")
    print("\nTop variables:")
    for k, v in list(metrics["top_variables"].items())[:8]:
        print(f"  {k:<32} {v:.4f}")


if __name__ == "__main__":
    main()
