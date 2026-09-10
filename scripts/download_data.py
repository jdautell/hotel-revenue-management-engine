"""
Descarga el dataset publico Hotel Booking Demand.

El CSV pesa ~17 MB y no se versiona en el repositorio: se baja con este script.
Los modelos entrenados si estan en models/, asi que el dashboard corre sin
descargar nada. Esto solo hace falta para reentrenar.

Fuente:
  Antonio, N., de Almeida, A., Nunes, L. (2019).
  "Hotel booking demand datasets". Data in Brief 22, 41-49.
  https://doi.org/10.1016/j.dib.2018.11.126

Espejo usado: TidyTuesday (2020-02-11), copia integra del dataset original.
"""
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"
DEST = Path(__file__).resolve().parents[1] / "data" / "hotel_bookings.csv"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando -> {DEST}")
    with urlopen(URL, timeout=180) as r:
        DEST.write_bytes(r.read())
    print(f"Listo: {DEST.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
