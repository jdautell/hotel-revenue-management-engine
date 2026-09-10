# Motor de Revenue Management hotelero

Sistema de decisión tarifaria que combina **control de inventario EMSR-b**, un
**modelo de riesgo de cancelación** entrenado con datos reales, y una **curva de
demanda cuyo supuesto está declarado y es ajustable**.

> 🔗 **Demo interactivo:** _(pendiente de despliegue)_
> 📊 **Datos:** 119,390 reservas de dos hoteles portugueses, 2015–2017

---

## El problema

Un revenue manager decide, para cada solicitud de reserva, dos cosas a la vez:

1. **¿Acepto esta reserva?** Vender hoy una habitación barata puede impedir
   venderla cara mañana. Eso es un costo de oportunidad, no un ingreso.
2. **¿A qué tarifa la cotizo?** Muy alta y no convierte; muy baja y se regala
   inventario que se habría vendido igual.

Y hay una tercera capa que casi ningún proyecto de portafolio incluye: **el 37%
de estas reservas se cancela**. Un ingreso reservado no es un ingreso cobrado.

Este proyecto resuelve las tres, y —tan importante como eso— **es explícito
sobre qué parte está sostenida por datos y qué parte por supuestos**.

---

## Arquitectura

| Componente | Origen | Detalle |
|---|---|---|
| Capacidad del hotel | **Dato observado** | Percentil 99 de ocupación diaria reconstruida noche a noche → City 230 hab., Resort 190 hab. |
| Escalera tarifaria (Y/B/M/Q) | **Dato observado** | Cuartiles de ADR de reservas materializadas |
| Demanda por clase (μ, σ) | **Dato observado** | Reservas por fecha de llegada, segmentadas por temporada y tipo de día |
| Tarifa vigente del segmento | **Dato observado** | Mediana de ADR por hotel × segmento × mes |
| P(cancelación) | **Modelo entrenado** | XGBoost · AUC out-of-time **0.8375** |
| Niveles de protección | **Fórmula estándar** | EMSR-b (Belobaba, 1989) |
| Límite de sobreventa | **Fórmula estándar** | Cola binomial de presentados vs. capacidad |
| Elasticidad precio | ⚠️ **Supuesto** | No es estimable con estos datos — ver abajo |
| Conversión base | ⚠️ **Supuesto** | El dataset no registra solicitudes que no convirtieron |

Esa última columna es el punto del proyecto. Un motor de pricing que no
distingue entre lo que midió y lo que asumió no se puede auditar, y un revenue
manager no debería firmarlo.

---

## La matemática

### 1. EMSR-b → *bid price*

Para clases `1..n` ordenadas de tarifa mayor a menor, el nivel de protección
`y_j` de las clases `1..j` frente a la clase `j+1` sale de igualar el ingreso
marginal de ambas opciones:

```
P( S_j > y_j ) = p_{j+1} / p̄_j

  S_j  = demanda agregada de las clases 1..j  ~  Normal(Σμ, √Σσ²)
  p̄_j  = tarifa promedio ponderada por demanda de las clases 1..j
```

Con `S_j` normal:  `y_j = μ_S + σ_S · Φ⁻¹(1 − p_{j+1}/p̄_j)`

Con capacidad restante `x`, la clase `j+1` está abierta si `x > y_j`. El **bid
price** es la tarifa más baja aún abierta: el costo de oportunidad real de la
última habitación.

Es la generalización multiclase de la regla de Littlewood (1972) y es lo que
usan los RMS comerciales a nivel de leg/noche. **No es una heurística
inventada.**

Comportamiento verificado (City Hotel, temporada alta, fin de semana):

| Habitaciones libres | Clases abiertas | Bid price |
|---:|---|---:|
| 120 | Y · B · M · Q | $69.06 |
| 40 | Y · B · M | $91.47 |
| 25 | Y · B | $113.52 |
| 15 | Y | $159.62 |

### 2. Curva de demanda

Disposición a pagar logística, calibrada con **dos condiciones que el analista
elige y puede discutir**:

```
P(reserva | p) = 1 / (1 + exp((p − m) / s))

  ε(p) = −(p/s) · (1 − P(p))
    ⟹  s = −p_ref · (1 − P₀) / ε
    ⟹  m = p_ref + s · ln(P₀ / (1 − P₀))
```

donde `p_ref` es la tarifa vigente observada, `P₀` la conversión que se asume a
esa tarifa, y `ε` la elasticidad asumida.

**Por qué no elasticidad constante (`p^ε`):** con esa forma funcional el
ingreso esperado es monótono en el precio, así que el óptimo siempre cae en un
extremo del intervalo. Eso no es un óptimo económico, es un artefacto de la
especificación. La logística sí tiene óptimo interior.

### 3. Ingreso esperado ajustado por riesgo

```
R(p) = p × noches × P(reserva | p) × [ (1 − pc(p)) + pc(p) · retención ]
```

sujeto a `p ≥ bid_price(x)`.

`pc(p)` es la probabilidad de cancelación del modelo XGBoost **evaluada en cada
precio candidato** — la tarifa es una de sus variables, así que el modelo forma
parte de la derivada, no es decorativo. `retención` vale 1.0 en tarifas no
reembolsables y 0.0 en reembolsables.

### 4. Sobreventa

Presentados ~ `Binomial(A, 1 − tasa de cancelación)`. Se autoriza el mayor `A`
tal que `P(presentados > capacidad) ≤ tolerancia`.

Con 230 habitaciones, 28.2% de cancelación y 5% de tolerancia: **autorizar 303
reservas**.

---

## Tres hallazgos que cambiaron el diseño

### 1. El artefacto de `deposit_type`

| Política | Reservas | Tasa de cancelación |
|---|---:|---:|
| No Deposit | 102,650 | 28.7% |
| **Non Refund** | **14,586** | **99.4%** |
| Refundable | 162 | 22.2% |

Un XGBoost entrenado sobre el dataset completo le da a `deposit_type` el **62%
de la importancia**. Pero ese 99.4% no es comportamiento: es cómo quedó
registrado el no-show en una tarifa no reembolsable, **donde el hotel ya cobró**.

Un modelo que no lo detecta aprende "no reembolsable = cancela" y el optimizador
concluye que hay que dejar de vender tarifas no reembolsables. Es exactamente al
revés: la no reembolsable es la que asegura el ingreso.

**Solución:** esas reservas se excluyen del clasificador y se tratan en la
ecuación económica con retención de ingreso del 100%. La columna deja de ser una
variable predictiva y pasa a ser lo que en realidad es, una **palanca
comercial**.

**Costo:** ninguno. El AUC out-of-time pasó de 0.8412 a **0.8375** y el error
máximo de calibración *mejoró*. A cambio, el modelo ahora se explica por
comportamiento del huésped: estacionamiento, cancelaciones previas, segmento,
peticiones especiales.

### 2. La elasticidad no es estimable con estos datos

Panel llegada × hotel × segmento, regresión log-log con controles crecientes:

| Especificación | Elasticidad | IC 95% | R² |
|---|---:|---|---:|
| Sin controles | **+0.475** | [+0.432, +0.518] | 0.050 |
| + hotel, segmento | **+0.325** | [+0.279, +0.371] | 0.444 |
| + mes, día, año | **+0.208** | [+0.138, +0.278] | 0.468 |
| + semana del año | **+0.184** | [+0.113, +0.255] | 0.475 |

**Todas dan elasticidad positiva**, lo cual es imposible para un bien normal.
Es sesgo de simultaneidad de manual: el hotel sube tarifas justo cuando la
demanda está fuerte, así que precio y cantidad se determinan a la vez. Los
controles reducen el sesgo pero no lo eliminan.

Se intentó corregir con variable instrumental estilo Hausman (el precio del
mismo segmento en el otro hotel, misma fecha). El 2SLS da **−0.021, IC 95%
[−0.179, +0.138]** — cruza el cero. Y sobre todo, **la primera etapa sale
negativa (−0.473)**: si el precio del otro hotel mueve la demanda de este en
contra, hay sustitución, y entonces el instrumento afecta la demanda por un
canal directo. **Se rompe la restricción de exclusión: el instrumento no es
válido.**

**Consecuencia:** la elasticidad entra como supuesto explícito, anclado al rango
publicado para hotelería (≈ −0.4 a −1.5), con análisis de sensibilidad
obligatorio en el dashboard. Obtenerla de verdad requiere un test A/B de tarifas
o datos de rate shopping.

Documentar un resultado negativo es parte del trabajo. Un número inventado con
pasos intermedios sigue siendo un número inventado.

### 3. El modelo tiene que ver el precio

En una versión anterior, el clasificador se llamaba siempre con una tarifa fija
de referencia mientras el optimizador movía el precio. Al ser constante respecto
al precio, **la salida del modelo se cancelaba al derivar** y la tarifa
recomendada quedaba determinada por completo por unas reglas escritas a mano.
Sustituir la probabilidad del modelo por cualquier constante daba la misma
recomendación, hasta el centavo.

Aquí el precio candidato entra al modelo en cada evaluación de la curva. La
diferencia se ve: dos solicitudes idénticas salvo el historial del huésped
reciben tarifas e ingresos esperados distintos.

| Perfil (60 hab. libres, ε = −1.2) | ADR óptimo | P(cancelación) | Ingreso esperado |
|---|---:|---:|---:|
| Solicita estacionamiento | $104.79 | 0.1% | **$231.87** |
| Huésped frecuente | $104.79 | 4.1% | $222.55 |
| Corporativo, 3 días de anticipación | $106.67 | 14.7% | $197.85 |
| OTA, 45 días de anticipación | $95.39 | 38.5% | $140.65 |
| Con 2 cancelaciones previas | $72.82 | 98.6% | **$2.67** |

La última fila es la que un motor sin capa de cancelación no puede ver: esa
reserva casi no vale nada, y aceptarla desplazando inventario es destruir
ingreso.

---

## Dónde se fuga el ingreso

Antes del motor, la pregunta comercial: de todo lo que se reserva, ¿cuánto entra
a caja?

```
Valor bruto reservado   $42.7 M
Ingreso retenido        $29.6 M
Fuga por cancelación    $13.1 M   (30.7%)
```

**De cada peso reservado se pierden 31 centavos.** Esa es la magnitud que
justifica modelar el riesgo en lugar de ignorarlo.

Valor por solicitud recibida, por segmento:

| Segmento | Reservas | ADR | Cancelación | No reemb. | Bruto/solicitud | **Neto/solicitud** | Fuga |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct | 12,361 | $107.00 | 15.4% | 0% | $411.88 | **$331.79** | 19.4% |
| Offline TA/TO | 23,884 | $86.56 | 34.6% | 21% | $341.08 | **$294.03** | 13.8% |
| Online TA | 56,089 | $110.00 | 36.9% | 0% | $426.73 | **$244.61** | **42.7%** |
| Groups | 19,557 | $70.53 | 61.7% | 47% | $238.77 | **$205.77** | 13.8% |
| Corporate | 5,211 | $65.00 | 18.9% | 6% | $148.58 | **$123.25** | 17.1% |

Tres lecturas que cambian decisiones:

1. **El segmento que más cancela no es el que más dinero pierde.** Groups cancela
   el 61.7% pero solo fuga el 13.8% del valor, porque casi la mitad de sus
   reservas son no reembolsables. Online TA cancela el 36.9% y fuga el **42.7%**,
   porque prácticamente ninguna lo es.
2. **Por eso la palanca no es "reducir cancelaciones" sino mover la mezcla hacia
   tarifas que retengan ingreso.** Es una decisión de política tarifaria, no de
   operación.
3. **El ranking de canales se invierte.** Por ADR, Online TA parece el mejor
   canal ($110 vs $107 de Direct). Por valor **neto** por solicitud, Direct lo
   supera en **36%**. Cualquier decisión de mezcla tomada sobre ADR va en la
   dirección equivocada.

Y el riesgo crece de forma monótona con la anticipación —9.7% dentro de 7 días,
55.9% a más de 180— que es justo el gradiente que justifica proteger inventario
para la demanda tardía.

---

## Métricas del modelo de cancelación

Partición **temporal por fecha de reserva** (llegada − lead time), no aleatoria:
se entrena con lo reservado antes del 1 de marzo de 2017 y se evalúa con lo
posterior. Es lo que el modelo habría visto en producción.

| Métrica | Valor | Referencia |
|---|---:|---|
| ROC AUC | **0.8375** | — |
| PR AUC | 0.6270 | tasa base 0.282 |
| Brier | **0.1465** | 0.2025 (predecir la tasa base) |
| Log loss | 0.4393 | 0.5950 |
| Reservas de prueba | 12,533 | out-of-time |

Se excluyen las columnas que solo existen después del desenlace
(`reservation_status`, `reservation_status_date`, `assigned_room_type`).
Incluirlas da un AUC cercano a 0.99 que se derrumba en producción.

**Limitación conocida y no escondida:** el modelo sobreestima en el decil más
alto (predice 0.81, se observa 0.69). No es del modelo: la tasa base de
cancelación cayó de 38.5% a 28.2% entre ambos periodos. Se probó recalibración
isotónica sobre validación (no mejora) y sobre los primeros 30 días del periodo
de prueba simulando producción (empeora: Brier 0.1482 → 0.1509, 3,393
observaciones no alcanzan). **Se publican las probabilidades sin calibrar, que
son las mejor calibradas de las tres opciones medidas.** En producción se
recalibraría con ventana móvil de varios meses.

---

## Cómo correrlo

```bash
pip install -r requirements.txt
streamlit run app.py
```

El dashboard funciona directo: los modelos entrenados están en `models/`.

Para reentrenar desde cero:

```bash
python scripts/download_data.py     # baja el CSV (~17 MB, no se versiona)
python scripts/run_pipeline.py      # modelo, calibración de demanda, elasticidad
```

---

## Estructura

```
├── app.py                        dashboard Streamlit
├── src/
│   ├── data.py                   carga, limpieza, partición temporal, features
│   ├── train_cancellation.py     modelo de cancelación + métricas + calibración
│   ├── demand.py                 capacidad, escalera tarifaria, demanda por clase
│   ├── estimate_elasticity.py    el intento de estimar elasticidad y por qué falla
│   ├── insights.py               análisis de fuga de ingreso por segmento y canal
│   └── optimizer.py              EMSR-b, curva de demanda, sobreventa, optimización
├── models/                       artefactos entrenados (versionados)
├── reports/                      métricas y análisis en JSON
└── scripts/                      descarga de datos y pipeline completo
```

---

## Qué le falta para producción

Honestamente, y en orden de importancia:

1. **Elasticidad medida, no asumida.** Un test A/B de tarifas por segmento, o
   datos de rate shopping de la competencia. Es lo primero que haría.
2. **Optimización a nivel de estancia, no de noche.** EMSR-b controla una noche
   a la vez. Una reserva de 3 noches consume inventario de 3 fechas con
   demandas distintas; lo correcto es *network revenue management* con bid
   prices por fecha (programación lineal, descomposición determinística).
3. **Curvas de reserva (*booking curves*).** El pronóstico de demanda por clase
   aquí es estático. En producción se actualiza conforme entra la demanda, por
   días restantes a la llegada.
4. **Costo real del *walk*.** El límite de sobreventa usa una tolerancia de
   probabilidad; lo correcto es el costo esperado de reubicar al huésped contra
   el ingreso de la habitación extra.
5. **Monitoreo de deriva.** La tasa de cancelación se movió 10 puntos entre 2016
   y 2017. Sin recalibración periódica esto se degrada solo.

---

## Datos

Antonio, N., de Almeida, A., Nunes, L. (2019). *Hotel booking demand datasets.*
**Data in Brief**, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126

Dos hoteles portugueses (uno urbano, uno vacacional), llegadas entre julio de
2015 y agosto de 2017. Datos reales, anonimizados por los autores.

> Proyecto de demostración técnica. Las cifras describen los hoteles del
> dataset; no representan a ninguna cadena hotelera en particular.
