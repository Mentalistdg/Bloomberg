# Outline de slides — exposición 15-20 min, máximo 10 láminas

> Estructura sugerida. Cada slide: 1.5-2 min de exposición. Markdown listo
> para pegar en PowerPoint / Keynote / Google Slides.

---

## Slide 1 — Portada + mensaje principal

**Título:** Scoring de fondos mutuos — Renta Variable Indirecta
**Subtítulo (1 línea):** Modelo cuantitativo para informar la decisión
de inversión en un universo de 277 fondos USA, con énfasis en disciplina
anti-leakage y honestidad estadística.

**Datos clave de portada:**
- 250 fondos modelables · 120 meses · 9 folds walk-forward
- Modelo principal: ElasticNet · IC = +0.006, CI95 incluye cero
- Conclusión central: la señal predictiva es débil pero no lo es la
  disciplina del proceso — el modelo es honesto en su humildad.

---

## Slide 2 — Definición del problema

**Variable objetivo:** percentil cross-seccional del retorno total a 12
meses forward.

**Por qué este target:**
- Cross-seccional → robusto a regímenes (no se contamina con el beta del universo).
- 12 meses → alineado al holding period típico de un fondo de pensiones.
- Total return → incluye eventos de capital (`evento_pct`), no solo NAV.

**Enfoque: explicativo > predictivo.** Trade-off explícito:
- Audiencia del modelo (comité) entiende factores; interpretabilidad pesa.
- Tamaño efectivo del problema desfavorece black-boxes.
- Literatura (Carhart 1997, Fama-French 2010) reporta persistencia post-fee marginal.

→ ElasticNet primary, LightGBM como sanity check no-lineal.

---

## Slide 3 — Validación de datos contra el brief

> Antes de modelar, verificar que entendimos el dato.

| Item del brief | Validación realizada |
|---|---|
| "Eventos de capital" (`evento_pct`) | 99% positivos, cadencia mensual, suma anual 1-3% (yield consistent) → distribuciones de capital. Fórmula correcta de retorno: `precio.pct_change() + evento_pct`. |
| "Concentración del primer decil de subyacentes" | Inspección empírica muestra piso duro en 30%. Confirmado por descripción oficial: cantidad mínima de holdings para alcanzar el 30% del AUM. |
| "Fees" | 89% NULL en raw, varían en escalones por fondo, distribución bimodal (ETFs 0.03% vs activos 1.5%). Forward-fill por fondo + imputación cross-seccional cuando ausente. |

**Observación crítica:** plot demostrando el piso 30% en `pct_acum`.

---

## Slide 4 — Construcción de features (intuición financiera)

| Grupo | Features | Intuición |
|---|---|---|
| Momentum | `ret_1m`, `ret_3m`, `ret_6m`, `ret_12m` | Persistencia de retornos (Carhart 1997) |
| Riesgo | `vol_12m`, `max_dd_12m`, `sharpe_12m` | Penalización por riesgo asumido |
| Costo | `fee` + flag `fee_disponible` | Predictor robusto en literatura — fees altos predicen underperformance |
| Concentración | `log_n_instrumentos` + flag | Convicción vs. closet indexing |
| Cross-sectional ranks | de cada feature anterior | Robustez a regímenes |

**Diseño anti-leakage:** todas las rolling windows hacia atrás. El target
es la única columna que mira hacia adelante.

---

## Slide 5 — Esquema de validación: walk-forward + embargo

**Por qué embargo de 12 meses:** el target spans 12 meses forward, por lo
que sin embargo el target del set de entrenamiento se solapa con el de
validación.

**Layout temporal:**

```
[--- TRAIN (expanding) ---][embargo 12m][--- VAL (12m) ---]
```

**9 folds resultantes** cubriendo 2016-2024 como períodos completamente
fuera de muestra. Cada fold expande train en 12m e incluye nuevos datos.

→ Foto del diagrama temporal de los folds.

---

## Slide 6 — Resultados out-of-sample

| Modelo | IC mensual | IC IR | % meses IC>0 | Q5-Q1 spread | CI95 IC |
|---|---|---|---|---|---|
| **ElasticNet** | +0.006 | +0.02 | 43% | -0.26% | [-0.063, +0.076] |
| LightGBM | -0.045 | -0.20 | 40% | -1.55% | [-0.088, -0.002] |
| Benchmark naive | +0.031 | +0.09 | 61% | -0.26% | [-0.039, +0.096] |

**Diebold-Mariano (ElasticNet vs benchmark):** p > 0.65 → no se rechaza igualdad.

**Lectura honesta:**
- El IC del ElasticNet **no es estadísticamente distinguible de cero** al 95%.
- El benchmark naive (fee bajo + momentum 12m, sin entrenamiento) compite con el modelo entrenado.
- Resultado consistente con literatura post-fee.

---

## Slide 7 — Drivers del score (interpretabilidad)

→ Plot horizontal de coeficientes ElasticNet promedio entre folds.

**Signos consistentes con intuición financiera:**
- + `sharpe_12m`, + `ret_12m_rank`, + `ret_3m_rank`
- − `max_dd_12m`, − `vol_12m_rank`, − `log_n_instrumentos_rank`

**Magnitud:** chica. La regularización seleccionada por CV interna (α≈0.46,
l1_ratio≈0.23) es robusta, lo que es coherente con señal débil real más
que con un problema de configuración.

---

## Slide 8 — Backtest top-Q5 vs bot-Q1

→ Línea temporal del retorno realizado a 12m por quintil del score.

**Lectura:**
- Spread Q5-Q1 ligeramente negativo en promedio.
- 39% de los meses tienen spread positivo.
- Al excluir 2022 (régimen de alta inflación) el spread mejora levemente.

**Conclusión:** el modelo no genera valor tradeable de manera consistente
en este universo y período. Útil como **uno de varios inputs**, no como
selector único.

---

## Slide 9 — Limitaciones y extensiones

**Limitaciones reconocidas:**
1. Concentración con 47% de cobertura, fees con 12% pre-imputación.
2. Sesgo de supervivencia (fondos discontinuados invisibles).
3. Anonimización impide features de estilo / sector / región.
4. Tamaño efectivo del problema modesto.
5. Horizonte único (12m) sin análisis de sensibilidad.

**Con más tiempo o más datos haría:**
- Features macro y de régimen (VIX, term spread, dollar index).
- Holdings agregados (style box, sector, net flows) en lugar del proxy de concentración.
- Backtest con costos de turnover.
- Combinación con due diligence cualitativa — el score como **input dentro
  de un proceso multi-factor**, no como decisión única.

---

## Slide 10 — Uso de IA y reflexión metodológica

**Etapas con asistencia de Claude (Anthropic):**

1. **Validación del dataset** — particularmente la reconciliación de la
   descripción del campo `pct_acum` (brief decía "primer decil") con la
   firma empírica (piso duro en 30%). Hipótesis competidoras se testearon
   con queries SQL y datos reales de Bloomberg en vivo (cap-weight vs
   equal-weight S&P 500 para validar interpretaciones).
2. **Code review del walk-forward + embargo** — verificación de ausencia
   de leakage en el target a 12 meses.
3. **Estructuración del informe**.

**Validación de salidas del LLM:** todo código se ejecutó contra los
datos reales; conclusiones cualitativas se contrastaron contra literatura
(Carhart, Berk-Green, López de Prado, Diebold-Mariano).

**Reflexión personal:** el resultado más importante de este caso no es
el IC del modelo. Es la disciplina del proceso — definición, anti-leakage,
validación estadística, honestidad sobre los hallazgos. Un modelo que
reporta CI95 incluyendo cero es más útil para un comité de inversiones
que un modelo black-box afirmando precisión que no puede defender.
