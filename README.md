# 🌱 EcoGrid AI
### Gestión Inteligente de Energías Renovables

> Sistema de Machine Learning para predicción de producción renovable y clasificación del estado de la red eléctrica española.

🚀 **[Ver demo en vivo](https://ecogrid-ai.streamlit.app)**

---

## ¿Qué hace EcoGrid AI?

El sistema eléctrico español genera más del 50% de su electricidad con fuentes renovables, pero su naturaleza intermitente obliga a REE a mantener costosas reservas de potencia. EcoGrid AI predice con **24h de antelación** cuánta energía renovable producirá el sistema y en qué estado operativo se encontrará la red.

## Arquitectura

| | Modelo | Tarea | Resultado |
|---|---|---|---|
| M1 | XGBoost Regressor | Predice producción renovable (MW) | R²=0.89 · MAE=2.453 GWh |
| M2 | XGBoost Classifier | Clasifica estado de red (ESTABLE/ALERTA/CRÍTICO) | F1=0.80 · Recall-CRÍTICO=94.5% |

Los modelos están apilados: la predicción de M1 se usa como feature de entrada en M2.

## Fuentes de datos

- **ENTSO-E API** — Generación y demanda eléctrica horaria 2022–2026
- **Open-Meteo** — Meteorología horaria en 8 ubicaciones estratégicas
- **CAMS Copernicus** — Radiación solar GHI satelital
- **AEMET OpenData** — Precipitación en embalses
- **Spain.csv** — Precio eléctrico horario OMIE 2015–2026

## Stack tecnológico

`Python` · `XGBoost` · `Random Forest` · `
