# 🌱 EcoGrid AI
### Sistema de ML para predicción de energías renovables en la red eléctrica española

> Pipeline end-to-end de Machine Learning que predice con 24h de antelación
> la producción renovable horaria y clasifica el estado operativo de la red eléctrica,
> validado con datos reales de enero 2026.

🚀 **[Ver demo en vivo →](https://ecogrid-ai.streamlit.app)**

![Dashboard EcoGrid AI](docs/screenshot-dashboard.png)

---

## 🎯 El problema

El sistema eléctrico español genera más del **50% de su electricidad con fuentes renovables**,
pero la naturaleza intermitente del sol y el viento obliga a Red Eléctrica de España (REE)
a mantener costosas reservas de generación convencional.

**Una mejora del 10% en la precisión de la previsión renovable puede suponer
ahorros de decenas de millones de euros anuales** y reducir emisiones evitables de CO₂.

## 📊 Resultados

| Modelo | Tarea | Test (2022-2025) | Validación real (enero 2026) |
|---|---|---|---|
| **XGBoost Regressor** | Predicción producción renovable (MW) | R² = 0.89 | R² = 0.72 |
| **XGBoost Classifier** | Clasificación estado red (3 clases) | F1 = 0.83 | F1 = 0.73 |

**🎯 El sistema detecta el 94% de las horas en estado CRÍTICO real** — la métrica más
relevante operativamente, ya que un falso negativo en estado crítico cuesta millones a un operador.

## 🏗️ Arquitectura

Variables exógenas (clima + mercado + calendario)
↓
XGBoost Regressor (Modelo 1)
→ predice total_renovable_mw
↓
XGBoost Classifier (Modelo 2)
→ clasifica ESTABLE / ALERTA / CRÍTICO

**Stacking explícito**: la predicción del Modelo 1 se usa como feature de entrada del Modelo 2,
mejorando la capacidad predictiva del clasificador.

## 🔬 Decisiones técnicas clave

Algunas de las decisiones más relevantes documentadas en la memoria del proyecto:

- **Detección y corrección de data leakage**: el baseline obtuvo R²=1.0 porque el target
  era suma directa de cuatro features. Diagnóstico y eliminación de `solar_mw`,
  `wind_onshore_mw`, `wind_offshore_mw`, `hydro_mw` y `cobertura_renovable` del set de features.

- **Arquitectura exógena pura**: el análisis SHAP reveló que las variables autorregresivas
  (lags) dominaban las predicciones, impidiendo la operación en horizonte futuro.
  Decisión consciente de **sacrificar R²=0.871 → 0.820** a cambio de independencia operativa total.

- **CAMS Copernicus sobre Open-Meteo**: detectamos que Open-Meteo reportaba 100% de nubosidad
  en horas donde la producción solar nacional superaba los 15.000 MW. Incorporamos radiación
  satelital directa CAMS para 8 ubicaciones, mejorando R² de 0.815 a 0.890 (+9.2%).

- **Precipitación AEMET como proxy del nivel de embalses**: en ausencia de datos directos
  de REE, integramos precipitación acumulada (7d y 30d) en cuencas hidrográficas.
  Mejora del modelo hidráulico: R² 0.576 → 0.755 (+31.3%).

- **8 ubicaciones meteorológicas estratégicas**: Madrid, Zaragoza (corredor Ebro),
  Sevilla (zona solar sur), A Coruña (eólica Galicia), Albacete, Burgos (Sierra Demanda),
  Cádiz (Tarifa) y Pamplona.

## 📡 Fuentes de datos

| Fuente | Tipo | Cobertura |
|---|---|---|
| **ENTSO-E API** | Generación y demanda eléctrica horaria | 2022–2026 · 129.922 registros |
| **Open-Meteo API** | Meteorología horaria | 8 ubicaciones · 2022–2026 |
| **CAMS Copernicus** | Radiación solar GHI satelital | 8 ubicaciones · 2022–2026 |
| **AEMET OpenData** | Precipitación en embalses | 8 estaciones · 2022–2026 |
| **Ember Energy** | Precio horario mercado OMIE | 2015–2026 |

**Dataset maestro final**: 35.029 registros horarios × 89 variables, 0 nulos.

## 🛠️ Stack tecnológico

- **Lenguaje**: Python 3.11
- **ML**: scikit-learn, XGBoost
- **Interpretabilidad**: SHAP
- **Datos**: pandas, NumPy
- **Visualización**: Plotly, Matplotlib, Seaborn
- **Frontend**: Streamlit
- **APIs**: entsoe-py, cdsapi, requests
- **Deploy**: Streamlit Community Cloud

## 📁 Estructura del repositorio
ecogrid-ai/
├── app.py                          # Dashboard Streamlit
├── EcoGrid_AI.ipynb                # Notebook con pipeline completo
├── Memoria_EcoGrid_AI.pdf          # Memoria del proyecto (40 pp)
├── requirements.txt                # Dependencias
├── datos/                          # CSVs Open-Meteo, CAMS, Spain.csv
└── docs/                           # Screenshots, gráficas, SHAP plots
## ▶️ Cómo reproducir los resultados

**Demo en vivo** (sin instalación): [ecogrid-ai.streamlit.app](https://ecogrid-ai.streamlit.app)

**Pipeline completo en local**:

1. Clonar repositorio:
```bash
   git clone https://github.com/telmorodri1995-creator/ecogrid-ai.git
   cd ecogrid-ai
```
2. Instalar dependencias: `pip install -r requirements.txt`
3. Obtener API keys (gratuitas):
   - ENTSO-E: `transparency@entsoe.eu`
   - AEMET: `opendata.aemet.es`
4. Ejecutar notebook: `EcoGrid_AI.ipynb`

## 📈 Limitaciones y próximos pasos

Decisiones técnicas pendientes documentadas honestamente:

- **Estimación del precio D+1**: el sistema usa precio real de Spain.csv;
  en producción requeriría un modelo intermedio que prediga el precio OMIE D+1.
- **Capacidad instalada por zona y temperatura de paneles**: factores no recogidos
  que explican la subestimación de picos solares en semanas 2-4 de la validación.
- **MLOps**: containerización con Docker, pipeline de reentrenamiento mensual
  con MLflow y monitorización de data drift para producción.

## 📚 Recursos del proyecto

- 📄 [Memoria completa (PDF)](./Memoria_EcoGrid_AI.pdf) · 40 páginas
- 📓 [Notebook de análisis](./EcoGrid_AI.ipynb)
- 🎬 [Demo dashboard](https://ecogrid-ai.streamlit.app)

## 👤 Autor

**Telmo Rodríguez Gastañaga**
Máster en Marketing Digital e Inteligencia Artificial · CEI · Madrid 2026
[LinkedIn](https://www.linkedin.com/in/telmo-rodriguez) · [GitHub](https://github.com/telmorodri1995-creator)

---

*Proyecto académico desarrollado en el módulo de IA del Máster en Marketing Digital e Inteligencia Artificial del CEI (Madrid, 2026). Calificación obtenida: 10/10.*
