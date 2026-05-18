# 🌱 EcoGrid AI — Dashboard

Dashboard de predicción y clasificación del estado de la red eléctrica española.
**Proyecto Final de Máster — CEI · Telmo Rodríguez Gastañaga · 2026**

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Cloud (gratuito)

1. Sube esta carpeta a un repositorio GitHub (público o privado).
2. Ve a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con GitHub.
3. Haz clic en **"New app"** → selecciona tu repositorio → archivo: `app.py`.
4. Haz clic en **Deploy**. En ~2 minutos tendrás una URL pública.

## Para conectar los modelos reales (opcional)

En Google Colab, después de entrenar, añade esta celda:

```python
import joblib
joblib.dump(modelo_xgb_regressor, 'modelo_regresion.pkl')
joblib.dump(modelo_xgb_classifier, 'modelo_clasificacion.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

Luego descarga los `.pkl` y colócalos en esta misma carpeta.
El dashboard detectará automáticamente si existen y cargará los modelos reales.

## Estructura del proyecto

```
ecogrid_dashboard/
├── app.py              ← Dashboard principal
├── requirements.txt    ← Dependencias Python
└── README.md           ← Este archivo
```

## Métricas del sistema (validación enero 2026)

| Modelo | Métrica | Valor |
|--------|---------|-------|
| XGBoost Regressor | R² | 0.8925 (test) / 0.7199 (ene-26) |
| XGBoost Regressor | MAE | 2.453 GWh |
| XGBoost Classifier | F1-Crítico | 0.804 |
| XGBoost Classifier | Recall-Crítico | 94.5% |
