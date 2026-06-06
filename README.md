# Clasificador de Panes 🍞

Web app para clasificar panes **Concha vs Ojo** usando SVM Lineal y RBF.

## Uso
1. Abre la app en el navegador
2. Toca **Cargar Imagen** y elige una foto de pan
3. Toca **Clasificar** — verás el resultado de ambos modelos

## Deploy en Render
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Python version: 3.11
