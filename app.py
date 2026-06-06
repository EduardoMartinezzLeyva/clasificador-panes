import os
import numpy as np
import cv2
import joblib
from flask import Flask, request, jsonify, render_template_string

# ── Configuración igual que el script original ────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELOS = os.path.join(BASE_DIR, "modelos_panes")
TAMANO_IMG   = (64, 64)
USAR_COLOR   = False   # mismo valor que en el script original

# ── Carga de modelos ──────────────────────────────────────────────────────────
print("Cargando modelos...")
modelo_lineal = joblib.load(os.path.join(RUTA_MODELOS, "svm_lineal.pkl"))
modelo_rbf    = joblib.load(os.path.join(RUTA_MODELOS, "svm_rbf.pkl"))
scaler        = joblib.load(os.path.join(RUTA_MODELOS, "scaler.pkl"))
pca           = joblib.load(os.path.join(RUTA_MODELOS, "pca.pkl"))
print("Modelos cargados OK")

app = Flask(__name__)

# ── ClasificadorPanes: COPIA EXACTA del script original ──────────────────────
class ClasificadorPanes:
    def __init__(self, modelo_lineal, modelo_rbf, scaler, pca, usar_color=False):
        self.modelo_lineal = modelo_lineal
        self.modelo_rbf    = modelo_rbf
        self.scaler        = scaler
        self.pca           = pca
        self.usar_color    = usar_color
        self.tamano_img    = TAMANO_IMG

    def preprocesar_imagen_bytes(self, img_bytes):
        """Igual que preprocesar_imagen pero recibe bytes en vez de ruta."""
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("No se pudo decodificar la imagen")
        img = cv2.resize(img, self.tamano_img)
        if not self.usar_color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.0
        return img.flatten()

    def clasificar_bytes(self, img_bytes):
        img_vector = self.preprocesar_imagen_bytes(img_bytes)
        img_pca    = self.pca.transform([img_vector])
        img_norm   = self.scaler.transform(img_pca)
        clase_lineal = self.modelo_lineal.predict(img_norm)[0]
        clase_rbf    = self.modelo_rbf.predict(img_norm)[0]
        return {'lineal': int(clase_lineal), 'rbf': int(clase_rbf)}

    def determinar_fondo(self, img_bytes):
        """Igual que determinar_fondo de InterfazPanes."""
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return "Desconocido"
        h, w = img.shape[:2]
        esquinas = [
            img[0:20,    0:20   ],
            img[0:20,    w-20:w ],
            img[h-20:h,  0:20   ],
            img[h-20:h,  w-20:w ],
        ]
        brillos = []
        for esq in esquinas:
            gray = cv2.cvtColor(esq, cv2.COLOR_BGR2GRAY)
            brillos.append(np.mean(gray))
        brillo_promedio = np.mean(brillos)
        return "Fondo Blanco" if brillo_promedio > 200 else "Fondo Color"

clasificador = ClasificadorPanes(modelo_lineal, modelo_rbf, scaler, pca, USAR_COLOR)

# ── HTML: misma lógica visual que la interfaz tkinter original ─────────────────
HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Clasificador de Panes</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', Arial, sans-serif;
      background: #f0f0f0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 0 0 40px 0;
    }

    /* ── Barra superior ── */
    .topbar {
      width: 100%;
      background: #fff;
      border-bottom: 2px solid #ddd;
      padding: 16px 20px 12px;
      text-align: center;
    }
    .topbar h1 {
      font-size: 1.45rem;
      color: #333;
      font-weight: 800;
    }
    .topbar p {
      color: #888;
      font-size: .82rem;
      margin-top: 2px;
    }

    /* ── Tarjeta principal ── */
    .card {
      background: #fff;
      border-radius: 12px;
      box-shadow: 0 2px 10px rgba(0,0,0,.12);
      margin: 20px 14px 0;
      width: calc(100% - 28px);
      max-width: 500px;
      overflow: hidden;
    }

    /* ── Zona de imagen ── */
    .img-zone {
      background: #fff;
      border: 2px solid #e0e0e0;
      border-radius: 8px;
      margin: 16px;
      min-height: 220px;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      overflow: hidden;
    }
    .img-zone img {
      max-width: 100%;
      max-height: 320px;
      display: none;
      object-fit: contain;
    }
    .img-placeholder {
      text-align: center;
      color: #bbb;
      padding: 30px;
    }
    .img-placeholder span { font-size: 2.5rem; display: block; }
    .img-placeholder p { font-size: .85rem; margin-top: 8px; }

    /* ── Botones principales (igual a los de tkinter: verde=Cargar, azul=Clasificar) ── */
    .btn-row {
      display: flex;
      gap: 12px;
      padding: 0 16px 16px;
    }
    /* input file oculto, activado por label */
    #file-input { display: none; }

    .btn {
      flex: 1;
      padding: 12px 8px;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      text-align: center;
      display: flex;
      align-items: center;
      justify-content: center;
      user-select: none;
      -webkit-user-select: none;
    }
    .btn-cargar    { background: #4CAF50; color: #fff; }
    .btn-cargar:active  { background: #388e3c; }
    .btn-clasificar { background: #2196F3; color: #fff; }
    .btn-clasificar:active { background: #1565c0; }
    .btn:disabled { opacity: .45; cursor: default; pointer-events: none; }

    /* ── Spinner ── */
    .spinner {
      display: none;
      width: 24px; height: 24px;
      border: 3px solid #2196F3;
      border-top-color: transparent;
      border-radius: 50%;
      animation: spin .7s linear infinite;
      margin: 0 auto 12px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Resultados: misma estructura que los LabelFrame de tkinter ── */
    .resultados {
      display: none;
      padding: 0 16px 16px;
    }
    .result-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .result-box {
      border: 2px solid;
      border-radius: 8px;
      padding: 14px 10px 12px;
      text-align: center;
    }
    .result-box.lineal { border-color: #2196F3; background: #f0f7ff; }
    .result-box.rbf    { border-color: #4CAF50; background: #f0faf0; }

    /* título del frame (como el LabelFrame de tk) */
    .result-box .box-title {
      font-size: .7rem;
      font-weight: 800;
      letter-spacing: 1.2px;
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    .result-box.lineal .box-title { color: #2196F3; }
    .result-box.rbf    .box-title { color: #4CAF50; }

    /* nombre del pan (grande y bold, igual al label_*_tipo) */
    .result-box .box-clase {
      font-size: 1.3rem;
      font-weight: 800;
    }
    .result-box.lineal .box-clase { color: #1a73e8; }
    .result-box.rbf    .box-clase { color: #2e7d32; }

    /* tipo de fondo (gris pequeño, igual al label_*_fondo) */
    .result-box .box-fondo {
      font-size: .82rem;
      color: #888;
      margin-top: 6px;
    }

    /* ── Error ── */
    .error-msg {
      display: none;
      margin: 0 16px 16px;
      background: #fdecea;
      color: #c0392b;
      border-radius: 8px;
      padding: 10px 14px;
      font-size: .88rem;
      text-align: center;
    }

    /* ── Barra de estado (igual al status_bar de tk) ── */
    .status-bar {
      width: calc(100% - 28px);
      max-width: 500px;
      margin-top: 10px;
      background: #ddd;
      border-radius: 6px;
      padding: 7px 14px;
      font-size: .8rem;
      color: #555;
    }
  </style>
</head>
<body>

  <div class="topbar">
    <h1>Clasificador de Panes</h1>
    <p>Concha vs Ojo · SVM Lineal &amp; RBF</p>
  </div>

  <div class="card">

    <!-- zona de imagen -->
    <div class="img-zone" id="img-zone">
      <div class="img-placeholder" id="placeholder">
        <span>🖼️</span>
        <p>Carga una imagen para clasificar</p>
      </div>
      <img id="preview" src="" alt="imagen cargada"/>
    </div>

    <!-- input file oculto -->
    <input type="file" id="file-input" accept="image/*"/>

    <!-- botones: Cargar Imagen (verde) | Clasificar (azul) -->
    <div class="btn-row">
      <label for="file-input" class="btn btn-cargar">
        Cargar Imagen
      </label>
      <button class="btn btn-clasificar" id="btn-clasificar" disabled>
        Clasificar
      </button>
    </div>

    <!-- spinner -->
    <div class="spinner" id="spinner"></div>

    <!-- resultados -->
    <div class="resultados" id="resultados">
      <div class="result-grid">
        <div class="result-box lineal">
          <div class="box-title">SVM Lineal</div>
          <div class="box-clase" id="res-lineal-clase">—</div>
          <div class="box-fondo" id="res-lineal-fondo"></div>
        </div>
        <div class="result-box rbf">
          <div class="box-title">SVM RBF</div>
          <div class="box-clase" id="res-rbf-clase">—</div>
          <div class="box-fondo" id="res-rbf-fondo"></div>
        </div>
      </div>
    </div>

    <!-- error -->
    <div class="error-msg" id="error-msg"></div>

  </div>

  <!-- barra de estado -->
  <div class="status-bar" id="status-bar">Listo. Cargue una imagen para clasificar</div>

<script>
  const fileInput      = document.getElementById('file-input');
  const preview        = document.getElementById('preview');
  const placeholder    = document.getElementById('placeholder');
  const btnClasif      = document.getElementById('btn-clasificar');
  const spinner        = document.getElementById('spinner');
  const resultados     = document.getElementById('resultados');
  const resLinealClase = document.getElementById('res-lineal-clase');
  const resLinealFondo = document.getElementById('res-lineal-fondo');
  const resRbfClase    = document.getElementById('res-rbf-clase');
  const resRbfFondo    = document.getElementById('res-rbf-fondo');
  const errorMsg       = document.getElementById('error-msg');
  const statusBar      = document.getElementById('status-bar');

  let selectedFile = null;

  // ── Cargar imagen ──
  fileInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;
    selectedFile = file;

    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.style.display = 'block';
    placeholder.style.display = 'none';

    btnClasif.disabled = false;
    resultados.style.display = 'none';
    errorMsg.style.display   = 'none';
    statusBar.textContent = `Imagen cargada: ${file.name}`;
  });

  // ── Clasificar ──
  btnClasif.addEventListener('click', async () => {
    if (!selectedFile) return;

    btnClasif.disabled      = true;
    spinner.style.display   = 'block';
    resultados.style.display = 'none';
    errorMsg.style.display  = 'none';
    statusBar.textContent   = 'Clasificando...';

    const formData = new FormData();
    formData.append('imagen', selectedFile);

    try {
      const resp = await fetch('/clasificar', { method: 'POST', body: formData });
      const data = await resp.json();
      if (data.error) throw new Error(data.error);

      resLinealClase.textContent = data.clase_lineal;
      resLinealFondo.textContent = '📸 ' + data.tipo_fondo;
      resRbfClase.textContent    = data.clase_rbf;
      resRbfFondo.textContent    = '📸 ' + data.tipo_fondo;

      resultados.style.display = 'block';
      statusBar.textContent =
        `Clasificación: Lineal=${data.clase_lineal}, RBF=${data.clase_rbf} | ${data.tipo_fondo}`;

    } catch(err) {
      errorMsg.textContent    = '⚠️ ' + err.message;
      errorMsg.style.display  = 'block';
      statusBar.textContent   = 'Error: ' + err.message;
    } finally {
      spinner.style.display = 'none';
      btnClasif.disabled    = false;
    }
  });
</script>
</body>
</html>
"""

# ── Rutas Flask ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/clasificar", methods=["POST"])
def clasificar_route():
    if "imagen" not in request.files:
        return jsonify({"error": "No se recibió imagen"}), 400
    file = request.files["imagen"]
    if file.filename == "":
        return jsonify({"error": "Archivo vacío"}), 400

    try:
        img_bytes = file.read()

        # Clasificar — exactamente igual que ClasificadorPanes.clasificar()
        resultados  = clasificador.clasificar_bytes(img_bytes)
        tipo_fondo  = clasificador.determinar_fondo(img_bytes)

        clase_lineal = "CONCHA" if resultados['lineal'] == 0 else "OJO"
        clase_rbf    = "CONCHA" if resultados['rbf']    == 0 else "OJO"

        return jsonify({
            "clase_lineal": clase_lineal,
            "clase_rbf":    clase_rbf,
            "tipo_fondo":   tipo_fondo,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Arranque ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import socket
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = "127.0.0.1"
    print("=" * 55)
    print("  CLASIFICADOR DE PANES - Servidor Web")
    print("=" * 55)
    print(f"  Local:  http://127.0.0.1:5000")
    print(f"  Movil:  http://{local_ip}:5000")
    print("  (Misma red WiFi)")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False)
