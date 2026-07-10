
let datos_cargados = null;
let predicciones_batch = [];
const CLASES = ["ILESO", "LESIONADO", "FALLECIDO"];
const COLORES_CLASES = {
    "ILESO": { icon: "🟢", color: "#10b981", bg: "#d1fae5" },
    "LESIONADO": { icon: "🟡", color: "#f59e0b", bg: "#fef3c7" },
    "FALLECIDO": { icon: "🔴", color: "#ef4444", bg: "#fee2e2" }
};


document.addEventListener('DOMContentLoaded', function () {
    console.log("✅ Página cargada");

    cargarInfoModelo();

    setupNavegacion();

    setupFormularioPrincipal();

    setupCargaArchivos();
});

function setupNavegacion() {
    const botones = document.querySelectorAll('.nav-btn');

    botones.forEach(btn => {
        btn.addEventListener('click', function () {
            const tabName = this.dataset.tab;

            botones.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));

            this.classList.add('active');
            document.getElementById(`tab-${tabName}`).classList.add('active');
        });
    });
}

function cargarInfoModelo() {
    fetch('/api/info')
        .then(response => response.json())
        .then(data => {
            console.log("📊 Info del modelo:", data);

            if (data.accuracy !== null) {
                document.getElementById('metric-accuracy').textContent =
                    (data.accuracy * 100).toFixed(2) + '%';
            }
            if (data.num_features) {
                document.getElementById('metric-features').textContent = data.num_features;
            }
        })
        .catch(error => console.error("Error cargando info:", error));
}


function setupFormularioPrincipal() {
    const form = document.getElementById('formulario-prediccion');

    form.addEventListener('submit', function (e) {
        e.preventDefault();
        realizarPrediccion();
    });
}

function realizarPrediccion() {
    console.log("🔮 Realizando predicción...");

    const form = document.getElementById('formulario-prediccion');
    const formData = new FormData(form);

    const datos = {
        "EDAD": parseFloat(formData.get('edad')),
        "TIPO PERSONA": formData.get('TIPO PERSONA'),
        "SEXO": formData.get('SEXO'),
        "POSEE LICENCIA": formData.get('POSEE LICENCIA'),
        "ESTADO LICENCIA": formData.get('ESTADO LICENCIA'),
        "CLASE_LICENCIA": formData.get('CLASE_LICENCIA'),
        "¿SE SOMETIÓ A DOSAJE ETÍLICO CUALITATIVO?": formData.get('¿SE SOMETIÓ A DOSAJE ETÍLICO CUALITATIVO?'),
        "RESULTADO DEL DOSAJE ETÍLICO CUALITATIVO": formData.get('RESULTADO DEL DOSAJE ETÍLICO CUALITATIVO'),
        "¿SE SOMETIÓ A DOSAJE ETÍLICO CUANTITATIVO?": formData.get('¿SE SOMETIÓ A DOSAJE ETÍLICO CUANTITATIVO?'),
        "VEHÍCULO": formData.get('VEHÍCULO'),
        "CLASE DE SINIESTRO": formData.get('CLASE DE SINIESTRO'),
        "CAUSA": formData.get('CAUSA'),
        "CAUSA ESPECIFICA": formData.get('CAUSA ESPECIFICA'),
        "TIPO DE VÍA": formData.get('TIPO DE VÍA'),
        "RED VIAL": formData.get('RED VIAL'),
        "DEPARTAMENTO": formData.get('DEPARTAMENTO'),
        "MES": formData.get('MES'),
        "DIA": formData.get('DIA'),
        "HORA": formData.get('HORA')
    };

    console.log("📤 Datos enviados:", datos);

    const btn = document.querySelector('#formulario-prediccion button[type="submit"]');
    const btnText = btn.innerHTML;
    btn.classList.add('btn-loading');
    btn.disabled = true;
    btn.innerHTML = '<span>⏳</span> Prediciendo...';

    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(datos)
    })
        .then(response => response.json())
        .then(result => {
            btn.classList.remove('btn-loading');
            btn.disabled = false;
            btn.innerHTML = btnText;

            if (result.success) {
                console.log("✅ Predicción exitosa:", result);
                mostrarResultadoPrediccion(result);
            } else {
                console.error("❌ Error en predicción:", result.error);
                mostrarError(result.error);
            }
        })
        .catch(error => {
            btn.classList.remove('btn-loading');
            btn.disabled = false;
            btn.innerHTML = btnText;
            console.error("❌ Error:", error);
            mostrarError(error.message);
        });
}

function mostrarResultadoPrediccion(result) {
    const clase = result.prediccion;
    const confianza = result.confianza;
    const probs = result.probabilidades;

    const resultados = document.getElementById('resultados');
    const errorMensaje = document.getElementById('error-mensaje');

    errorMensaje.style.display = 'none';

    const clasePred = document.getElementById('clase-predicida');
    clasePred.innerHTML = `
        <span class="icono-clase">${COLORES_CLASES[clase].icon}</span>
        <span class="nombre-clase">${clase}</span>
    `;

    document.getElementById('confianza-pct').textContent =
        (confianza * 100).toFixed(2) + '%';

    const probBars = document.getElementById('prob-bars');
    probBars.innerHTML = '';

    const probsArray = Object.entries(probs)
        .sort((a, b) => b[1] - a[1]);

    probsArray.forEach(([label, prob]) => {
        const labelClass = label.toLowerCase();
        const barHTML = `
            <div class="prob-bar">
                <div class="prob-label">${COLORES_CLASES[label].icon} ${label}</div>
                <div class="prob-background">
                    <div class="prob-fill ${labelClass}" style="width: ${prob * 100}%;">
                        ${(prob * 100).toFixed(1)}%
                    </div>
                </div>
            </div>
        `;
        probBars.innerHTML += barHTML;
    });

    setTimeout(() => {
        const fills = probBars.querySelectorAll('.prob-fill');
        fills.forEach(fill => {
            fill.style.width = fill.style.width;
        });
    }, 100);

    resultados.style.display = 'block';
    resultados.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function mostrarError(mensaje) {
    const errorDiv = document.getElementById('error-mensaje');
    errorDiv.textContent = mensaje;
    errorDiv.style.display = 'block';
    errorDiv.scrollIntoView({ behavior: 'smooth' });
}


function setupCargaArchivos() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');

    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            cargarArchivo(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            cargarArchivo(e.target.files[0]);
        }
    });

    const btnPredecir = document.getElementById('btn-predecir-batch');
    if (btnPredecir) {
        btnPredecir.addEventListener('click', predecirBatch);
    }

    
    const btnDescargar = document.getElementById('btn-descargar-csv');
    if (btnDescargar) {
        btnDescargar.addEventListener('click', descargarResultados);
    }
}

function cargarArchivo(file) {
    console.log("📁 Cargando archivo:", file.name);

    const reader = new FileReader();

    reader.onload = function (e) {
        const csv = e.target.result;
        const lineas = csv.split('\n');

        if (lineas.length < 2) {
            mostrarErrorBatch("El archivo está vacío");
            return;
        }

        
        const headers = lineas[0].split(',').map(h => h.trim());
        const registros = [];

        for (let i = 1; i < lineas.length; i++) {
            if (lineas[i].trim() === '') continue;

            const valores = lineas[i].split(',').map(v => v.trim());
            const registro = {};

            headers.forEach((header, idx) => {
                if (header === 'EDAD') {
                    registro[header] = parseFloat(valores[idx]) || 0;
                } else {
                    registro[header] = valores[idx] || '';
                }
            });

            registros.push(registro);
        }

        datos_cargados = registros;
        console.log("✅ Archivos cargado:", registros.length, "registros");

        
        mostrarPreview(registros);
    };

    reader.readAsText(file);
}

function mostrarPreview(registros) {
    const previewDiv = document.getElementById('batch-preview');
    const table = document.getElementById('preview-table');

    table.innerHTML = '';

    if (registros.length === 0) return;

    
    const headers = Object.keys(registros[0]);
    let html = '<thead><tr>';
    headers.forEach(header => {
        html += `<th>${header}</th>`;
    });
    html += '</tr></thead>';

    
    html += '<tbody>';
    for (let i = 0; i < Math.min(5, registros.length); i++) {
        html += '<tr>';
        headers.forEach(header => {
            const valor = registros[i][header];
            html += `<td>${valor}</td>`;
        });
        html += '</tr>';
    }
    html += '</tbody>';

    table.innerHTML = html;
    previewDiv.style.display = 'block';

    
    previewDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function predecirBatch() {
    console.log("🔮 Prediciendo lote...");

    if (!datos_cargados || datos_cargados.length === 0) {
        mostrarErrorBatch("No hay datos para predecir");
        return;
    }

    const btn = document.getElementById('btn-predecir-batch');
    const btnText = btn.innerHTML;
    btn.classList.add('btn-loading');
    btn.disabled = true;
    btn.innerHTML = '<span>⏳</span> Prediciendo...';

    
    fetch('/api/predict-batch', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ registros: datos_cargados })
    })
        .then(response => response.json())
        .then(result => {
            btn.classList.remove('btn-loading');
            btn.disabled = false;
            btn.innerHTML = btnText;

            if (result.success) {
                console.log("✅ Batch predicción exitosa:", result);
                predicciones_batch = result.resultados;
                mostrarResultadosBatch(result);
            } else {
                console.error("❌ Error en batch:", result.error);
                mostrarErrorBatch(result.error);
            }
        })
        .catch(error => {
            btn.classList.remove('btn-loading');
            btn.disabled = false;
            btn.innerHTML = btnText;
            console.error("❌ Error:", error);
            mostrarErrorBatch(error.message);
        });
}

function mostrarResultadosBatch(result) {
    const resultsDiv = document.getElementById('batch-results');
    const statsDiv = document.getElementById('batch-stats');
    const table = document.getElementById('results-table');

    
    const conteos = {
        'ILESO': 0,
        'LESIONADO': 0,
        'FALLECIDO': 0
    };

    result.resultados.forEach(r => {
        conteos[r.prediccion]++;
    });

    statsDiv.innerHTML = `
        <div class="stat-box">
            <div class="stat-number">${conteos['ILESO']}</div>
            <div class="stat-label">🟢 ILESO</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">${conteos['LESIONADO']}</div>
            <div class="stat-label">🟡 LESIONADO</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">${conteos['FALLECIDO']}</div>
            <div class="stat-label">🔴 FALLECIDO</div>
        </div>
    `;

    
    table.innerHTML = '<thead><tr><th>#</th><th>Predicción</th><th>Confianza</th></tr></thead>';

    let tbody = '<tbody>';
    result.resultados.forEach((r, idx) => {
        const icon = COLORES_CLASES[r.prediccion].icon;
        const confianza = (r.confianza * 100).toFixed(2);
        tbody += `
            <tr>
                <td>${idx + 1}</td>
                <td>${icon} ${r.prediccion}</td>
                <td>${confianza}%</td>
            </tr>
        `;
    });
    tbody += '</tbody>';

    table.innerHTML += tbody;

    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function mostrarErrorBatch(mensaje) {
    const errorDiv = document.getElementById('batch-error');
    errorDiv.textContent = mensaje;
    errorDiv.style.display = 'block';
    errorDiv.scrollIntoView({ behavior: 'smooth' });
}

function descargarResultados() {
    if (predicciones_batch.length === 0) {
        alert('No hay resultados para descargar');
        return;
    }

    
    let csv = 'Índice,Predicción,Confianza\n';

    predicciones_batch.forEach((result, idx) => {
        const confianza = (result.confianza * 100).toFixed(2);
        csv += `${idx + 1},"${result.prediccion}",${confianza}\n`;
    });

    
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.setAttribute('href', URL.createObjectURL(blob));
    link.setAttribute('download', 'predicciones_resultados.csv');
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    console.log("📥 Resultados descargados");
}

console.log("✅ Script cargado exitosamente");
