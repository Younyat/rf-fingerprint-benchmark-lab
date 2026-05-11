const state = {
  datasets: [],
  modelCatalog: {},
  selected: null,
  reports: null,
  samples: [],
  benchmark: null,
  improvementPlan: null,
  activeJob: null,
  pollTimer: null,
};

const $ = (id) => document.getElementById(id);

function numberOrNull(id) {
  const value = $(id).value.trim();
  return value === "" ? null : Number(value);
}

function params() {
  return {
    dataset: $("datasetSelect").value,
    model_kind: $("modelKind").value,
    window_size: numberOrNull("windowSize") || 4096,
    window_strategy: $("windowStrategy").value,
    windows_per_file: numberOrNull("windowsPerFile") || 3,
    test_size: numberOrNull("testSize") || 0.25,
    max_files_per_class: numberOrNull("maxFilesPerClass"),
    max_data_gb: numberOrNull("maxDataGb"),
    max_data_percent: numberOrNull("maxDataPercent"),
    limit_files: numberOrNull("limitFiles"),
    prediction_limit: numberOrNull("predictionLimit") || 5,
    ieee_target: $("ieeeTarget").value,
    compare_mode: $("compareMode")?.value || "evaluate_existing",
    stability_compare: ($("compareMode")?.value || "evaluate_existing") !== "evaluate_existing",
  };
}

function selectedCompareModels() {
  const checked = [...document.querySelectorAll(".compareModelCheck:checked")].map((el) => el.value);
  return checked.length ? checked : [$("modelKind").value].filter(Boolean);
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Error de API");
  return data;
}

function fmt(value) {
  if (value === null || value === undefined) return "n/a";
  if (typeof value === "number") return Number(value).toFixed(4);
  return String(value);
}

function fmtBytes(bytes) {
  if (bytes === null || bytes === undefined) return "n/a";
  const n = Number(bytes || 0);
  if (!Number.isFinite(n) || n <= 0) return "0 MB";
  const gb = n / (1024 ** 3);
  if (gb >= 1) return `${gb.toFixed(4)} GB`;
  return `${(n / (1024 ** 2)).toFixed(2)} MB`;
}

function fmtModelSize(bytes) {
  if (bytes === null || bytes === undefined) return "pendiente";
  const n = Number(bytes || 0);
  if (n >= 1024 ** 2) return `${(n / (1024 ** 2)).toFixed(2)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(2)} KB`;
  return `${n} B`;
}

function pct(value) {
  const n = Number(value || 0);
  return `${Math.max(0, Math.min(100, n * 100)).toFixed(1)}%`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderDatasetMeta() {
  const dataset = state.datasets.find((d) => d.name === $("datasetSelect").value);
  state.selected = dataset;
  if (!dataset) return;
  $("datasetMeta").innerHTML = `
    <div class="kv"><strong>Modelo</strong><span>${escapeHtml(dataset.model_kind)}</span></div>
    <div class="kv"><strong>Dtype</strong><span>${escapeHtml(dataset.default_dtype)}</span></div>
    <div class="kv"><strong>Registros</strong><span>${fmt(dataset.inventory_records)}</span></div>
    <div class="kv"><strong>Peso dataset</strong><span>${fmt(dataset.size?.data_size_gb)} GB</span></div>
    <div class="kv"><strong>Min balanceado</strong><span>${fmt(dataset.size?.min_balanced_gb)} GB</span></div>
    <div class="kv"><strong>Archivos IQ</strong><span>${fmt(dataset.size?.data_files)}</span></div>
    <div class="kv"><strong>Accuracy</strong><span>${fmt(dataset.holdout_accuracy)}</span></div>
  `;
  const existing = (dataset.available_models || []).map((m) => m.model_kind);
  [...$("modelKind").options].forEach((opt) => {
    opt.textContent = existing.includes(opt.value) ? `${opt.value} (trained)` : opt.value;
  });
  updateCompareModelLabels(existing);
  $("ieeeTarget").disabled = dataset.name !== "ieee_cbrs";
  if (dataset.name === "uav_lightbridge" && !$("maxFilesPerClass").value && !$("maxDataGb").value) {
    $("maxFilesPerClass").placeholder = "recomendado: 100-300";
    $("maxDataGb").placeholder = "opcional; 0.5-1 GB";
  } else {
    $("maxFilesPerClass").placeholder = "sin limite";
    $("maxDataGb").placeholder = ">= min balanceado";
  }
  renderDataBudgetPreview();
}

function renderDataBudgetPreview() {
  const el = $("dataBudgetPreview");
  if (!el) return;
  const dataset = state.selected || state.datasets.find((d) => d.name === $("datasetSelect").value);
  if (!dataset?.size?.data_size_bytes) {
    el.textContent = "";
    return;
  }
  const totalBytes = Number(dataset.size.data_size_bytes || 0);
  const totalGb = Number(dataset.size.data_size_gb || 0);
  const minBalancedBytes = Number(dataset.size.min_balanced_bytes || 0);
  const gb = numberOrNull("maxDataGb");
  const percent = numberOrNull("maxDataPercent");
  if (gb !== null) {
    const bytes = gb * (1024 ** 3);
    const percentOfTotal = totalGb ? (gb / totalGb) * 100 : 0;
    el.textContent = `Presupuesto activo por GB: ${fmtBytes(bytes)} (${percentOfTotal.toFixed(2)}% de ${fmtBytes(totalBytes)}). Si pones GB y porcentaje, manda GB. Con presupuesto activo se ignoran Limite total y Max archivos por clase para no truncar el experimento.`;
    return;
  }
  if (percent !== null) {
    const bytes = totalBytes * (percent / 100);
    const warning = minBalancedBytes && bytes < minBalancedBytes
      ? ` Inferior al minimo balanceado aproximado (${fmtBytes(minBalancedBytes)}).`
      : "";
    el.textContent = `Presupuesto por porcentaje: ${percent.toFixed(2)}% = ${fmtBytes(bytes)} de ${fmtBytes(totalBytes)}.${warning} Con presupuesto activo se ignoran Limite total y Max archivos por clase para no truncar el experimento.`;
    return;
  }
  el.textContent = `Sin presupuesto: se pueden usar todos los archivos elegibles (${fmtBytes(totalBytes)}). Puedes poner GB exactos o porcentaje.`;
}

function updateCompareModelLabels(existing = []) {
  document.querySelectorAll(".compareModelItem").forEach((item) => {
    const value = item.dataset.model;
    const label = item.querySelector(".modelCheckName");
    if (label) label.textContent = existing.includes(value) ? `${value} (trained)` : value;
  });
}

function renderCompareModelSelector() {
  const target = $("compareModelList");
  if (!target) return;
  target.innerHTML = Object.entries(state.modelCatalog)
    .map(([key, meta]) => `
      <label class="compareModelItem" data-model="${escapeHtml(key)}">
        <input class="compareModelCheck" type="checkbox" value="${escapeHtml(key)}" checked />
        <span class="modelCheckName">${escapeHtml(key)}</span>
        <small>${escapeHtml(meta.family)}</small>
      </label>
    `)
    .join("");
}

function applyDatasetPreset() {
  const dataset = $("datasetSelect").value;
  if (dataset === "uav_lightbridge") {
    $("windowSize").value = 1024;
    $("windowStrategy").value = "linspace";
    $("windowsPerFile").value = 1;
    $("maxFilesPerClass").value = 100;
    $("maxDataGb").value = "";
    $("maxDataPercent").value = "";
    $("predictionLimit").value = 5;
  } else if (dataset === "kri_wifi") {
    $("windowSize").value = 32768;
    $("windowStrategy").value = "energy";
    $("windowsPerFile").value = 3;
    $("maxFilesPerClass").value = 2;
    $("maxDataGb").value = "";
    $("maxDataPercent").value = "";
    $("predictionLimit").value = 5;
  } else if (dataset === "ieee_cbrs") {
    $("windowSize").value = 4096;
    $("windowStrategy").value = "linspace";
    $("windowsPerFile").value = 1;
    $("maxFilesPerClass").value = 100;
    $("maxDataGb").value = "";
    $("maxDataPercent").value = "";
    $("ieeeTarget").value = "band";
    $("predictionLimit").value = 5;
  } else if (dataset === "wifi_dat_day") {
    $("windowSize").value = 32768;
    $("windowStrategy").value = "energy";
    $("windowsPerFile").value = 6;
    $("maxFilesPerClass").value = "";
    $("maxDataGb").value = "";
    $("maxDataPercent").value = "";
    $("predictionLimit").value = 6;
  }
  renderDataBudgetPreview();
}

async function loadDatasets(options = {}) {
  if (!Object.keys(state.modelCatalog).length) {
    const catalog = await api("/api/model-catalog");
    state.modelCatalog = catalog.models || {};
    $("modelKind").innerHTML = Object.entries(state.modelCatalog)
      .map(([key, meta]) => `<option value="${escapeHtml(key)}">${escapeHtml(key)} · ${escapeHtml(meta.family)}</option>`)
      .join("");
    renderCompareModelSelector();
    renderModelCatalog();
  }
  const data = await api("/api/datasets");
  state.datasets = data.datasets;
  $("datasetSelect").innerHTML = state.datasets
    .map((d) => `<option value="${escapeHtml(d.name)}">${escapeHtml(d.name)}</option>`)
    .join("");
  renderDatasetMeta();
  if (!options.keepParams) applyDatasetPreset();
  renderDataBudgetPreview();
  await loadReports();
  await loadSamples();
}

async function loadReports() {
  const dataset = $("datasetSelect").value;
  const modelKind = $("modelKind").value;
  state.reports = await api(`/api/reports?dataset=${encodeURIComponent(dataset)}&model_kind=${encodeURIComponent(modelKind)}`);
  renderInventory();
  renderMetrics();
  renderRanking(state.reports?.benchmark?.report || null, { preserveTab: true });
}

function renderModelCatalog() {
  const rows = Object.entries(state.modelCatalog)
    .map(
      ([key, meta]) => `
      <tr>
        <td>${escapeHtml(key)}</td>
        <td>${escapeHtml(meta.family)}</td>
        <td>${escapeHtml(meta.paper_link)}</td>
        <td>${escapeHtml(meta.description)}</td>
      </tr>`
    )
    .join("");
  $("modelCatalogView").innerHTML = `
    <h2>Tecnicas IA disponibles</h2>
    <table>
      <thead><tr><th>Modelo</th><th>Familia</th><th>Base documental</th><th>Uso en este sistema</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

async function loadSamples() {
  const dataset = $("datasetSelect").value;
  const ieeeTarget = $("ieeeTarget").value;
  const data = await api(`/api/samples?dataset=${encodeURIComponent(dataset)}&ieee_target=${encodeURIComponent(ieeeTarget)}&max_files_per_class=6&limit=60`);
  state.samples = data.samples;
  $("sampleSelect").innerHTML = state.samples
    .map((s, idx) => `<option value="${idx}">${escapeHtml(s.label)} | ${escapeHtml(s.meta_path)}</option>`)
    .join("");
}

function renderInventory() {
  const inv = state.reports?.inventory?.report;
  if (!inv) {
    $("inventoryView").innerHTML = `<section class="panel"><h2>Inventario</h2><p>No hay inventario generado para este dataset.</p></section>`;
    return;
  }
  const classRows = Object.entries(inv.classes || {})
    .map(([label, count]) => `<tr><td>${escapeHtml(label)}</td><td>${count}</td></tr>`)
    .join("");
  const examples = (inv.examples || [])
    .map(
      (ex) => `
      <div class="sample">
        <strong>${escapeHtml(ex.label)}</strong>
        <span>${escapeHtml(ex.data)}</span>
        <span>dtype=${escapeHtml(ex.dtype)} samples=${fmt(ex.sample_count)}</span>
      </div>`
    )
    .join("");
  $("inventoryView").innerHTML = `
    <section class="panel">
      <h2>Inventario del dataset</h2>
      <div class="statsGrid">
        <div class="metric"><span>Registros</span><div class="value">${fmt(inv.records)}</div></div>
        <div class="metric"><span>Clases</span><div class="value">${Object.keys(inv.classes || {}).length}</div></div>
      </div>
      <h3>Distribucion de clases</h3>
      <table><thead><tr><th>Clase</th><th>Archivos</th></tr></thead><tbody>${classRows}</tbody></table>
    </section>
    <section class="panel">
      <h2>Muestras verificadas</h2>
      ${examples}
      <p>Reporte: ${escapeHtml(state.reports.inventory.path)}</p>
    </section>
  `;
}

function renderMetrics() {
  const validation = state.reports?.validation?.report;
  const trainedModels = state.reports?.models?.models || [];
  const history = state.reports?.history?.report;
  if (!validation) {
    const modelsTable = renderTrainedModelsTable(trainedModels);
    $("metricsView").innerHTML = `<section class="panel"><h2>Validacion</h2><p>No hay reporte de validacion generado para el modelo seleccionado.</p></section>${modelsTable}`;
    return;
  }
  const report = validation.classification_report || {};
  const classNames = validation.classes || [];
  const rows = classNames
    .map((label) => {
      const row = report[label] || {};
      return `<tr><td>${escapeHtml(label)}</td><td>${fmt(row.precision)}</td><td>${fmt(row.recall)}</td><td>${fmt(row["f1-score"])}</td><td>${fmt(row.support)}</td></tr>`;
    })
    .join("");
  const matrix = validation.confusion_matrix || [];
  const matrixRows = matrix
    .map((row, idx) => `<tr><th>${escapeHtml(classNames[idx] || idx)}</th>${row.map((v) => `<td>${v}</td>`).join("")}</tr>`)
    .join("");
  const matrixHead = `<tr><th>real \\ pred</th>${classNames.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr>`;
  const samples = (validation.sample_predictions || [])
    .map(
      (s) => `<tr><td>${escapeHtml(s.expected)}</td><td>${escapeHtml(s.predicted)}</td><td>${fmt(s.confidence)}</td><td>${escapeHtml(s.file)}</td></tr>`
    )
    .join("");
  $("metricsView").innerHTML = `
    <section class="panel">
      <h2>Metricas holdout</h2>
      <div class="statsGrid">
        <div class="metric"><span>Modelo IA</span><div class="value">${escapeHtml(validation.model_kind || "n/a")}</div></div>
        <div class="metric"><span>Accuracy</span><div class="value">${fmt(validation.holdout_accuracy)}</div></div>
        <div class="metric"><span>Balanced accuracy</span><div class="value">${fmt(validation.balanced_accuracy)}</div></div>
        <div class="metric"><span>Macro F1</span><div class="value">${fmt(validation.macro_f1)}</div></div>
        <div class="metric"><span>Peso modelo</span><div class="value">${fmtModelSize(validation.model_size_bytes)}</div></div>
        <div class="metric"><span>Ventanas</span><div class="value">${fmt(validation.windows)}</div></div>
        <div class="metric"><span>Registros usados</span><div class="value">${fmt(validation.records_used)}</div></div>
        <div class="metric"><span>Features</span><div class="value">${fmt(validation.features)}</div></div>
      </div>
      <p>Modelo: ${escapeHtml(validation.model_path)}</p>
    </section>
    <section class="panel">
      <h2>Reporte por clase</h2>
      <table><thead><tr><th>Clase</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>${rows}</tbody></table>
    </section>
    <section class="panel matrix">
      <h2>Matriz de confusion</h2>
      <table><thead>${matrixHead}</thead><tbody>${matrixRows}</tbody></table>
    </section>
    <section class="panel">
      <h2>Predicciones de control</h2>
      <table><thead><tr><th>Esperado</th><th>Predicho</th><th>Confianza</th><th>Archivo</th></tr></thead><tbody>${samples}</tbody></table>
    </section>
    ${renderModelHistory(history, validation)}
    ${renderTrainedModelsTable(trainedModels)}
  `;
}

function polylineChart(points, key, label, color = "#2563eb", options = {}) {
  const rows = (points || [])
    .map((p, idx) => ({
      x: Number(options.xKey ? p[options.xKey] : idx + 1),
      y: Number(p[key]),
    }))
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  const values = rows.map((p) => p.y);
  if (values.length < 2) {
    return `
      <div class="miniChart">
        <strong>${escapeHtml(label)}</strong>
        <p>No hay suficientes puntos para curva.</p>
        <span>Eje X: ${escapeHtml(options.xLabel || "ejecucion cronologica")}</span>
        <span>Eje Y: ${escapeHtml(options.yLabel || key)}</span>
      </div>`;
  }
  const width = 560;
  const height = 190;
  const padLeft = 62;
  const padRight = 18;
  const padTop = 22;
  const padBottom = 48;
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const min = options.fixed01 ? 0 : Math.min(0, rawMin);
  const max = options.fixed01 ? 1 : Math.max(rawMax, rawMin + 1e-9);
  const xMin = Math.min(...rows.map((p) => p.x));
  const xMax = Math.max(...rows.map((p) => p.x));
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;
  const coords = rows.map((p) => {
    const x = padLeft + ((p.x - xMin) / Math.max(1e-9, xMax - xMin)) * plotW;
    const y = padTop + (1 - ((p.y - min) / Math.max(1e-9, max - min))) * plotH;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const xLabel = options.xLabel || "ejecucion cronologica (#)";
  const yLabel = options.yLabel || key;
  const unit = options.unit || "";
  return `
    <div class="miniChart">
      <strong>${escapeHtml(label)}</strong>
      <div class="miniChartMeta">
        <span>X: ${escapeHtml(xLabel)}</span>
        <span>Y: ${escapeHtml(yLabel)}${unit ? ` (${escapeHtml(unit)})` : ""}</span>
      </div>
      <svg viewBox="0 0 ${width} ${height}" role="img">
        <line x1="${padLeft}" y1="${height - padBottom}" x2="${width - padRight}" y2="${height - padBottom}" />
        <line x1="${padLeft}" y1="${padTop}" x2="${padLeft}" y2="${height - padBottom}" />
        <text x="${padLeft}" y="${height - 16}" text-anchor="middle">${escapeHtml(String(xMin))}</text>
        <text x="${width - padRight}" y="${height - 16}" text-anchor="end">${escapeHtml(String(xMax))}</text>
        <text x="${width / 2}" y="${height - 2}" text-anchor="middle">${escapeHtml(xLabel)}</text>
        <text x="6" y="${padTop + 4}" text-anchor="start">${fmt(max)}</text>
        <text x="6" y="${height - padBottom + 4}" text-anchor="start">${fmt(min)}</text>
        <text x="12" y="${height / 2}" transform="rotate(-90 12 ${height / 2})" text-anchor="middle">${escapeHtml(yLabel)}</text>
        <polyline points="${coords}" fill="none" stroke="${color}" stroke-width="3" />
        ${coords.split(" ").map((c) => {
          const [x, y] = c.split(",");
          return `<circle cx="${x}" cy="${y}" r="3.5" fill="${color}" />`;
        }).join("")}
      </svg>
      <span>n=${values.length} | ultimo=${fmt(values[values.length - 1])}${unit ? ` ${escapeHtml(unit)}` : ""} | mejor=${fmt(Math.max(...values))}${unit ? ` ${escapeHtml(unit)}` : ""} | rango=[${fmt(min)}, ${fmt(max)}]</span>
    </div>
  `;
}

function renderLossCurve(trainingHistory) {
  const rows = trainingHistory || [];
  if (!rows.length) return "";
  return polylineChart(rows, "loss", "Curva de perdida por epoca (deep learning)", "#dc2626", {
    xKey: "epoch",
    xLabel: "epoca de entrenamiento",
    yLabel: "loss de entrenamiento",
    unit: "cross-entropy",
  });
}

function renderModelHistory(history, validation) {
  const runs = history?.runs || [];
  const recent = runs.slice(-12);
  const lastDeepHistory = validation.training_history || runs[runs.length - 1]?.training_history || [];
  const runRows = recent.slice().reverse().map((run, idx) => `
    <tr>
      <td>${recent.length - idx}</td>
      <td>${escapeHtml(run.recorded_at || "")}</td>
      <td>${fmt(run.holdout_accuracy)}</td>
      <td>${fmt(run.balanced_accuracy)}</td>
      <td>${fmt(run.macro_f1)}</td>
      <td>${fmt(run.windows)}</td>
      <td>${fmt(run.estimated_iq_gb_read)} GB</td>
      <td>${fmtModelSize(run.model_size_bytes)}</td>
    </tr>
  `).join("");
  return `
    <section class="panel">
      <h2>Historial de aprendizaje del modelo</h2>
      <p>Curvas generadas desde entrenamientos/reentrenamientos guardados para este dataset y tecnica. Permiten ver si el modelo mejora, se estanca o sobreajusta entre ejecuciones.</p>
      <div class="chartGrid historyCharts">
        ${polylineChart(recent, "macro_f1", "Macro-F1 por ejecucion", "#16a34a", {
          xLabel: "ejecucion cronologica guardada (#)",
          yLabel: "Macro-F1 holdout",
          unit: "0-1",
          fixed01: true,
        })}
        ${polylineChart(recent, "holdout_accuracy", "Accuracy holdout por ejecucion", "#2563eb", {
          xLabel: "ejecucion cronologica guardada (#)",
          yLabel: "accuracy holdout",
          unit: "0-1",
          fixed01: true,
        })}
        ${polylineChart(recent, "balanced_accuracy", "Balanced accuracy por ejecucion", "#7c3aed", {
          xLabel: "ejecucion cronologica guardada (#)",
          yLabel: "balanced accuracy holdout",
          unit: "0-1",
          fixed01: true,
        })}
        ${renderLossCurve(lastDeepHistory)}
      </div>
      <table>
        <thead><tr><th>#</th><th>Fecha</th><th>Accuracy</th><th>Balanced</th><th>Macro-F1</th><th>Ventanas</th><th>GB leidos</th><th>Peso</th></tr></thead>
        <tbody>${runRows || `<tr><td colspan="8">Aun no hay historial persistente. Entrena o reentrena este modelo para crear la primera entrada.</td></tr>`}</tbody>
      </table>
      <p>Historial: ${escapeHtml(state.reports?.history?.path || "n/a")}</p>
    </section>
  `;
}

function renderTrainedModelsTable(models) {
  const rows = (models || [])
    .map(
      (m) => `
      <tr>
        <td>${escapeHtml(m.model_kind)}</td>
        <td>${m.model_exists ? "entrenado" : "pendiente"}</td>
        <td>${fmtModelSize(m.model_size_bytes)}</td>
        <td>${fmt(m.holdout_accuracy)}</td>
        <td>${fmt(m.balanced_accuracy)}</td>
        <td>${fmt(m.macro_f1)}</td>
        <td>${fmt(m.records_used)}</td>
        <td>${fmt(m.estimated_iq_gb_read)} GB</td>
        <td>${fmt((m.referenced_data_size_bytes || m.data_size_bytes || 0) / (1024 ** 3))} GB</td>
        <td>${fmt(m.windows)}</td>
        <td>${escapeHtml(m.model_path)}</td>
      </tr>`
    )
    .join("");
  return `
    <section class="panel matrix">
      <h2>Serie de modelos por dataset</h2>
      <p>Cada fila usa archivo propio por dataset y tecnica: <code>models/&lt;dataset&gt;_&lt;modelo&gt;_model.joblib</code>. Cambiar de dataset no sobreescribe los modelos de otro dataset.</p>
      <table>
        <thead><tr><th>Modelo</th><th>Estado</th><th>Peso modelo</th><th>Accuracy</th><th>Balanced acc</th><th>Macro-F1</th><th>Registros</th><th>GB leidos est.</th><th>GB referenciados</th><th>Ventanas</th><th>Archivo</th></tr></thead>
        <tbody>${rows || `<tr><td colspan="11">No hay catalogo de modelos para este dataset.</td></tr>`}</tbody>
      </table>
    </section>
  `;
}

function buildImprovementPlan(result, best) {
  const changes = {};
  const reasons = [];
  const current = params();
  const metricSet = best.retrain || best.evaluation || {};
  const macroF1 = Number(metricSet.macro_f1 || 0);
  const predAcc = Number(best.prediction_accuracy || 0);
  const stabilityGap = Number(best.stability_gap_macro_f1 || 0);
  const dataset = result.dataset;

  if (macroF1 < 0.5) {
    changes.windows_per_file = Math.max(Number(current.windows_per_file || 3), 6);
    changes.prediction_limit = Math.max(Number(current.prediction_limit || 5), 20);
    reasons.push("Aumentar ventanas por archivo y predicciones de control porque Macro-F1 es bajo.");
  }

  if (dataset === "kri_wifi" && macroF1 < 0.3) {
    changes.window_strategy = "energy";
    changes.window_size = Math.max(Number(current.window_size || 4096), 65536);
    changes.windows_per_file = Math.max(Number(changes.windows_per_file || current.windows_per_file || 3), 6);
    reasons.push("KRI WiFi necesita ventanas de mayor longitud y seleccion por energia para reducir ventanas poco informativas.");
  }

  if (dataset === "uav_lightbridge" && macroF1 < 0.5) {
    changes.window_strategy = "linspace";
    changes.window_size = Math.max(Number(current.window_size || 1024), 2048);
    changes.windows_per_file = Math.max(Number(changes.windows_per_file || current.windows_per_file || 1), 2);
    reasons.push("UAV Lightbridge suele ser costoso por muchos bursts; se priorizan ventanas uniformes y pocas ventanas mas largas.");
  }

  if (predAcc < 0.4) {
    changes.prediction_limit = Math.max(Number(changes.prediction_limit || current.prediction_limit || 5), 20);
    reasons.push("Aumentar predicciones de control para que el conteo de aciertos/fallos no dependa de 3-5 muestras.");
  }

  if (stabilityGap > 0.1) {
    changes.test_size = Math.max(Number(current.test_size || 0.25), 0.3);
    reasons.push("Aumentar holdout para medir estabilidad con una particion de prueba mas exigente.");
  }

  const budget = result.data_budget || {};
  const requestedPercent = Number(budget.requested_max_data_percent || current.max_data_percent || 0);
  const requestedGb = Number(budget.requested_max_data_gb || current.max_data_gb || 0);
  if (!requestedPercent && !requestedGb && dataset === "kri_wifi") {
    changes.max_data_percent = 10;
    reasons.push("Fijar presupuesto explicito del 10% para que el experimento sea reproducible y no dependa de presets.");
  }

  if (Object.keys(changes).length === 0) return null;
  return { changes, reasons };
}

function setFieldIfPresent(id, value) {
  if (value === undefined || value === null) return;
  const el = $(id);
  if (!el) return;
  el.value = value;
}

function applyImprovementPlan() {
  const plan = state.improvementPlan;
  if (!plan) return;
  const c = plan.changes || {};
  setFieldIfPresent("windowSize", c.window_size);
  setFieldIfPresent("windowStrategy", c.window_strategy);
  setFieldIfPresent("windowsPerFile", c.windows_per_file);
  setFieldIfPresent("testSize", c.test_size);
  setFieldIfPresent("predictionLimit", c.prediction_limit);
  setFieldIfPresent("maxDataPercent", c.max_data_percent);
  setFieldIfPresent("maxDataGb", c.max_data_gb);
  renderDataBudgetPreview();
}

function renderImprovementPlan(plan) {
  if (!plan) return "";
  const rows = Object.entries(plan.changes || {})
    .map(([key, value]) => `<tr><td>${escapeHtml(key)}</td><td>${escapeHtml(value)}</td></tr>`)
    .join("");
  const reasons = (plan.reasons || []).map((reason) => `<p>${escapeHtml(reason)}</p>`).join("");
  return `
    <div class="recommendationBox">
      <strong>Medidas recomendadas para el siguiente experimento</strong>
      ${reasons}
      <table><thead><tr><th>Parametro</th><th>Nuevo valor</th></tr></thead><tbody>${rows}</tbody></table>
      <div class="recommendationActions">
        <button id="applyImprovementBtn" type="button">Adoptar mejoras en parametros</button>
        <button id="applyAndCompareBtn" type="button" class="secondary">Adoptar y comparar otra vez</button>
      </div>
    </div>
  `;
}

function renderRanking(result, options = {}) {
  if (!result || !result.results) {
    $("rankingView").innerHTML = `<section class="panel"><h2>Comparacion cientifica</h2><p>Ejecuta "Comparar tecnicas IA" para evaluar modelos ya entrenados. Cambia "Modo de comparacion" solo si quieres lanzar reentrenamiento o train from scratch experimental sin pisar el modelo actual.</p></section>`;
    return;
  }
  state.benchmark = result;
  const rows = result.results
    .map((row) => {
      const train = row.train || {};
      const retrain = row.retrain || {};
      const evaluation = row.evaluation || {};
      const metricSet = row.retrain || row.evaluation || {};
      return `
        <tr>
          <td>${row.rank ?? "-"}</td>
          <td>${escapeHtml(row.model_kind)}</td>
          <td>${escapeHtml(row.model_family || row.family)}</td>
          <td>${escapeHtml(row.model_action || row.training_action || row.status || "n/a")}</td>
          <td>${escapeHtml(metricSet.model_version || row.model_version || "n/a")}</td>
          <td>${escapeHtml(metricSet.seed || row.seed || "n/a")}</td>
          <td>${escapeHtml(metricSet.trained_at || row.trained_at || "n/a")}</td>
          <td>${fmt(row.score)}</td>
          <td>${fmt(row.requested_max_data_percent)}</td>
          <td>${fmtBytes(row.requested_max_data_bytes)}</td>
          <td>${fmt(row.estimated_iq_gb_read)} GB</td>
          <td>${fmt((row.referenced_data_size_bytes || row.data_size_bytes || 0) / (1024 ** 3))} GB</td>
          <td>${fmt(row.referenced_percent_of_total)}%</td>
          <td>${row.prediction_hits}/${row.prediction_count}</td>
          <td>${row.prediction_failures}</td>
          <td>${fmt(row.prediction_accuracy)}</td>
          <td>${fmt(train.macro_f1 || evaluation.macro_f1)}</td>
          <td>${fmt(retrain.macro_f1)}</td>
          <td>${fmt(metricSet.balanced_accuracy)}</td>
          <td>${fmt(row.stability_gap_macro_f1)}</td>
          <td>${fmt(train.time_seconds)} s</td>
          <td>${fmt(retrain.time_seconds)} s</td>
          <td>${fmt(row.mean_prediction_time_seconds)} s</td>
          <td>${fmt(row.total_benchmark_time_seconds)} s</td>
        </tr>`;
    })
    .join("");
  const best = result.best_model || result.results[0] || {};
  const totalHits = result.results.reduce((acc, row) => acc + Number(row.prediction_hits || 0), 0);
  const totalFailures = result.results.reduce((acc, row) => acc + Number(row.prediction_failures || 0), 0);
  const totalPredictions = result.results.reduce((acc, row) => acc + Number(row.prediction_count || 0), 0);
  const totalTime = result.results.reduce((acc, row) => acc + Number(row.total_benchmark_time_seconds || 0), 0);
  const budget = result.data_budget || {};
  const predictionRows = (best.predictions || [])
    .map(
      (p) => `
      <tr>
        <td>${p.hit ? "OK" : "FAIL"}</td>
        <td>${escapeHtml(p.expected)}</td>
        <td>${escapeHtml(p.predicted)}</td>
        <td>${fmt(p.confidence)}</td>
        <td>${fmt(p.prediction_time_seconds)} s</td>
        <td>${escapeHtml(p.file)}</td>
      </tr>`
    )
    .join("");
  const protocolRows = (result.protocol?.steps || [])
    .map((step, idx) => `<tr><td>${idx + 1}</td><td>${escapeHtml(step)}</td></tr>`)
    .join("");
  const familyRows = (result.family_summary || [])
    .map(
      (row, idx) => `
      <tr>
        <td>${idx + 1}</td>
        <td>${escapeHtml(row.model_family)}</td>
        <td>${row.model_count}</td>
        <td>${escapeHtml(row.best_model)}</td>
        <td>${fmt(row.best_score)}</td>
        <td>${fmt(row.mean_score)}</td>
        <td>${fmt(row.mean_macro_f1_retrain)}</td>
        <td>${fmt(row.mean_balanced_accuracy_retrain)}</td>
        <td>${fmt(row.mean_prediction_accuracy)}</td>
        <td>${fmt(row.mean_total_time_seconds)} s</td>
        <td>${fmt(row.cumulative_total_time_seconds)} s</td>
      </tr>`
    )
    .join("");
  const maxTime = Math.max(...result.results.map((row) => Number(row.total_benchmark_time_seconds || 0)), 1);
  const chartRows = result.results
    .filter((row) => row.score !== null && row.score !== undefined)
    .map(
      (row) => {
        const metricSet = row.retrain || row.evaluation || {};
        return `
      <div class="chartRow">
        <strong>${row.rank}. ${escapeHtml(row.model_kind)}</strong>
        <span>${escapeHtml(row.model_family || "")} - ${escapeHtml(row.model_action || row.training_action || "")}</span>
        <div class="chartMetric"><span>Score ${fmt(row.score)}</span><div class="bar"><div style="width:${pct(row.score)}"></div></div></div>
        <div class="chartMetric"><span>Pred acc ${fmt(row.prediction_accuracy)} (${row.prediction_hits}/${row.prediction_count})</span><div class="bar okBar"><div style="width:${pct(row.prediction_accuracy)}"></div></div></div>
        <div class="chartMetric"><span>Macro-F1 ${fmt(metricSet.macro_f1)}</span><div class="bar f1Bar"><div style="width:${pct(metricSet.macro_f1)}"></div></div></div>
        <div class="chartMetric"><span>Tiempo ${fmt(row.total_benchmark_time_seconds)} s</span><div class="bar timeBar"><div style="width:${pct(Number(row.total_benchmark_time_seconds || 0) / maxTime)}"></div></div></div>
      </div>`;
      }
    )
    .join("");
  const warnings = [];
  const bestMetrics = best.retrain || best.evaluation || {};
  if (Number(bestMetrics.macro_f1 || 0) < 0.5) {
    warnings.push("El mejor modelo tiene Macro-F1 bajo. Esto indica que, aunque acierte algunas predicciones de control, no separa bien todas las clases en holdout.");
  }
  if (result.dataset === "kri_wifi" && Number(bestMetrics.macro_f1 || 0) < 0.3) {
    warnings.push("Para KRI WiFi, este resultado suele indicar ventanas poco informativas o señal/noise mezclados. Usa seleccion de ventanas por energia RF, ventana IQ de 32768 o 65536 y mas ventanas por archivo antes de repetir el benchmark.");
  }
  if (Number(best.prediction_accuracy || 0) - Number(bestMetrics.macro_f1 || 0) > 0.5) {
    warnings.push("Hay una diferencia grande entre prediccion de control y Macro-F1. Conviene aumentar 'Predicciones control' y revisar la matriz de confusion antes de concluir que el modelo identifica dispositivos de forma robusta.");
  }
  if (Number(best.stability_gap_macro_f1 || 0) > 0.1) {
    warnings.push("El gap entre train y retrain es alto. El resultado depende de la semilla o del split y necesita repetirse con mas semillas.");
  }
  if (result.family_comparison?.alert) {
    warnings.push(result.family_comparison.alert);
  }
  const improvementPlan = buildImprovementPlan(result, best);
  state.improvementPlan = improvementPlan;
  const warningBlock = warnings.length
    ? `<div class="scientificNote warnNote"><strong>Lectura critica</strong>${warnings.map((w) => `<p>${escapeHtml(w)}</p>`).join("")}${renderImprovementPlan(improvementPlan)}</div>`
    : `<div class="scientificNote okNote"><strong>Lectura critica</strong><p>No hay alertas fuertes con los umbrales actuales. Aun asi, valida con mas predicciones, sesiones y distancias.</p></div>`;
  $("rankingView").innerHTML = `
    <section class="panel">
      <h2>Comparacion cientifica de modelos</h2>
      <div class="statsGrid wideStats">
        <div class="metric"><span>Mejor tecnica</span><div class="value">${escapeHtml(best.model_kind || "n/a")}</div></div>
        <div class="metric"><span>Score</span><div class="value">${fmt(best.score)}</div></div>
        <div class="metric"><span>Aciertos control</span><div class="value">${totalHits}/${totalPredictions}</div></div>
        <div class="metric"><span>Fallos control</span><div class="value">${totalFailures}</div></div>
        <div class="metric"><span>Modelos evaluados</span><div class="value">${result.results.length}</div></div>
        <div class="metric"><span>Tiempo total</span><div class="value">${fmt(totalTime)} s</div></div>
        <div class="metric"><span>Presupuesto pedido</span><div class="value">${budget.requested_max_data_percent != null ? `${fmt(budget.requested_max_data_percent)}%` : fmtBytes(budget.requested_max_data_bytes)}</div></div>
        <div class="metric"><span>Total dataset</span><div class="value">${fmtBytes(budget.dataset_total_bytes)}</div></div>
      </div>
      <p>${escapeHtml(result.protocol?.ranking_score || "")}</p>
      <p>${escapeHtml(result.protocol?.data_budget || "")}</p>
      <p>Reporte: ${escapeHtml(result.benchmark_path || "")}</p>
      <p>Markdown: ${escapeHtml(result.benchmark_markdown_path || "")}</p>
    </section>
    ${warningBlock}
    <section class="panel">
      <h2>Graficas comparativas</h2>
      <div class="chartGrid">${chartRows}</div>
    </section>
    <section class="panel">
      <h2>Protocolo experimental</h2>
      <table>
        <thead><tr><th>#</th><th>Etapa ejecutada por cada modelo</th></tr></thead>
        <tbody>${protocolRows}</tbody>
      </table>
      <p>${escapeHtml(result.protocol?.split || "")}</p>
      <p>${escapeHtml(result.protocol?.prediction_control || "")}</p>
      <p>${escapeHtml(result.protocol?.timing || "")}</p>
    </section>
    <section class="panel matrix">
      <h2>Ranking por familia</h2>
      <table>
        <thead><tr><th>#</th><th>Familia</th><th>Modelos</th><th>Mejor modelo</th><th>Mejor score</th><th>Score medio</th><th>Macro-F1 medio</th><th>Balanced acc media</th><th>Pred acc media</th><th>Tiempo medio</th><th>Tiempo acumulado</th></tr></thead>
        <tbody>${familyRows || `<tr><td colspan="11">No hay familias evaluadas.</td></tr>`}</tbody>
      </table>
    </section>
    <section class="panel matrix">
      <h2>Tabla comparativa completa</h2>
      <table>
        <thead>
          <tr>
            <th>#</th><th>Modelo</th><th>Familia</th><th>Accion</th><th>Version</th><th>Seed</th><th>Fecha</th><th>Score</th><th>% pedido</th><th>Presupuesto</th><th>GB leidos est.</th><th>GB referenciados</th><th>% real ref.</th><th>Aciertos</th><th>Fallos</th><th>Pred acc</th>
            <th>Macro-F1 train</th><th>Macro-F1 retrain</th><th>Balanced acc retrain</th><th>Gap estabilidad</th>
            <th>Train</th><th>Retrain</th><th>Pred/media</th><th>Total</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </section>
    <section class="panel">
      <h2>Predicciones del mejor modelo</h2>
      <table>
        <thead><tr><th>Hit</th><th>Esperado</th><th>Predicho</th><th>Confianza</th><th>Tiempo</th><th>Archivo</th></tr></thead>
        <tbody>${predictionRows}</tbody>
      </table>
    </section>
  `;
  const applyBtn = $("applyImprovementBtn");
  const rerunBtn = $("applyAndCompareBtn");
  if (applyBtn) applyBtn.addEventListener("click", applyImprovementPlan);
  if (rerunBtn) {
    rerunBtn.addEventListener("click", () => {
      applyImprovementPlan();
      startOperation("compare", {
        ...params(),
        model_kinds: selectedCompareModels(),
        prediction_limit: numberOrNull("predictionLimit") || 5,
      });
    });
  }
  if (!options.preserveTab) showTab("ranking");
}

async function startOperation(operation, payload = params()) {
  const job = await api(`/api/${operation}`, { method: "POST", body: JSON.stringify(payload) });
  state.activeJob = job.id;
  showTab("jobs");
  pollJob(job.id);
}

async function pollJob(jobId) {
  clearTimeout(state.pollTimer);
  const job = await api(`/api/jobs/${jobId}`);
  await renderJobs();
  if (job.status === "completed") {
    await loadDatasets({ keepParams: true });
    if (job.operation === "predict") renderPrediction(job.result);
    if (job.operation === "compare") renderRanking(job.result);
    if (job.operation !== "predict" && job.operation !== "compare") showTab(job.operation === "summary" ? "jobs" : "metrics");
    return;
  }
  if (job.status === "failed") {
    showTab("jobs");
    return;
  }
  state.pollTimer = setTimeout(() => pollJob(jobId), 1500);
}

async function renderJobs() {
  const data = await api("/api/jobs");
  $("jobsView").innerHTML = data.jobs
    .map(
      (job) => {
        const progress = Math.max(0, Math.min(100, Number(job.progress_percent || 0)));
        const dataText = job.progress_target_gb
          ? `${fmt(job.progress_data_gb || 0)} / ${fmt(job.progress_target_gb)} GB referenciados`
          : job.progress_data_gb
            ? `${fmt(job.progress_data_gb)} GB referenciados`
            : "";
        return `
      <section class="job">
        <strong>${escapeHtml(job.operation)} <span class="status ${escapeHtml(job.status)}">${escapeHtml(job.status)}</span></strong>
        <span>${new Date(job.created_at * 1000).toLocaleString()}</span>
        <div class="progressHeader">
          <span>${escapeHtml(job.progress_label || "Sin progreso detallado")}</span>
          <strong>${progress.toFixed(1)}%</strong>
        </div>
        <div class="progressBar"><div style="width:${progress}%"></div></div>
        ${dataText ? `<div class="progressData">${escapeHtml(dataText)}</div>` : ""}
        ${job.error ? `<pre>${escapeHtml(job.error + "\n" + (job.traceback || ""))}</pre>` : ""}
        ${job.result ? `<pre>${escapeHtml(JSON.stringify(job.result, null, 2))}</pre>` : ""}
      </section>`;
      }
    )
    .join("");
}

function renderPrediction(result) {
  if (!result) return;
  const pred = result.prediction || {};
  const bars = (pred.top_k || [])
    .map(
      (item) => `
      <div class="sample">
        <strong>${escapeHtml(item.label)}</strong>
        <span>${fmt(item.probability)}</span>
        <div class="probBar"><div style="width:${Math.max(0, Math.min(100, item.probability * 100))}%"></div></div>
      </div>`
    )
    .join("");
  $("predictionView").innerHTML = `
    <div class="metric"><span>Prediccion</span><div class="value">${escapeHtml(pred.prediction)}</div></div>
    <div class="metric"><span>Esperado por metadata</span><div class="value">${escapeHtml(result.expected_from_metadata || "n/a")}</div></div>
    <div class="metric"><span>Confianza</span><div class="value">${fmt(pred.confidence)}</div></div>
    <p>${escapeHtml(result.data_path)}</p>
    ${bars}
  `;
  showTab("predict");
}

function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tabPanel").forEach((p) => p.classList.toggle("active", p.id === name));
}

function bind() {
  $("refreshBtn").addEventListener("click", async () => {
    await loadDatasets();
    await renderJobs();
  });
  $("datasetSelect").addEventListener("change", async () => {
    renderDatasetMeta();
    applyDatasetPreset();
    await loadReports();
    await loadSamples();
  });
  $("ieeeTarget").addEventListener("change", loadSamples);
  $("modelKind").addEventListener("change", loadReports);
  $("maxDataGb").addEventListener("input", renderDataBudgetPreview);
  $("maxDataPercent").addEventListener("input", renderDataBudgetPreview);
  $("selectAllModelsBtn").addEventListener("click", () => {
    document.querySelectorAll(".compareModelCheck").forEach((el) => { el.checked = true; });
  });
  $("clearModelsBtn").addEventListener("click", () => {
    document.querySelectorAll(".compareModelCheck").forEach((el) => { el.checked = false; });
  });
  $("presetBtn").addEventListener("click", applyDatasetPreset);
  $("discoverBtn").addEventListener("click", () => startOperation("discover"));
  $("trainBtn").addEventListener("click", () => startOperation("train"));
  $("compareBtn").addEventListener("click", () =>
    startOperation("compare", {
      ...params(),
      model_kinds: selectedCompareModels(),
      prediction_limit: numberOrNull("predictionLimit") || 5,
    })
  );
  $("retrainBtn").addEventListener("click", () => startOperation("retrain"));
  $("summaryBtn").addEventListener("click", () => startOperation("summary", {}));
  $("validateBtn").addEventListener("click", async () => {
    await loadReports();
    showTab("metrics");
  });
  $("predictBtn").addEventListener("click", () => {
    const sample = state.samples[Number($("sampleSelect").value)];
    const selectedKind = $("modelKind").value;
    const trained = (state.reports?.models?.models || []).find((m) => m.model_kind === selectedKind);
    const modelPath = trained?.model_path || state.reports?.model_path || state.selected.model_path;
    startOperation("predict", {
      model_path: modelPath,
      meta_path: sample.meta_path,
      data_path: sample.data_path,
      dtype: sample.dtype,
      top_k: Number($("topK").value || 5),
    });
  });
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => showTab(tab.dataset.tab)));
}

bind();
loadDatasets().then(() => {
  renderJobs();
}).catch((err) => {
  $("inventoryView").innerHTML = `<pre>${escapeHtml(err.stack || err.message)}</pre>`;
});
