import { activationValue, activationSlope, experimentCsv } from "./tta-math.js";

// Replay measured PyTorch outputs. No substitute classifier or invented scores.
async function initializeTta() {
  const root = document.getElementById("tta-lab");
  if (!root) return;
  const get = (id) => document.getElementById("tta-" + id);
  const methods = ["source", "tent", "buffer", "actta"];
  const names = { source: "Source", tent: "Tent", buffer: "Buffer", actta: "AcTTA" };
  const domains = {
    clean: "Clean images",
    gaussian_noise: "Gaussian noise",
    brightness: "Brightness",
    defocus_blur: "Defocus blur",
    jpeg_compression: "JPEG compression",
  };
  const percent = (v) => (v * 100).toFixed(1) + "%";
  const number = (v) => Number(v).toFixed(4);
  const signed = (v) => (v < 0 ? "−" : "+") + Math.abs(v).toFixed(4);
  const traces = new Map(),
    packed = new Map();
  let manifest,
    trace,
    position = 0,
    timer = null,
    request = 0,
    embedded = null;
  const embeddedElement = document.getElementById("embedded-tta-data");
  if (embeddedElement) embedded = JSON.parse(embeddedElement.textContent);
  function color(method) {
    return getComputedStyle(root.querySelector(".tta-method.tta-" + method))
      .getPropertyValue("--tta-method-color")
      .trim();
  }
  function palette() {
    const style = getComputedStyle(root);
    return {
      ink: style.color,
      muted: style.getPropertyValue("--global-text-color-light").trim() || "#777",
      rule: style.getPropertyValue("--global-divider-color").trim() || "#ddd",
    };
  }
  function stop() {
    clearInterval(timer);
    timer = null;
    get("play").textContent = "Play";
  }
  function enableFrame(value) {
    for (const id of ["step", "play", "reset", "scrub", "sample", "layer", "channel", "metric", "slope", "difference", "export"])
      get(id).disabled = !value;
  }
  async function packedTrace(meta) {
    if (packed.has(meta.id)) return packed.get(meta.id);
    let bytes;
    if (embedded) bytes = Uint8Array.from(atob(embedded.files[meta.id]), (c) => c.charCodeAt(0));
    else {
      const url = new URL(meta.file, new URL(root.dataset.traceRoot, location.href));
      url.searchParams.set("sha256", meta.sha256);
      const response = await fetch(url, { credentials: "omit" });
      if (!response.ok) throw new Error("This recorded experiment could not be loaded.");
      bytes = new Uint8Array(await response.arrayBuffer());
    }
    if (crypto.subtle) {
      const hash = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), (v) => v.toString(16).padStart(2, "0")).join("");
      if (hash !== meta.sha256) throw new Error("The recorded experiment failed its checksum. Please reload the page.");
    }
    packed.set(meta.id, bytes);
    return bytes;
  }
  async function readTrace(meta) {
    if (traces.has(meta.id)) return traces.get(meta.id);
    if (typeof DecompressionStream === "undefined") throw new Error("Please use a browser that supports gzip DecompressionStream.");
    const bytes = await packedTrace(meta);
    const result = await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).json();
    traces.set(meta.id, result);
    return result;
  }
  const plotDimensions = new WeakMap();
  function axes(canvas, min, max, xMax, yLabel, xLabel, xMin = 0, xAxis = {}) {
    // Draw in CSS pixels so labels remain readable at every display width.
    // Closed details have no layout; retain their original size until opened.
    if (!plotDimensions.has(canvas)) plotDimensions.set(canvas, { width: canvas.width, height: canvas.height });
    const fallback = plotDimensions.get(canvas),
      bounds = canvas.getBoundingClientRect(),
      width = Math.round(bounds.width) || fallback.width,
      height = Math.round(bounds.height) || fallback.height,
      ratio = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * ratio)) canvas.width = Math.round(width * ratio);
    if (canvas.height !== Math.round(height * ratio)) canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext("2d"),
      p = palette();
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.font = "12px system-ui, sans-serif";
    if (max === min) {
      min -= 0.01;
      max += 0.01;
    }
    const precision = Math.min(6, Math.max(0, Math.ceil(-Math.log10((max - min) / 4)) + 1)),
      ticks = Array.from({ length: 5 }, (_, i) => min + ((max - min) * i) / 4),
      labels = ticks.map((v) => String(Number(v.toFixed(precision))).replace("-", "−"));
    const left = Math.max(44, Math.ceil(Math.max(...labels.map((label) => ctx.measureText(label).width))) + 12),
      right = width - 14,
      top = 34,
      bottom = height - 54;
    const x = (v) => left + ((v - xMin) / (xMax - xMin || 1)) * (right - left);
    const y = (v) => bottom - ((v - min) / (max - min)) * (bottom - top);
    ctx.clearRect(0, 0, width, height);
    ticks.forEach((v, i) => {
      ctx.strokeStyle = p.rule;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(left, y(v));
      ctx.lineTo(right, y(v));
      ctx.stroke();
      ctx.fillStyle = p.muted;
      ctx.textAlign = "right";
      ctx.fillText(labels[i], left - 8, y(v) + 4);
    });
    ctx.textAlign = "center";
    for (const v of xAxis.ticks || [xMin, (xMin + xMax) / 2, xMax]) {
      ctx.fillStyle = p.muted;
      ctx.fillText(xAxis.format ? xAxis.format(v) : String(v).replace("-", "−"), x(v), bottom + 20);
    }
    ctx.fillStyle = p.muted;
    ctx.textAlign = "left";
    ctx.fillText(yLabel, left, 17);
    ctx.textAlign = "center";
    ctx.fillText(xLabel, (left + right) / 2, height - 8);
    return { ctx, x, y, left, right, top, bottom, p };
  }
  function line(plot, points, stroke, dash = [], width = 2) {
    const c = plot.ctx;
    c.strokeStyle = stroke;
    c.lineWidth = width;
    c.setLineDash(dash);
    c.beginPath();
    points.forEach(([x, y], i) => (i ? c.lineTo(plot.x(x), plot.y(y)) : c.moveTo(plot.x(x), plot.y(y))));
    c.stroke();
    c.setLineDash([]);
  }
  function drawMap(canvas, bytes, scale, description) {
    const start = (Number(get("sample").value) * manifest.channels + Number(get("channel").value)) * 64;
    const c = canvas.getContext("2d"),
      cell = canvas.width / 8;
    let minimum = Infinity,
      maximum = -Infinity;
    bytes.subarray(start, start + 64).forEach((v, i) => {
      const q = (v - 128) / 127,
        t = Math.min(1, Math.abs(q));
      const a = [242, 239, 228],
        b = q < 0 ? [180, 88, 56] : [36, 93, 139];
      c.fillStyle = "rgb(" + a.map((value, k) => Math.round(value + (b[k] - value) * t)).join(",") + ")";
      c.fillRect((i % 8) * cell, Math.floor(i / 8) * cell, cell, cell);
      minimum = Math.min(minimum, q * scale);
      maximum = Math.max(maximum, q * scale);
    });
    canvas.setAttribute("aria-label", description + "; values " + number(minimum) + " to " + number(maximum));
  }
  const maps = new WeakMap();
  function mapBytes(item, reference, residual = false) {
    const key = residual ? "residual" : "map";
    let cache = maps.get(item);
    if (!cache) {
      cache = {};
      maps.set(item, cache);
    }
    if (!cache[key]) {
      cache[key] = Uint8Array.from(atob(item[key]), (c) => c.charCodeAt(0));
      if (!residual) {
        const base = Uint8Array.from(atob(reference), (c) => c.charCodeAt(0));
        cache[key] = cache[key].map((v, i) => v ^ base[i]);
      }
    }
    return cache[key];
  }
  function drawInternals() {
    if (!trace) return;
    const f = trace.frames[position],
      layer = get("layer").value,
      channel = Number(get("channel").value);
    const states = Object.fromEntries(methods.map((m) => [m, f.methods[m].layers[layer]]));
    const description = get("layer").selectedOptions[0].textContent + ", channel " + channel + ", image " + (Number(get("sample").value) + 1);
    methods.forEach((m) =>
      drawMap(get(m + "-map"), mapBytes(states[m], f.featureReference[layer]), trace.mapScales[layer], names[m] + " activation, " + description)
    );
    get("map-scale").textContent =
      "Shared scale: −" +
      number(trace.mapScales[layer]) +
      " to +" +
      number(trace.mapScales[layer]) +
      " · rust: negative · cream: zero · blue: positive";
    const dg = states.tent.gamma.map((v, i) => v - states.source.gamma[i]);
    const db = states.tent.beta.map((v, i) => v - states.source.beta[i]);
    const limit = Math.max(0.001, ...dg.map(Math.abs), ...db.map(Math.abs)) * 1.2;
    const affine = axes(get("affine-plot"), -limit, limit, 15, "Change from Source", "BN channel", 0, { ticks: [0, 5, 10, 15] });
    const c = affine.ctx;
    c.strokeStyle = color("tent");
    c.setLineDash([3, 5]);
    c.beginPath();
    c.moveTo(affine.x(channel), affine.top);
    c.lineTo(affine.x(channel), affine.bottom);
    c.stroke();
    c.setLineDash([]);
    line(
      affine,
      dg.map((v, i) => [i, v]),
      color("tent")
    );
    line(
      affine,
      db.map((v, i) => [i, v]),
      affine.p.ink,
      [5, 4]
    );
    get("affine-values").textContent = "Channel " + channel + " · Δγ " + signed(dg[channel]) + " (solid) · Δβ " + signed(db[channel]) + " (dashed)";
    drawMap(get("residual-map"), mapBytes(states.buffer, null, true), trace.residualScales[layer], "Buffer signed correction, " + description);
    get("residual-scale").textContent = "Own scale: ±" + trace.residualScales[layer].toPrecision(3);
    get("buffer-values").textContent =
      "α " + states.buffer.alpha.toPrecision(4) + " · s₁ " + number(states.buffer.scale1) + " · s₃ " + number(states.buffer.scale2);
    const a = states.actta,
      center = a.center[channel],
      positive = a.positive[channel],
      negative = a.negative[channel];
    const slope = get("slope").checked,
      difference = get("difference").checked;
    const values = [],
      base = [];
    for (let i = 0; i <= 160; i++) {
      const x = -3 + (6 * i) / 160;
      const r = slope ? (x > 0 ? 1 : 0) : Math.max(0, x);
      const v = slope ? activationSlope(x, center, positive, negative) : activationValue(x, center, positive, negative);
      values.push([x, difference ? v - r : v]);
      base.push([x, difference ? 0 : r]);
    }
    const all = [...values, ...base].map((p) => p[1]);
    let min = Math.min(...all),
      max = Math.max(...all);
    const padding = Math.max(0.0005, (max - min) * 0.12);
    const curve = axes(
      get("activation-plot"),
      min - padding,
      max + padding,
      3,
      difference ? "Change from ReLU" : slope ? "Local slope" : "Activation output",
      "Input x",
      -3,
      { ticks: [-3, -1.5, 0, 1.5, 3] }
    );
    line(curve, base, curve.p.muted, [6, 5]);
    line(curve, values, color("actta"));
    get("actta-values").textContent = "c " + signed(center) + " · λ+ " + signed(positive) + " · λ− " + signed(negative);
  }
  function drawHistory() {
    if (!trace) return;
    const metric = get("metric").value,
      stream = trace.kind === "stream";
    const values = methods.map((m) =>
      trace.frames.map((f) => (stream ? f.online[m] : f.methods[m]).metrics[metric] * (metric === "entropy" ? 1 : 100))
    );
    const flat = values.flat();
    const max = metric === "entropy" ? Math.max(0.2, ...flat) * 1.1 : 100;
    const plot = axes(
      get("history-plot"),
      0,
      max,
      trace.frames.length - 1,
      metric === "entropy" ? "Entropy (nats)" : "%",
      stream ? "Incoming batch (1–16)" : "Gradient updates (0–8)",
      0,
      { ticks: stream ? [0, 3, 7, 11, 15] : [0, 2, 4, 6, 8], format: (v) => String(stream ? v + 1 : v) }
    );
    if (stream) {
      [3.5, 7.5, 11.5].forEach((v) => {
        plot.ctx.strokeStyle = plot.p.rule;
        plot.ctx.setLineDash([5, 5]);
        plot.ctx.beginPath();
        plot.ctx.moveTo(plot.x(v), plot.top);
        plot.ctx.lineTo(plot.x(v), plot.bottom);
        plot.ctx.stroke();
      });
      plot.ctx.setLineDash([]);
      ["Clean", "Noise", "Blur", "Clean"].forEach((name, i) => {
        plot.ctx.fillStyle = plot.p.muted;
        plot.ctx.textAlign = "center";
        plot.ctx.fillText(name, plot.x(1.5 + i * 4), plot.top + 18);
      });
    }
    methods.forEach((m, i) => {
      line(
        plot,
        values[i].map((v, index) => [index, v]),
        color(m),
        m === "source" ? [6, 5] : []
      );
      plot.ctx.fillStyle = color(m);
      plot.ctx.beginPath();
      plot.ctx.arc(plot.x(position), plot.y(values[i][position]), 5, 0, 2 * Math.PI);
      plot.ctx.fill();
    });
    plot.ctx.strokeStyle = plot.p.muted;
    plot.ctx.lineWidth = 1;
    plot.ctx.setLineDash([2, 5]);
    plot.ctx.beginPath();
    plot.ctx.moveTo(plot.x(position), plot.top);
    plot.ctx.lineTo(plot.x(position), plot.bottom);
    plot.ctx.stroke();
    plot.ctx.setLineDash([]);
    get("history-caption").textContent = stream
      ? "Online predictions before each incoming batch's update. The table above is a post-update diagnostic."
      : "The full, same batch evaluated after each recorded update. This is not an unseen-data test.";
  }
  function render() {
    if (!trace) return;
    const f = trace.frames[position],
      sample = Number(get("sample").value),
      stream = trace.kind === "stream";
    get("original").src = f.original[sample];
    get("shifted").src = f.images[sample];
    get("domain").textContent = domains[f.domain] + (f.severity ? " · severity " + f.severity : "");
    get("label").textContent = "Ground truth: " + manifest.classes[f.labels[sample]] + " · evaluation only";
    get("position").textContent = stream ? "Incoming batch " + (position + 1) + " / 16 · one update per batch" : "Update " + position + " / 8";
    get("scrub").max = trace.frames.length - 1;
    get("scrub").value = position;
    get("step").textContent = stream ? "Next batch" : "Next update";
    get("step").disabled = position === trace.frames.length - 1;
    get("table-caption").textContent =
      "Current batch (" + trace.batch + " images), " + (stream ? "after its one update" : "after " + position + " updates");
    methods.forEach((method) => {
      const state = f.methods[method],
        p = state.probabilities[sample];
      const top = p
        .map((v, i) => [v, i])
        .sort((a, b) => b[0] - a[0])
        .slice(0, 3);
      get(method + "-prediction").textContent = manifest.classes[top[0][1]];
      get(method + "-confidence").textContent = percent(top[0][0]) + " confidence";
      const bars = get(method + "-bars");
      bars.replaceChildren();
      top.forEach(([v, id]) => {
        const row = document.createElement("div"),
          label = document.createElement("span"),
          value = document.createElement("span"),
          track = document.createElement("i");
        label.textContent = manifest.classes[id];
        value.textContent = percent(v);
        track.style.width = percent(v);
        row.append(label, value, track);
        bars.append(row);
      });
      for (const metric of ["accuracy", "ece", "entropy"]) {
        const cell = get(method + "-" + metric);
        cell.textContent = metric === "entropy" ? state.metrics[metric].toFixed(3) : percent(state.metrics[metric]);
        if (metric === "accuracy") {
          const count = document.createElement("small");
          count.className = "tta-score-count";
          count.textContent = Math.round(state.metrics.accuracy * trace.batch) + " / " + trace.batch;
          cell.append(count);
        }
      }
    });
    get("sample-resolution").textContent = trace.batch + " images · One image = " + (100 / trace.batch).toFixed(2) + " percentage points.";
    const baseline = stream ? f.online : trace.frames[0].methods;
    const changes = methods.slice(1).map((m) => {
      const delta = (f.methods[m].metrics.accuracy - baseline[m].metrics.accuracy) * 100;
      return names[m] + " " + (delta < 0 ? "−" : "+") + Math.abs(delta).toFixed(1) + " pp";
    });
    get("reading").textContent = (stream ? "This batch's update: " : "Learned change from step 0: ") + changes.join(" · ") + ".";
    drawInternals();
    drawHistory();
  }
  async function selectTrace(start = 0) {
    stop();
    enableFrame(false);
    trace = null;
    const token = ++request,
      stream = get("mode").value === "stream";
    get("policy-wrap").hidden = !stream;
    get("corruption").disabled = stream;
    get("severity").disabled = stream || get("corruption").value === "clean";
    get("corruption").closest("label").hidden = stream;
    get("severity").closest("label").hidden = stream;
    root.querySelector(".tta-controls").classList.toggle("tta-stream-controls", stream);
    const b = Number(get("batch").value),
      domain = get("corruption").value;
    const id = stream
      ? "stream-" + b + "-" + (get("policy").checked ? "reset" : "continual")
      : "batch-" + b + "-" + domain + "-" + (domain === "clean" ? 0 : get("severity").value);
    for (const [preset, selected] of [
      ["shift", id === "batch-128-gaussian_noise-3"],
      ["small", id === "batch-4-gaussian_noise-3"],
      ["return", id === "stream-128-continual"],
    ])
      get("preset-" + preset).setAttribute("aria-pressed", String(selected));
    get("status").textContent = "Loading this recorded experiment…";
    get("retry").hidden = true;
    try {
      const meta = manifest.traces.find((t) => t.id === id);
      if (!meta) throw new Error("This experiment is not in the run manifest.");
      const loaded = await readTrace(meta);
      if (token !== request) return;
      trace = loaded;
      const rates = manifest.protocol.learningRatesByBatch[String(b)];
      get("protocol-caption").textContent =
        "Same WRN checkpoint, inputs, and entropy objective · Adam LR: Tent " +
        rates.tent +
        " · Buffer " +
        rates.buffer +
        " · AcTTA " +
        rates.actta +
        ". Labels are used only for evaluation.";
      position = Math.min(start, trace.frames.length - 1);
      enableFrame(true);
      render();
      get("status").textContent = manifest.conditions + " experiments · " + manifest.frames + " measured states";
    } catch (error) {
      if (token === request) {
        get("status").textContent = error.message;
        get("retry").hidden = false;
      }
    }
  }
  function download(text, filename, type) {
    const url = URL.createObjectURL(new Blob([text], { type })),
      a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }
  async function saveOffline() {
    if (embedded) return;
    const button = get("offline");
    button.disabled = true;
    try {
      const response = await fetch(root.dataset.offlineShell, { credentials: "omit", cache: "no-store" });
      if (!response.ok) throw new Error("The offline copy could not be prepared.");
      const shell = await response.text(),
        files = {};
      for (let i = 0; i < manifest.traces.length; i++) {
        const meta = manifest.traces[i],
          bytes = await packedTrace(meta);
        let binary = "";
        for (let j = 0; j < bytes.length; j += 32768) binary += String.fromCharCode(...bytes.subarray(j, j + 32768));
        files[meta.id] = btoa(binary);
        button.textContent = "Preparing offline copy · " + (i + 1) + " / " + manifest.traces.length;
      }
      const data = JSON.stringify({ manifest, files }).replace(/</g, "\\u003c");
      const html = shell.replace("<!--TTA_DATA-->", '<script type="application/json" id="embedded-tta-data">' + data + "<\/script>");
      download(html, "Hyeongyu_Kim_TTA_Lab.html", "text/html");
    } catch (error) {
      get("status").textContent = error.message;
    } finally {
      button.disabled = false;
      button.textContent = "Save an offline copy ↗";
    }
  }
  async function boot() {
    try {
      let loaded;
      if (embedded) loaded = embedded.manifest;
      else {
        const r = await fetch(root.dataset.manifestUrl, { credentials: "omit", cache: "no-store" });
        if (!r.ok) throw new Error("The run manifest could not be loaded.");
        loaded = await r.json();
      }
      if (loaded.schema !== 2 || loaded.model !== "RobustBench Standard WRN-28-10" || !loaded.batches.includes(128))
        throw new Error("The current WRN recordings are not available yet. Please use Retry loading.");
      manifest = loaded;
      for (const id of ["mode", "batch", "corruption", "severity", "policy", "preset-shift", "preset-small", "preset-return", "offline"])
        get(id).disabled = false;
      methods.forEach((m) => {
        get(m + "-count").textContent = manifest.trainableParameters[m].toLocaleString();
      });
      get("channel").replaceChildren();
      for (let i = 0; i < manifest.channels; i++) {
        const o = document.createElement("option");
        o.value = i;
        o.textContent = i;
        get("channel").append(o);
      }
      if (embedded) {
        get("offline").disabled = true;
        get("offline").textContent = "This copy works offline";
      }
      await selectTrace();
    } catch (error) {
      get("status").textContent = error.message;
      get("retry").hidden = false;
    }
  }
  for (const id of ["mode", "corruption", "severity", "batch", "policy"]) get(id).addEventListener("change", () => selectTrace());
  for (const id of ["sample", "layer", "channel", "metric", "slope", "difference"]) get(id).addEventListener("change", render);
  get("step").addEventListener("click", () => {
    stop();
    position = Math.min(position + 1, trace.frames.length - 1);
    render();
  });
  get("reset").addEventListener("click", () => {
    stop();
    position = 0;
    render();
  });
  get("scrub").addEventListener("input", () => {
    stop();
    position = Number(get("scrub").value);
    render();
  });
  get("play").addEventListener("click", () => {
    if (timer) return stop();
    if (position === trace.frames.length - 1) position = 0;
    get("play").textContent = "Pause";
    render();
    timer = setInterval(() => {
      position = Math.min(position + 1, trace.frames.length - 1);
      render();
      if (position === trace.frames.length - 1) stop();
    }, 850);
  });
  for (const [id, batch, mode, start] of [
    ["shift", 128, "batch", 0],
    ["small", 4, "batch", 0],
    ["return", 128, "stream", 12],
  ])
    get("preset-" + id).addEventListener("click", () => {
      get("mode").value = mode;
      get("batch").value = batch;
      get("corruption").value = "gaussian_noise";
      get("severity").value = "3";
      get("policy").checked = false;
      selectTrace(start);
    });
  get("retry").addEventListener("click", () => (manifest ? selectTrace() : boot()));
  get("export").addEventListener("click", () => download(experimentCsv(trace), "TTA_" + trace.kind + "_B" + trace.batch + ".csv", "text/csv"));
  get("offline").addEventListener("click", saveOffline);
  new MutationObserver(() => {
    if (trace) render();
  }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  const refreshPlots = () => {
    if (!trace) return;
    drawInternals();
    drawHistory();
  };
  for (const id of ["internals", "history"])
    get(id).addEventListener("toggle", () => {
      if (get(id).open) refreshPlots();
    });
  if (typeof ResizeObserver !== "undefined") {
    const plotObserver = new ResizeObserver(refreshPlots);
    for (const id of ["affine-plot", "activation-plot", "history-plot"]) plotObserver.observe(get(id));
  } else window.addEventListener("resize", refreshPlots);
  await boot();
}
initializeTta();
