import { corruptDigits, firstLayer, alignFeatures, predictFeatures, prediction, batchMetrics, tileMetrics, tilePosition } from "./lab-math.js";

const element = (id) => document.getElementById(id);
const text = (id, value) => {
  element(id).textContent = value;
};
const percent = (value) => `${(value * 100).toFixed(1)}%`;
const colors = () => {
  const style = getComputedStyle(document.documentElement);
  return {
    blue: style.getPropertyValue("--global-theme-color").trim(),
    text: style.getPropertyValue("--global-text-color").trim(),
    divider: style.getPropertyValue("--global-divider-color").trim(),
    background: style.getPropertyValue("--global-bg-color").trim(),
  };
};

function drawDigit(canvas, pixels) {
  const context = canvas.getContext("2d");
  pixels.forEach((pixel, i) => {
    const level = Math.round(pixel * 255);
    context.fillStyle = `rgb(${level},${level},${level})`;
    context.fillRect((i % 8) * 20, Math.floor(i / 8) * 20, 20, 20);
  });
}

function showBars(id, probabilities) {
  const container = element(id);
  container.replaceChildren();
  probabilities
    .map((value, label) => ({ value, label }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 3)
    .forEach((item) => {
      const row = document.createElement("div");
      row.className = "probability-row";
      const label = document.createElement("span");
      label.textContent = String(item.label);
      const track = document.createElement("span");
      track.className = "probability-track";
      const bar = document.createElement("span");
      bar.className = "probability-fill";
      bar.style.width = percent(item.value);
      track.append(bar);
      const value = document.createElement("span");
      value.textContent = percent(item.value);
      row.append(label, track, value);
      container.append(row);
    });
}

async function setupDigits() {
  const root = element("digit-lab");
  if (!root) return;
  try {
    const response = await fetch(root.dataset.modelUrl, { credentials: "omit" });
    if (!response.ok) throw new Error("Digit data could not be loaded");
    const model = await response.json();
    let cursor = 0,
      sample = 0,
      adapted = false;
    let currentSamples = [],
      inputs = [],
      features = [],
      source = [],
      adaptation = [];

    const renderSample = () => {
      drawDigit(
        element("digit-original"),
        currentSamples[sample].pixels.map((value) => value / 16)
      );
      drawDigit(element("digit-shifted"), inputs[sample]);
      text("digit-sample", `Sample ${sample + 1} / ${currentSamples.length}`);
      text("digit-label", `Ground truth: ${currentSamples[sample].label} · used for evaluation only`);
      const original = prediction(source[sample]);
      text("source-prediction", `Digit ${original.label} · ${percent(original.confidence)}`);
      showBars("source-bars", source[sample]);
      if (adapted) {
        const update = prediction(adaptation[sample]);
        text("adapt-prediction", `Digit ${update.label} · ${percent(update.confidence)}`);
        showBars("adapt-bars", adaptation[sample]);
      } else {
        text("adapt-prediction", "—");
        const note = document.createElement("p");
        note.className = "lab-caption";
        note.textContent = "Select “Adapt this batch” to update its statistics.";
        element("adapt-bars").replaceChildren(note);
      }
    };

    const showMetrics = (prefix, probabilities) => {
      const metrics = batchMetrics(
        probabilities,
        currentSamples.map((item) => item.label)
      );
      text(`${prefix}-accuracy`, percent(metrics.accuracy));
      text(`${prefix}-ece`, percent(metrics.ece));
      text(`${prefix}-confidence`, percent(metrics.confidence));
    };

    const renderBatch = () => {
      const count = Number(element("digit-batch").value);
      const severity = Number(element("digit-severity").value);
      currentSamples = Array.from({ length: count }, (_, i) => model.samples[(cursor + i) % model.samples.length]);
      inputs = corruptDigits(currentSamples, element("digit-shift").value, severity, model.seed + cursor);
      features = firstLayer(inputs, model);
      source = predictFeatures(features, model);
      adapted = false;
      sample = 0;
      text("digit-severity-value", `${severity} / 5`);
      text("digit-status", `${count} held-out images · fixed source weights · adaptation uses the current batch without labels`);
      ["accuracy", "ece", "confidence"].forEach((key) => text(`adapt-${key}`, "—"));
      showMetrics("source", source);
      renderSample();
    };

    ["digit-shift", "digit-batch"].forEach((id) => element(id).addEventListener("change", renderBatch));
    element("digit-severity").addEventListener("input", renderBatch);
    element("digit-adapt").addEventListener("click", () => {
      adaptation = predictFeatures(alignFeatures(features, model.sourceMean, model.sourceStd), model);
      adapted = true;
      showMetrics("adapt", adaptation);
      renderSample();
      text("digit-status", `Adapted ${currentSamples.length} unlabeled images · classifier weights are unchanged`);
    });
    element("digit-next-batch").addEventListener("click", () => {
      cursor = (cursor + currentSamples.length) % model.samples.length;
      renderBatch();
    });
    element("digit-previous").addEventListener("click", () => {
      sample = (sample + currentSamples.length - 1) % currentSamples.length;
      renderSample();
    });
    element("digit-next").addEventListener("click", () => {
      sample = (sample + 1) % currentSamples.length;
      renderSample();
    });
    element("digit-reset").addEventListener("click", () => {
      cursor = 0;
      element("digit-shift").value = "brightness";
      element("digit-severity").value = "3";
      element("digit-batch").value = "64";
      renderBatch();
    });
    ["digit-adapt", "digit-next-batch", "digit-reset", "digit-previous", "digit-next"].forEach((id) => {
      element(id).disabled = false;
    });
    renderBatch();
  } catch (error) {
    text("digit-status", "The digit experiment could not load. Reload this page, or use the model download below.");
    console.error(error);
  }
}

function drawMatrix(canvas, rows, columns, tileRows, tileColumns, active) {
  const context = canvas.getContext("2d");
  const palette = colors();
  const margin = 10,
    width = canvas.width - margin * 2,
    height = canvas.height - margin * 2;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = palette.background;
  context.fillRect(margin, margin, width, height);
  context.globalAlpha = 0.22;
  context.fillStyle = palette.blue;
  context.fillRect(
    margin + (active.c0 / columns) * width,
    margin + (active.r0 / rows) * height,
    ((active.c1 - active.c0) / columns) * width,
    ((active.r1 - active.r0) / rows) * height
  );
  context.globalAlpha = 1;
  context.strokeStyle = palette.divider;
  context.lineWidth = 1;
  context.beginPath();
  for (let row = 0; row <= rows; row += tileRows) {
    context.moveTo(margin, margin + (row / rows) * height);
    context.lineTo(margin + width, margin + (row / rows) * height);
  }
  for (let column = 0; column <= columns; column += tileColumns) {
    context.moveTo(margin + (column / columns) * width, margin);
    context.lineTo(margin + (column / columns) * width, margin + height);
  }
  context.stroke();
  context.strokeRect(margin, margin, width, height);
  context.strokeStyle = palette.blue;
  context.lineWidth = 2;
  context.strokeRect(
    margin + (active.c0 / columns) * width,
    margin + (active.r0 / rows) * height,
    ((active.c1 - active.c0) / columns) * width,
    ((active.r1 - active.r0) / rows) * height
  );
  canvas.setAttribute(
    "aria-label",
    `Matrix: ${rows} rows, ${columns} columns; active rows ${active.r0} to ${active.r1 - 1}, columns ${active.c0} to ${active.c1 - 1}`
  );
}

function setupTiles() {
  if (!element("tile-lab")) return;
  let step = 0,
    timer;
  const config = () => Object.fromEntries(["m", "n", "k", "tm", "tn", "tk"].map((key) => [key, Number(element(`tile-${key}`).value)]));
  const kib = (value) => `${(value / 1024).toLocaleString(undefined, { maximumFractionDigits: 2 })} KiB`;
  const stop = () => {
    clearInterval(timer);
    timer = undefined;
    text("tile-play", "Play");
  };
  const render = () => {
    const c = config(),
      metric = tileMetrics(c),
      position = tilePosition(c, step);
    const { m0, m1, n0, n1, k0, k1 } = position;
    drawMatrix(element("tile-a"), c.m, c.k, c.tm, c.tk, { r0: m0, r1: m1, c0: k0, c1: k1 });
    drawMatrix(element("tile-b"), c.k, c.n, c.tk, c.tn, { r0: k0, r1: k1, c0: n0, c1: n1 });
    drawMatrix(element("tile-c"), c.m, c.n, c.tm, c.tn, { r0: m0, r1: m1, c0: n0, c1: n1 });
    text("tile-footprint", kib(metric.footprint));
    text("tile-traffic", kib(metric.traffic));
    text("tile-operations", metric.operations.toLocaleString());
    text("tile-flops", metric.flops.toLocaleString());
    text("tile-position", `Tile ${step + 1} / ${metric.operations}`);
    text("tile-ranges", `Active ranges: M [${m0}, ${m1}), N [${n0}, ${n1}), K [${k0}, ${k1}). The C tile stays resident across K tiles.`);
    text(
      "tile-loop",
      `# Output-stationary loop sketch, including edge tiles\nfor mo in range(0, ${c.m}, ${c.tm}):\n    for no in range(0, ${c.n}, ${c.tn}):\n        acc = zeros(min(${c.tm}, ${c.m} - mo), min(${c.tn}, ${c.n} - no))\n        for ko in range(0, ${c.k}, ${c.tk}):\n            acc += A[mo:mo+${c.tm}, ko:ko+${c.tk}] @ B[ko:ko+${c.tk}, no:no+${c.tn}]\n        C[mo:mo+${c.tm}, no:no+${c.tn}] = acc`
    );
  };
  const advance = () => {
    step = (step + 1) % tileMetrics(config()).operations;
    render();
  };
  element("tile-step").addEventListener("click", () => {
    stop();
    advance();
  });
  element("tile-reset").addEventListener("click", () => {
    stop();
    step = 0;
    render();
  });
  element("tile-play").addEventListener("click", () => {
    if (timer) stop();
    else {
      text("tile-play", "Pause");
      timer = setInterval(advance, 500);
    }
  });
  ["m", "n", "k", "tm", "tn", "tk"].forEach((key) =>
    element(`tile-${key}`).addEventListener("change", () => {
      stop();
      step = 0;
      render();
    })
  );
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stop();
  });
  window.addEventListener("pagehide", stop);
  new MutationObserver(render).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  render();
}

setupDigits();
setupTiles();
