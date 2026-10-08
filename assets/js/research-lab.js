import { corruptDigits, firstLayer, alignFeatures, predictFeatures, prediction, batchMetrics } from "./lab-math.js";

const element = (id) => document.getElementById(id);
const text = (id, value) => {
  element(id).textContent = value;
};
const percent = (value) => `${(value * 100).toFixed(1)}%`;
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

setupDigits();
