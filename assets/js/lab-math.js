// Pure numerical routines. Adaptation accepts features and source statistics,
// never labels. The evaluation routines are kept separate.
export function randomGenerator(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function corruptDigits(samples, shift, severity, seed = 27) {
  const random = randomGenerator(seed);
  const normal = () => Math.sqrt(-2 * Math.log(Math.max(random(), 1e-9))) * Math.cos(2 * Math.PI * random());
  return samples.map((sample) =>
    sample.pixels.map((pixel) => {
      let value = pixel / 16;
      const noise = normal();
      if (shift === "brightness") value += 0.055 * severity;
      if (shift === "contrast") value = value * (1 - 0.13 * severity) + 0.035 * severity;
      if (shift === "noise") value += noise * 0.045 * severity;
      if (shift === "mixed") value = value * (1 - 0.1 * severity) + 0.05 * severity + noise * 0.025 * severity;
      return Math.min(1, Math.max(0, value));
    })
  );
}

export function firstLayer(inputs, model) {
  return inputs.map((input) =>
    model.b1.map((bias, j) => {
      let value = bias;
      for (let i = 0; i < input.length; i++) value += input[i] * model.w1[i][j];
      return value;
    })
  );
}

export function alignFeatures(features, sourceMean, sourceStd) {
  if (!features.length) return [];
  const mean = sourceMean.map((_, j) => features.reduce((sum, row) => sum + row[j], 0) / features.length);
  const std = mean.map((value, j) => Math.max(Math.sqrt(features.reduce((sum, row) => sum + (row[j] - value) ** 2, 0) / features.length), 1e-6));
  return features.map((row) => row.map((value, j) => ((value - mean[j]) / std[j]) * sourceStd[j] + sourceMean[j]));
}

export function predictFeatures(features, model) {
  return features.map((feature) => {
    const logits = model.b2.map((bias, j) => {
      let value = bias;
      for (let i = 0; i < feature.length; i++) value += Math.max(0, feature[i]) * model.w2[i][j];
      return value;
    });
    const maximum = Math.max(...logits);
    const exp = logits.map((value) => Math.exp(value - maximum));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map((value) => value / sum);
  });
}

export function prediction(probabilities) {
  const confidence = Math.max(...probabilities);
  return { label: probabilities.indexOf(confidence), confidence };
}

export function batchMetrics(probabilities, labels) {
  const bins = Array.from({ length: 10 }, () => ({ count: 0, correct: 0, confidence: 0 }));
  let correct = 0;
  let confidence = 0;
  probabilities.forEach((row, i) => {
    const item = prediction(row);
    const hit = Number(item.label === labels[i]);
    correct += hit;
    confidence += item.confidence;
    const bin = bins[Math.min(9, Math.floor(item.confidence * 10))];
    bin.count++;
    bin.correct += hit;
    bin.confidence += item.confidence;
  });
  const count = probabilities.length;
  const ece = bins.reduce((sum, bin) => sum + (bin.count ? Math.abs(bin.correct - bin.confidence) / count : 0), 0);
  return { accuracy: correct / count, confidence: confidence / count, ece };
}
