import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
const source = fs.readFileSync(new URL("../assets/js/lab-math.js", import.meta.url), "utf8");
const { corruptDigits, firstLayer, alignFeatures, predictFeatures, batchMetrics } = await import(
  `data:text/javascript,${encodeURIComponent(source)}`
);
const model = JSON.parse(fs.readFileSync(new URL("../assets/lab/digits-model.json", import.meta.url), "utf8"));
const near = (a, b, tolerance = 1e-8) => assert.ok(Math.abs(a - b) < tolerance, `${a} differs from ${b}`);

test("browser inference agrees with independently exported Python probabilities", () => {
  const inputs = corruptDigits(model.samples, "brightness", 0);
  const probabilities = predictFeatures(firstLayer(inputs, model), model);
  probabilities[0].forEach((value, i) => near(value, model.referenceFirstProbability[i]));
  probabilities.forEach((row) =>
    near(
      row.reduce((a, b) => a + b, 0),
      1
    )
  );
  near(
    batchMetrics(
      probabilities,
      model.samples.map((item) => item.label)
    ).accuracy,
    model.cleanHeldOutAccuracy,
    1e-6
  );
});

test("the adaptation inputs are independent of held-out labels", () => {
  const relabeled = model.samples.slice(0, 64).map((item) => ({ ...item, label: (item.label + 1) % 10 }));
  const a = corruptDigits(model.samples.slice(0, 64), "mixed", 4, 99);
  const b = corruptDigits(relabeled, "mixed", 4, 99);
  assert.deepEqual(a, b);
  const predicted = predictFeatures(alignFeatures(firstLayer(a, model), model.sourceMean, model.sourceStd), model);
  const labels = model.samples.slice(0, 64).map((item) => item.label);
  assert.notEqual(
    batchMetrics(predicted, labels).accuracy,
    batchMetrics(
      predicted,
      relabeled.map((item) => item.label)
    ).accuracy
  );
});

test("feature alignment has the requested mean and population standard deviation", () => {
  const aligned = alignFeatures(
    [
      [1, 8],
      [2, 5],
      [4, 1],
      [9, 2],
    ],
    [-2, 3],
    [1.5, 0.75]
  );
  for (let j = 0; j < 2; j++) {
    const mean = aligned.reduce((sum, row) => sum + row[j], 0) / aligned.length;
    near(mean, [-2, 3][j]);
    near(Math.sqrt(aligned.reduce((sum, row) => sum + (row[j] - mean) ** 2, 0) / aligned.length), [1.5, 0.75][j]);
  }
  assert.ok(
    alignFeatures([[1], [1]], [0], [1])
      .flat()
      .every(Number.isFinite)
  );
});

test("ECE handles confidence-one boundaries and an analytically known batch", () => {
  const example = batchMetrics(
    [
      [0.8, 0.2],
      [0.8, 0.2],
    ],
    [0, 1]
  );
  near(example.accuracy, 0.5);
  near(example.confidence, 0.8);
  near(example.ece, 0.3);
  near(
    batchMetrics(
      [
        [1, 0],
        [0, 1],
      ],
      [0, 1]
    ).ece,
    0
  );
});
