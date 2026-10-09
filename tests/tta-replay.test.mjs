import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import crypto from "node:crypto";
import { gunzipSync } from "node:zlib";
import { activationValue, activationSlope, experimentCsv } from "../assets/js/tta-math.js";

const manifest = JSON.parse(fs.readFileSync(new URL("../assets/lab/tta-manifest.json", import.meta.url)));
const traces = manifest.traces.map((meta) => {
  const bytes = fs.readFileSync(new URL("../assets/lab/tta/" + meta.file, import.meta.url));
  assert.equal(bytes.length, meta.bytes);
  assert.equal(crypto.createHash("sha256").update(bytes).digest("hex"), meta.sha256);
  return JSON.parse(gunzipSync(bytes));
});
function recompute(p, labels) {
  const bins = Array.from({ length: 10 }, () => []);
  let correct = 0,
    confidence = 0,
    entropy = 0;
  p.forEach((row, index) => {
    const max = Math.max(...row),
      label = row.indexOf(max),
      hit = label === labels[index] ? 1 : 0;
    assert.ok(Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-5);
    assert.ok(row.every((v) => Number.isFinite(v) && v >= 0 && v <= 1));
    bins[Math.min(9, Math.floor(max * 10))].push([max, hit]);
    correct += hit;
    confidence += max;
    entropy -= row.reduce((sum, v) => sum + v * Math.log(Math.max(v, 1e-30)), 0);
  });
  const ece = bins.reduce((sum, bin) => sum + Math.abs(bin.reduce((value, [c, hit]) => value + hit - c, 0)) / p.length, 0);
  return { accuracy: correct / p.length, confidence: confidence / p.length, entropy: entropy / p.length, ece };
}
test("recordings declare pinned public models and batch-dependent method rates", () => {
  assert.equal(manifest.schema, 2);
  assert.equal(manifest.model, "RobustBench Standard WRN-28-10");
  assert.equal(manifest.sourceParameters, 36479194);
  assert.deepEqual(manifest.batches, [4, 16, 128]);
  assert.deepEqual(manifest.trainableParameters, { source: 0, tent: 17952, buffer: 9221171, actta: 3408 });
  assert.equal(manifest.implementation.actta.commit, "6bd3b9ab090240ec67038257c5fd9d8f11a283fc");
  assert.equal(manifest.implementation.buffer.commit, "b266f3c7904aa760b21618ebe5fba0fcd864bc16");
  assert.deepEqual(manifest.implementation.actta.prefixes, ["block1"]);
  assert.deepEqual(manifest.implementation.buffer.stages, [true, true, false]);
  assert.deepEqual(manifest.protocol.learningRatesByBatch["4"], { tent: 0.0001, buffer: 0.001, actta: 0.001 });
  assert.deepEqual(manifest.protocol.learningRatesByBatch["128"], { tent: 0.001, buffer: 0.001, actta: 0.01 });
  assert.equal(manifest.protocol.data.checkpointSha256, "6ed1c75dad63e8ebdbef365020dd1a7eee5e79cf8a1438bd3bdc3b9c7aab3d44");
  assert.equal(new Set(traces.map((t) => t.recordingProtocolSha256)).size, 1);
  assert.ok(traces.every((t) => manifest.batches.includes(t.batch) && /^[a-f0-9]{64}$/.test(t.recordingProtocolSha256)));
});
test("all 447 states independently recover their full-batch scores", () => {
  assert.equal(manifest.conditions, 45);
  assert.equal(manifest.frames, 447);
  for (const trace of traces)
    for (const frame of trace.frames) {
      for (const block of [frame.methods, ...(frame.online ? [frame.online] : [])])
        for (const arm of Object.values(block)) {
          assert.equal(arm.probabilities.length, trace.batch);
          const expected = recompute(arm.probabilities, frame.labels);
          for (const key of Object.keys(expected)) assert.ok(Math.abs(expected[key] - arm.metrics[key]) < 1e-8, key);
        }
    }
});
test("stream histories are pre-update and clean phases reuse original indices", () => {
  const streams = traces.filter((t) => t.kind === "stream");
  assert.equal(streams.length, 6);
  for (const trace of streams) {
    assert.equal(trace.frames.length, 16);
    assert.deepEqual(
      trace.frames.map((f) => f.domain),
      ["clean", "gaussian_noise", "defocus_blur", "clean"].flatMap((v) => Array(4).fill(v))
    );
    for (let i = 0; i < 4; i++) {
      assert.deepEqual(trace.frames[i].indices, trace.frames[12 + i].indices);
      assert.deepEqual(trace.frames[i].original, trace.frames[12 + i].original);
      if (trace.policy === "reset")
        for (const m of ["tent", "buffer", "actta"]) {
          assert.deepEqual(trace.frames[i].online[m].probabilities, trace.frames[12 + i].online[m].probabilities);
          assert.deepEqual(trace.frames[i].methods[m].probabilities, trace.frames[12 + i].methods[m].probabilities);
        }
    }
    const csv = experimentCsv(trace);
    assert.ok(csv.includes("pre_update_online"));
    assert.ok(csv.includes("post_update"));
  }
});
test("single-batch zero precedes learning; Source is fixed and AcTTA recovers ReLU", () => {
  for (const trace of traces.filter((t) => t.kind === "batch")) {
    assert.equal(trace.frames.length, 9);
    const first = trace.frames[0].methods;
    assert.deepEqual(first.tent.probabilities, first.actta.probabilities);
    for (const f of trace.frames) assert.deepEqual(f.methods.source.probabilities, first.source.probabilities);
    for (const layer of ["stem", "early"]) {
      assert.ok(first.actta.layers[layer].center.every((v) => v === 0));
      assert.ok(first.actta.layers[layer].positive.every((v) => v === 0));
      assert.ok(first.actta.layers[layer].negative.every((v) => v === 0));
    }
  }
});
test("feature map bytes have complete coverage with shared measured scales", () => {
  for (const trace of traces)
    for (const f of trace.frames)
      for (const layer of ["stem", "early"]) {
        const reference = Buffer.from(f.featureReference[layer], "base64");
        assert.equal(reference.length, 4 * 16 * 64);
        assert.ok(trace.mapScales[layer] > 0);
        assert.ok(trace.residualScales[layer] > 0);
        for (const m of ["source", "tent", "buffer", "actta"]) {
          const delta = Buffer.from(f.methods[m].layers[layer].map, "base64");
          assert.equal(delta.length, reference.length);
          if (m === "source") assert.ok(delta.every((v) => v === 0));
        }
      }
});
test("activation slope agrees with a numerical derivative away from ReLU's corner", () => {
  for (const x of [-2, -0.5, 0.5, 2])
    for (const c of [-0.1, 0, 0.1]) {
      const epsilon = 1e-5;
      const expected = (activationValue(x + epsilon, c, 0.12, -0.07) - activationValue(x - epsilon, c, 0.12, -0.07)) / (2 * epsilon);
      assert.ok(Math.abs(expected - activationSlope(x, c, 0.12, -0.07)) < 1e-7);
      assert.equal(activationValue(x, 0, 0, 0), Math.max(0, x));
    }
});
