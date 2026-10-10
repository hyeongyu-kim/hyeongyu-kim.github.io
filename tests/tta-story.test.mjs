import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { gunzipSync } from "node:zlib";
import { adaptationReading } from "../assets/js/tta-story.js";

const manifest = JSON.parse(fs.readFileSync(new URL("../assets/lab/tta-manifest.json", import.meta.url)));
const load = (id) => {
  const meta = manifest.traces.find((trace) => trace.id === id);
  return JSON.parse(gunzipSync(fs.readFileSync(new URL("../assets/lab/tta/" + meta.file, import.meta.url))));
};
const read = (trace, position, sample = 0) =>
  adaptationReading({
    frame: trace.frames[position],
    baseline: trace.kind === "stream" ? trace.frames[position].online : trace.frames[0].methods,
    classes: manifest.classes,
    sample,
    batch: trace.batch,
    position,
    stream: trace.kind === "stream",
    reset: trace.policy === "reset",
  });
const noisy = load("batch-128-gaussian_noise-3");

test("the opening credits batch statistics, with no learned update claim", () => {
  const story = read(noisy, 0);
  assert.equal(story.kind, "starting");
  assert.ok(story.changes.every((change) => change.delta === 0));
  assert.equal(story.sourceVerdict, "Confidently wrong");
  assert.match(story.sourceGuess, /^frog\./);
});
test("a confidence observation compares measured mean confidence and correct counts", () => {
  const story = read(noisy, 1);
  assert.equal(story.kind, "confidence");
  assert.equal(story.headline, "More confident. Same score.");
  assert.match(story.body, /102 → 102/);
  assert.ok(noisy.frames[1].methods.tent.metrics.confidence > noisy.frames[0].methods.tent.metrics.confidence);
});
test("mixed outcomes retain the losing method in the short reading", () => {
  const story = read(noisy, 8);
  assert.equal(story.kind, "gain");
  assert.equal(story.headline, "TTA updates. Mixed fortunes.");
  assert.match(story.body, /Buffer has 1 fewer correct/);
  assert.match(story.body, /102\/128 to 103\/128/);
});
test("the unanimous miss is specific to the inspected image, not the batch", () => {
  const story = read(noisy, 8, 2);
  assert.equal(story.kind, "miss");
  assert.equal(story.headline, "Unanimous. Confident. Wrong.");
  assert.match(story.body, /“bird”/);
  assert.match(story.body, /“ship”/);
  assert.match(story.body, /128 images/);
});
test("return-to-clean and reset stories distinguish retained from restored state", () => {
  const continual = load("stream-128-continual");
  const reset = load("stream-128-reset");
  assert.equal(read(continual, 12).kind, "return");
  const fresh = read(reset, 12);
  assert.equal(fresh.kind, "reset");
});
test("every recorded narration reports independently checked count changes and source correctness", () => {
  let frames = 0;
  for (const meta of manifest.traces) {
    const trace = load(meta.id);
    for (let position = 0; position < trace.frames.length; position++) {
      frames++;
      const story = read(trace, position);
      const frame = trace.frames[position];
      const baseline = trace.kind === "stream" ? frame.online : trace.frames[0].methods;
      for (const change of story.changes) {
        const count = (state) => state.probabilities.reduce((correct, p, i) => correct + Number(p.indexOf(Math.max(...p)) === frame.labels[i]), 0);
        assert.equal(change.after, count(frame.methods[change.method]));
        assert.equal(change.before, count(baseline[change.method]));
        assert.equal(change.delta, change.after - change.before);
      }
      const p = frame.methods.source.probabilities[0];
      assert.equal(story.sourceCorrect, p.indexOf(Math.max(...p)) === frame.labels[0]);
      if (story.kind === "confidence") {
        assert.ok(
          story.changes.some(
            ({ method, delta }) => delta <= 0 && frame.methods[method].metrics.confidence > baseline[method].metrics.confidence + 1e-6
          )
        );
      }
    }
  }
  assert.equal(frames, 447);
});
