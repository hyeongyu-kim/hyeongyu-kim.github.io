// Short readings of the recorded results, with the same baseline as the score table.
export function adaptationReading({ frame, baseline, classes, sample, batch, position, stream, reset }) {
  const adapters = ["tent", "buffer", "actta"];
  const names = { tent: "Tent", buffer: "Buffer", actta: "AcTTA" };
  const top = (state) => {
    const probabilities = state.probabilities[sample];
    const confidence = Math.max(...probabilities);
    return { label: probabilities.indexOf(confidence), confidence };
  };
  const pct = (value) => (value * 100).toFixed(1) + "%";
  const truth = frame.labels[sample];
  const source = top(frame.methods.source);
  const predictions = adapters.map((method) => top(frame.methods[method]));
  const changes = adapters.map((method) => {
    const before = Math.round(baseline[method].metrics.accuracy * batch);
    const after = Math.round(frame.methods[method].metrics.accuracy * batch);
    return { method, before, after, delta: after - before };
  });
  let headline, body, kind;
  if (!stream && position === 0) {
    kind = "starting";
    headline = "Batch statistics get a head start.";
    body =
      "The adapters start with this batch’s BN statistics. Source keeps its stored statistics. No learned update yet; try Next TTA update to change the trainable parameters without labels.";
  } else if (stream && frame.domain === "clean" && position >= 12 && !reset) {
    kind = "return";
    headline = "Back home. The updates came along.";
    body =
      "Clean inputs are back. Continual adapters retain their earlier updates, then update once on this batch. These scores are after that update.";
  } else if (stream && reset) {
    kind = "reset";
    headline = "New batch. Fresh start.";
    body = "Parameters and optimizer reset before each incoming batch. The predictions here are after its one recorded update.";
  } else if (predictions.every((prediction) => prediction.label !== truth)) {
    kind = "miss";
    const unanimous = predictions.every((prediction) => prediction.label === predictions[0].label && prediction.confidence >= 0.95);
    headline = unanimous ? "Unanimous. Confident. Wrong." : "Adapted. Still fooled.";
    body = unanimous
      ? "All three adapters say “" +
        classes[predictions[0].label] +
        "”. The label is “" +
        classes[truth] +
        "”. Whole-batch scores cover " +
        batch +
        " images."
      : "All three adapters miss this inspected image. The whole-batch scores summarize " +
        batch +
        " images; an individual prediction can tell another story.";
  } else if (batch === 4) {
    kind = "small";
    headline = "Four images. Big swings.";
    body = "One correct image is worth 25 percentage points here. A small batch makes scores noisy, so watch the correct-image counts too.";
  } else {
    const gain = [...changes].sort((a, b) => b.delta - a.delta)[0];
    const overconfident = changes.find(
      ({ method, delta }) => delta <= 0 && frame.methods[method].metrics.confidence > baseline[method].metrics.confidence + 1e-6
    );
    const loss = changes.find(({ delta }) => delta < 0);
    if (gain.delta > 0) {
      kind = "gain";
      const losses = changes.filter(({ delta }) => delta < 0);
      headline = losses.length
        ? "TTA updates. Mixed fortunes."
        : "Small tweaks. " + gain.delta + " more " + (gain.delta === 1 ? "image" : "images") + " right.";
      body =
        names[gain.method] +
        " goes from " +
        gain.before +
        "/" +
        batch +
        " to " +
        gain.after +
        "/" +
        batch +
        " correct. " +
        losses.map(({ method, delta }) => names[method] + " has " + -delta + " fewer correct. ").join("") +
        "These scores describe the inputs used for adaptation.";
    } else if (overconfident) {
      kind = "confidence";
      headline = overconfident.delta < 0 ? "More confident. Fewer correct." : "More confident. Same score.";
      const method = overconfident.method;
      body =
        names[method] +
        "’s mean confidence rises from " +
        pct(baseline[method].metrics.confidence) +
        " to " +
        pct(frame.methods[method].metrics.confidence) +
        ". Correct predictions: " +
        overconfident.before +
        " → " +
        overconfident.after +
        " out of " +
        batch +
        ".";
    } else if (loss) {
      kind = "loss";
      headline = "A tweak can go the wrong way.";
      body =
        names[loss.method] +
        " goes from " +
        loss.before +
        "/" +
        batch +
        " to " +
        loss.after +
        "/" +
        batch +
        " correct. Every recorded outcome is retained.";
    } else {
      kind = "steady";
      headline = "Small changes. Same score.";
      body =
        "The adapters have updated, but their whole-batch correct-image counts are unchanged at this step. Check the parameter readouts above for the learned changes.";
    }
  }
  return {
    kind,
    headline,
    body,
    changes,
    sourceGuess: classes[source.label] + ". " + pct(source.confidence) + " sure.",
    sourceVerdict: source.label === truth ? "Right this time" : source.confidence >= 0.9 ? "Confidently wrong" : "Wrong guess",
    sourceCorrect: source.label === truth,
  };
}
