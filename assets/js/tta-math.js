export function activationValue(x, center, positive, negative) {
  const z = x - center;
  const s = 1 / (1 + Math.exp(-z));
  return Math.max(0, z) + (negative + (positive - negative) * s) * z;
}

export function activationSlope(x, center, positive, negative) {
  const z = x - center;
  const s = 1 / (1 + Math.exp(-z));
  return (z > 0 ? 1 : 0) + negative + (positive - negative) * (s + z * s * (1 - s));
}

export function experimentCsv(trace) {
  const rows = [
    ["experiment", "batch_size", "frame", "domain", "severity", "policy", "measurement", "method", "accuracy", "ece", "confidence", "entropy_nats"],
  ];
  trace.frames.forEach((frame, index) => {
    for (const measurement of ["post_update", ...(frame.online ? ["pre_update_online"] : [])]) {
      const values = measurement === "post_update" ? frame.methods : frame.online;
      for (const method of ["source", "tent", "buffer", "actta"]) {
        const m = values[method].metrics;
        rows.push([
          trace.kind,
          trace.batch,
          index,
          frame.domain,
          frame.severity,
          trace.policy || "repeated_batch",
          measurement,
          method,
          m.accuracy,
          m.ece,
          m.confidence,
          m.entropy,
        ]);
      }
    }
  });
  return rows.map((r) => r.join(",")).join("\n") + "\n";
}
