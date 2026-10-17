export type AttentionValue = number | null;
export type AttentionPoint = { time: number; value: number };

/** Null is missing model evidence, never a zero score. */
export function attentionVector(value: unknown): AttentionValue[] | null {
  if (!Array.isArray(value) || value.length === 0) return null;
  if (!value.every((item) => item === null || (typeof item === "number" && Number.isFinite(item)))) return null;
  return value as AttentionValue[];
}

export function interpolateAttention(values: AttentionValue[], index: number, width: number): AttentionValue {
  if (values.length === 0) return null;
  if (values.length === 1) return values[0];
  const position = (index / Math.max(1, width - 1)) * (values.length - 1);
  const lower = Math.floor(position);
  const upper = Math.min(values.length - 1, lower + 1);
  const left = values[lower];
  const right = values[upper];
  if (position === lower) return left;
  if (left === null || right === null) return null;
  return left + (right - left) * (position - lower);
}

/** Interpolate on source timestamps only; never extrapolate or bridge a gap. */
export function attentionAtTime(values: AttentionValue[], times: number[], seconds: number): AttentionValue {
  if (values.length !== times.length || !times.length || seconds < times[0] || seconds > times[times.length - 1]) return null;
  let lower = 0;
  let upper = times.length - 1;
  while (upper - lower > 1) {
    const middle = Math.floor((lower + upper) / 2);
    if (times[middle] <= seconds) lower = middle;
    else upper = middle;
  }
  if (seconds === times[lower]) return values[lower];
  if (seconds === times[upper]) return values[upper];
  const left = values[lower];
  const right = values[upper];
  if (left === null || right === null || times[upper] <= times[lower]) return null;
  return left + (right - left) * ((seconds - times[lower]) / (times[upper] - times[lower]));
}

/** Separate drawing paths so raw curves, smoothing, and fills cannot cross gaps. */
export function attentionSegments(values: AttentionValue[], times: number[] | null, duration: number): AttentionPoint[][] {
  const segments: AttentionPoint[][] = [];
  let segment: AttentionPoint[] = [];
  values.forEach((value, index) => {
    if (value === null) {
      if (segment.length) segments.push(segment);
      segment = [];
      return;
    }
    const time = times?.length === values.length
      ? times[index]
      : (index / Math.max(1, values.length - 1)) * duration;
    segment.push({ time, value });
  });
  if (segment.length) segments.push(segment);
  return segments;
}
