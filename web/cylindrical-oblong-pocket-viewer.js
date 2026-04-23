import * as THREE from "https://esm.sh/three@0.179.1";
import { OrbitControls } from "https://esm.sh/three@0.179.1/examples/jsm/controls/OrbitControls.js";

const DEG_TO_RAD = Math.PI / 180;
const TWO_PI = Math.PI * 2;

const DEFAULT_PARAMS = {
  shaftDiameter: 37,
  pocketLength: 25,
  pocketEndRadius: 4,
  pocketDepth: 0.2,
  toolDiameter: 2,
  stepOver: 0.4,
  patternCount: 6,
  patternStartAngle: 0,
  axialCenter: 0,
  oblongAngleDeg: 0,
  pathPointSpacing: 0.35,
  meshAxialSamples: 280,
  meshAngularSamples: 420,
  showToolpath: true,
  showBoundaries: true,
  outerFinish: "matte",
  pocketFinish: "shiny",
};

const CONTROL_SECTIONS = [
  {
    title: "Pocket Geometry",
    fields: [
      ["shaftDiameter", "Shaft Diameter", 8, 180, 0.1, "mm"],
      ["pocketLength", "Pocket Length", 2, 120, 0.1, "mm"],
      ["pocketEndRadius", "End Radius", 0.8, 30, 0.05, "mm"],
      ["pocketDepth", "Pocket Depth", 0.01, 5, 0.01, "mm"],
    ],
  },
  {
    title: "Pattern Layout",
    fields: [
      ["patternCount", "Pocket Count", 1, 48, 1, ""],
      ["patternStartAngle", "Start Angle", 0, 360, 0.5, "deg"],
      ["axialCenter", "Axial Center", -120, 120, 0.1, "mm"],
      ["oblongAngleDeg", "Oblong Axis", -90, 90, 0.25, "deg"],
    ],
  },
  {
    title: "Toolpath Overlay",
    fields: [
      ["toolDiameter", "Tool Diameter", 0.4, 20, 0.05, "mm"],
      ["stepOver", "Step-Over", 0.02, 4, 0.01, "mm"],
      ["pathPointSpacing", "Path Spacing", 0.05, 5, 0.01, "mm"],
    ],
  },
];

const FINISH_OPTIONS = [
  { value: "matte", label: "Matte" },
  { value: "shiny", label: "Shiny" },
];

function cloneParams(params) {
  return structuredClone(params);
}

function degToRad(value) {
  return value * DEG_TO_RAD;
}

function wrapArcLength(delta, circumference) {
  const shifted = delta + 0.5 * circumference;
  return ((((shifted % circumference) + circumference) % circumference) - 0.5 * circumference);
}

function buildSpiralLevels(maxRadius, stepOver) {
  if (maxRadius <= 1e-9) {
    return [0];
  }

  const step = Math.max(0.02, Number(stepOver));
  const levels = [];
  for (let value = 0; value <= maxRadius + 0.5 * step; value += step) {
    levels.push(Number(value.toFixed(9)));
  }
  if (levels.length === 0) {
    levels.push(0, maxRadius);
  }
  if (levels[levels.length - 1] < maxRadius - 1e-9) {
    levels.push(maxRadius);
  }
  levels[0] = 0;
  levels[levels.length - 1] = maxRadius;
  return [...new Set(levels)];
}

function sampleLinearMove(xStart, sStart, xEnd, sEnd, zCut, spacing) {
  const distance = Math.hypot(xEnd - xStart, sEnd - sStart);
  const ds = Math.max(0.05, Number(spacing));
  const count = Math.max(2, Math.ceil(distance / ds) + 1);
  const out = [];
  for (let idx = 0; idx < count; idx += 1) {
    const t = idx / (count - 1);
    out.push([
      xStart + (xEnd - xStart) * t,
      sStart + (sEnd - sStart) * t,
      zCut,
    ]);
  }
  return out;
}

function sampleCenterline(straightHalfLen, zCut, spacing) {
  if (straightHalfLen <= 1e-9) {
    return [[0, 0, zCut]];
  }
  return sampleLinearMove(-straightHalfLen, 0, straightHalfLen, 0, zCut, spacing);
}

function sampleObroundLoop(levelRadius, straightHalfLen, zCut, spacing) {
  if (levelRadius <= 1e-9) {
    return sampleCenterline(straightHalfLen, zCut, spacing);
  }

  const ds = Math.max(0.05, Number(spacing));
  if (straightHalfLen <= 1e-9) {
    const count = Math.max(28, Math.ceil((TWO_PI * levelRadius) / ds) + 1);
    const out = [];
    for (let idx = 0; idx < count; idx += 1) {
      const theta = -0.5 * Math.PI + (TWO_PI * idx) / (count - 1);
      out.push([
        levelRadius * Math.cos(theta),
        levelRadius * Math.sin(theta),
        zCut,
      ]);
    }
    const first = out[0];
    const last = out[out.length - 1];
    if (Math.abs(first[0] - last[0]) > 1e-9 || Math.abs(first[1] - last[1]) > 1e-9) {
      out.push([...first]);
    }
    return out;
  }

  const lineCount = Math.max(2, Math.ceil((2 * straightHalfLen) / ds) + 1);
  const arcCount = Math.max(16, Math.ceil((Math.PI * levelRadius) / ds) + 1);
  const xVals = [];
  const sVals = [];

  for (let idx = 0; idx < lineCount; idx += 1) {
    const t = idx / (lineCount - 1);
    xVals.push(straightHalfLen + (-2 * straightHalfLen) * t);
    sVals.push(-levelRadius);
  }

  for (let idx = 1; idx < arcCount; idx += 1) {
    const theta = -0.5 * Math.PI + (Math.PI * idx) / (arcCount - 1);
    xVals.push(-straightHalfLen - levelRadius * Math.cos(theta));
    sVals.push(levelRadius * Math.sin(theta));
  }

  for (let idx = 1; idx < lineCount; idx += 1) {
    const t = idx / (lineCount - 1);
    xVals.push(-straightHalfLen + (2 * straightHalfLen) * t);
    sVals.push(levelRadius);
  }

  for (let idx = 1; idx < arcCount; idx += 1) {
    const theta = 0.5 * Math.PI + (-Math.PI * idx) / (arcCount - 1);
    xVals.push(straightHalfLen + levelRadius * Math.cos(theta));
    sVals.push(levelRadius * Math.sin(theta));
  }

  const loop = xVals.map((xValue, idx) => [xValue, sVals[idx], zCut]);
  const first = loop[0];
  const last = loop[loop.length - 1];
  if (Math.abs(first[0] - last[0]) > 1e-9 || Math.abs(first[1] - last[1]) > 1e-9) {
    loop.push([...first]);
  }
  return loop;
}

function rotateXSPath(path, angleDeg) {
  const angleRad = degToRad(Number(angleDeg));
  if (Math.abs(angleRad) <= 1e-12) {
    return path.map((point) => [...point]);
  }

  const c = Math.cos(angleRad);
  const s = Math.sin(angleRad);
  return path.map(([x, sv, z]) => [c * x - s * sv, s * x + c * sv, z]);
}

function rotateXSBoundary(boundary, angleDeg) {
  const angleRad = degToRad(Number(angleDeg));
  if (Math.abs(angleRad) <= 1e-12) {
    return boundary.map((point) => [...point]);
  }

  const c = Math.cos(angleRad);
  const s = Math.sin(angleRad);
  return boundary.map(([x, sv]) => [c * x - s * sv, s * x + c * sv]);
}

function estimateMaxCircumferentialSpan(params) {
  const radius = Number(params.pocketEndRadius);
  const straightHalfLen = Math.max(0, 0.5 * params.pocketLength - radius);
  const angleRad = degToRad(params.oblongAngleDeg);
  const c = Math.cos(angleRad);
  const s = Math.sin(angleRad);

  if (straightHalfLen <= 1e-12) {
    return 2 * radius;
  }

  const tSamples = Math.max(220, Math.ceil(straightHalfLen * 44));
  const xCenters = [];
  const sCenters = [];
  for (let idx = 0; idx < tSamples; idx += 1) {
    const t = -straightHalfLen + (2 * straightHalfLen * idx) / (tSamples - 1);
    xCenters.push(t * c);
    sCenters.push(t * s);
  }

  const xMin = Math.min(...xCenters) - radius;
  const xMax = Math.max(...xCenters) + radius;
  const xSamples = Math.max(320, Math.ceil((xMax - xMin) / 0.03));

  let best = 2 * radius;
  for (let idx = 0; idx < xSamples; idx += 1) {
    const xValue = xMin + ((xMax - xMin) * idx) / (xSamples - 1);
    let low = Infinity;
    let high = -Infinity;
    for (let centerIdx = 0; centerIdx < xCenters.length; centerIdx += 1) {
      const dx = xValue - xCenters[centerIdx];
      if (Math.abs(dx) > radius) {
        continue;
      }
      const root = Math.sqrt(Math.max(radius * radius - dx * dx, 0));
      low = Math.min(low, sCenters[centerIdx] - root);
      high = Math.max(high, sCenters[centerIdx] + root);
    }
    if (Number.isFinite(low) && Number.isFinite(high)) {
      best = Math.max(best, high - low);
    }
  }
  return best;
}

function validateGeometry(params) {
  const toolRadius = 0.5 * params.toolDiameter;
  if (params.shaftDiameter <= 0) {
    throw new Error("Shaft diameter must be positive.");
  }
  if (params.pocketDepth <= 0) {
    throw new Error("Pocket depth must be positive.");
  }
  if (params.pocketDepth >= 0.5 * params.shaftDiameter) {
    throw new Error("Pocket depth must stay smaller than the shaft radius.");
  }
  if (params.pocketEndRadius <= toolRadius + 1e-6) {
    throw new Error("Pocket end radius must be larger than the tool radius.");
  }
  if (params.pocketLength <= 2 * toolRadius + 1e-6) {
    throw new Error("Pocket length is too short for this tool diameter.");
  }
  if (params.pocketLength + 1e-9 < 2 * params.pocketEndRadius) {
    throw new Error("Pocket length must be at least 2x the end radius.");
  }

  const count = Math.max(1, Math.round(params.patternCount));
  if (count > 1) {
    const circumference = Math.PI * params.shaftDiameter;
    const pitch = circumference / count;
    const pocketWidth = estimateMaxCircumferentialSpan(params);
    if (pitch <= pocketWidth + 0.05) {
      throw new Error("Pattern count is too high. Pockets overlap around the shaft.");
    }
  }
}

function generateSinglePocketToolpath(params) {
  validateGeometry(params);
  const toolRadius = 0.5 * params.toolDiameter;
  const straightHalfLen = Math.max(0, 0.5 * params.pocketLength - params.pocketEndRadius);
  const innerRadius = params.pocketEndRadius - toolRadius;
  const zCut = -Math.abs(params.pocketDepth);
  const levels = buildSpiralLevels(innerRadius, params.stepOver);

  let segment = sampleCenterline(straightHalfLen, zCut, params.pathPointSpacing);
  for (let idx = 1; idx < levels.length; idx += 1) {
    const level = levels[idx];
    const connector = sampleLinearMove(
      segment[segment.length - 1][0],
      segment[segment.length - 1][1],
      straightHalfLen,
      -level,
      zCut,
      params.pathPointSpacing,
    );
    const loop = sampleObroundLoop(level, straightHalfLen, zCut, params.pathPointSpacing);
    segment = segment.concat(connector.slice(1), loop.slice(1));
  }

  if (segment.length < 2) {
    throw new Error("No valid pocket spiral generated.");
  }

  segment = rotateXSPath(segment, params.oblongAngleDeg);
  return segment.map(([x, s, z]) => [x + params.axialCenter, s, z]);
}

function generateToolpathSegments(params) {
  const baseSegment = generateSinglePocketToolpath(params);
  const shaftRadius = 0.5 * params.shaftDiameter;
  const count = Math.max(1, Math.round(params.patternCount));
  const segments = [];

  for (let idx = 0; idx < count; idx += 1) {
    const angleDeg = params.patternStartAngle + (360 * idx) / count;
    const sOffset = shaftRadius * degToRad(angleDeg);
    segments.push(baseSegment.map(([x, s, z]) => [x, s + sOffset, z]));
  }

  return segments;
}

function singlePocketBoundary(params) {
  const radius = Number(params.pocketEndRadius);
  const straightHalfLen = Math.max(0, 0.5 * params.pocketLength - radius);
  const perimeter = Math.max(1e-6, 4 * straightHalfLen + TWO_PI * radius);
  const spacing = Math.max(0.05, perimeter / Math.max(40, 160));
  const loop = sampleObroundLoop(radius, straightHalfLen, 0, spacing);
  const boundary = rotateXSBoundary(loop.map(([x, s]) => [x, s]), params.oblongAngleDeg);
  return boundary.map(([x, s]) => [x + params.axialCenter, s]);
}

function patternBoundaries(params) {
  validateGeometry(params);
  const baseBoundary = singlePocketBoundary(params);
  const shaftRadius = 0.5 * params.shaftDiameter;
  const count = Math.max(1, Math.round(params.patternCount));
  const boundaries = [];

  for (let idx = 0; idx < count; idx += 1) {
    const angleDeg = params.patternStartAngle + (360 * idx) / count;
    const sOffset = shaftRadius * degToRad(angleDeg);
    boundaries.push(baseBoundary.map(([x, s]) => [x, s + sOffset]));
  }
  return boundaries;
}

function spiralLevelCount(params) {
  const toolRadius = 0.5 * params.toolDiameter;
  const innerRadius = Math.max(0, params.pocketEndRadius - toolRadius);
  return buildSpiralLevels(innerRadius, params.stepOver).length;
}

function pointInsidePocketLocal(localX, localS, params) {
  const radius = params.pocketEndRadius;
  const straightHalfLen = Math.max(0, 0.5 * params.pocketLength - radius);
  if (Math.abs(localX) <= straightHalfLen) {
    return Math.abs(localS) <= radius + 1e-9;
  }
  const capCenterX = localX < 0 ? -straightHalfLen : straightHalfLen;
  return Math.hypot(localX - capCenterX, localS) <= radius + 1e-9;
}

function pointInsidePocketWorld(x, s, params, sCenter, circumference) {
  const angleRad = degToRad(params.oblongAngleDeg);
  const c = Math.cos(angleRad);
  const sinAngle = Math.sin(angleRad);
  const dx = x - params.axialCenter;
  const ds = wrapArcLength(s - sCenter, circumference);
  const localX = c * dx + sinAngle * ds;
  const localS = -sinAngle * dx + c * ds;
  return pointInsidePocketLocal(localX, localS, params);
}

function pointInsideAnyPocket(x, s, params, sCenters, circumference) {
  for (const sCenter of sCenters) {
    if (pointInsidePocketWorld(x, s, params, sCenter, circumference)) {
      return true;
    }
  }
  return false;
}

function computePocketCenters(params) {
  const count = Math.max(1, Math.round(params.patternCount));
  const shaftRadius = 0.5 * params.shaftDiameter;
  const out = [];
  for (let idx = 0; idx < count; idx += 1) {
    const angleDeg = params.patternStartAngle + (360 * idx) / count;
    out.push(shaftRadius * degToRad(angleDeg));
  }
  return out;
}

function computeBoundaryBounds(boundary) {
  let xMin = Infinity;
  let xMax = -Infinity;
  let sMin = Infinity;
  let sMax = -Infinity;

  for (const [x, s] of boundary) {
    xMin = Math.min(xMin, x);
    xMax = Math.max(xMax, x);
    sMin = Math.min(sMin, s);
    sMax = Math.max(sMax, s);
  }

  return { xMin, xMax, sMin, sMax };
}

function chooseOuterSurfaceSeam(pocketCenters, circumference) {
  if (pocketCenters.length === 0) {
    return 0;
  }

  const ordered = pocketCenters
    .map((value) => (((value % circumference) + circumference) % circumference))
    .sort((left, right) => left - right);

  let bestGap = -Infinity;
  let seam = 0;
  for (let idx = 0; idx < ordered.length; idx += 1) {
    const current = ordered[idx];
    const next = idx === ordered.length - 1 ? ordered[0] + circumference : ordered[idx + 1];
    const gap = next - current;
    if (gap > bestGap) {
      bestGap = gap;
      seam = current + 0.5 * gap;
    }
  }

  return seam % circumference;
}

function circularDistance(valueA, valueB, period) {
  const delta = Math.abs(valueA - valueB) % period;
  return Math.min(delta, period - delta);
}

function chooseBackFacingOuterSeam(boundaries, pocketCenters, shaftRadius, circumference) {
  const blockedIntervals = boundaries
    .map((boundary, idx) => {
      const loop = unwrapBoundaryAroundCenter(boundary, pocketCenters[idx], 0, circumference);
      const bounds = computeBoundaryBounds(loop);
      return [bounds.sMin, bounds.sMax];
    });

  const gaps = complementIntervals(blockedIntervals, 0, circumference);
  if (gaps.length === 0) {
    return chooseOuterSurfaceSeam(pocketCenters, circumference);
  }

  const defaultViewTheta = Math.atan2(2.15, 2.55);
  const hiddenTheta = (defaultViewTheta + Math.PI) % TWO_PI;
  const preferredS = shaftRadius * hiddenTheta;

  let bestSeam = null;
  let bestDistance = Infinity;
  for (const [gapStart, gapEnd] of gaps) {
    const gapWidth = gapEnd - gapStart;
    if (gapWidth <= 1e-9) {
      continue;
    }

    const margin = Math.min(gapWidth * 0.2, Math.max(0.5, gapWidth * 0.08));
    const innerStart = gapStart + Math.min(margin, 0.5 * gapWidth);
    const innerEnd = gapEnd - Math.min(margin, 0.5 * gapWidth);
    const candidate = innerStart <= preferredS && preferredS <= innerEnd
      ? preferredS
      : midpoint(gapStart, gapEnd);
    const distance = circularDistance(candidate, preferredS, circumference);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestSeam = candidate;
    }
  }

  return bestSeam ?? chooseOuterSurfaceSeam(pocketCenters, circumference);
}

function midpoint(valueA, valueB) {
  return 0.5 * (valueA + valueB);
}

function chooseSecondaryOuterSeam(holeIntervals, seamStart, circumference) {
  const rangeStart = seamStart;
  const rangeEnd = seamStart + circumference;
  const sorted = holeIntervals
    .map((interval) => ({ ...interval }))
    .sort((left, right) => left.sMin - right.sMin);

  const gaps = [];
  let cursor = rangeStart;
  for (const interval of sorted) {
    if (interval.sMin > cursor + 1e-9) {
      gaps.push({ start: cursor, end: interval.sMin });
    }
    cursor = Math.max(cursor, interval.sMax);
  }
  if (cursor < rangeEnd - 1e-9) {
    gaps.push({ start: cursor, end: rangeEnd });
  }

  const validGaps = gaps.filter((gap) => gap.end - gap.start > 1e-6);
  if (validGaps.length === 0) {
    return seamStart + 0.5 * circumference;
  }

  const target = seamStart + 0.5 * circumference;
  let bestGap = validGaps[0];
  let bestDistance = Infinity;
  for (const gap of validGaps) {
    const center = midpoint(gap.start, gap.end);
    const distance = Math.abs(center - target);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestGap = gap;
    }
  }

  return midpoint(bestGap.start, bestGap.end);
}

// CAD-like source model: explicit cylindrical floor/outer surfaces plus radial walls.
function buildParametricPocketPart(params) {
  validateGeometry(params);

  const partParams = cloneParams(params);
  const shaftRadius = 0.5 * partParams.shaftDiameter;
  const floorRadius = shaftRadius - partParams.pocketDepth;
  const circumference = TWO_PI * shaftRadius;
  const xMargin = Math.max(6, 0.35 * partParams.pocketLength);
  const xMin = partParams.axialCenter - 0.5 * partParams.pocketLength - xMargin;
  const xMax = partParams.axialCenter + 0.5 * partParams.pocketLength + xMargin;
  const pocketCenters = computePocketCenters(partParams);
  const boundaries = patternBoundaries(partParams);
  const seamStart = chooseBackFacingOuterSeam(
    boundaries,
    pocketCenters,
    shaftRadius,
    circumference,
  );

  const pockets = boundaries.map((boundary, idx) => {
    const bounds = computeBoundaryBounds(boundary);
    return {
      kind: "pocket",
      centerS: pocketCenters[idx],
      boundary,
      floorSurface: {
        kind: "trimmed-cylinder",
        role: "floor",
        radius: floorRadius,
        centerS: pocketCenters[idx],
        boundary,
        xMin: bounds.xMin,
        xMax: bounds.xMax,
        sMin: bounds.sMin,
        sMax: bounds.sMax,
      },
      wallSurface: {
        kind: "radial-wall",
        role: "wall",
        outerRadius: shaftRadius,
        innerRadius: floorRadius,
        boundary,
      },
    };
  });

  const outerHoleLoops = pockets
    .map((pocket) => ({
      pocket,
      loop: unwrapBoundaryAroundCenter(pocket.boundary, pocket.centerS, seamStart, circumference),
    }))
    .map((item) => {
      const bounds = computeBoundaryBounds(item.loop);
      return {
        ...item,
        sMin: bounds.sMin,
        sMax: bounds.sMax,
      };
    });

  const outerSurfaces = [
    {
      kind: "trimmed-cylinder",
      role: "outer",
      radius: shaftRadius,
      xMin,
      xMax,
      sStart: seamStart,
      sEnd: seamStart + circumference,
      holes: outerHoleLoops.map((loop) => loop.loop),
    },
  ];

  return {
    kind: "parametric-pocket-shaft",
    params: partParams,
    shaftRadius,
    floorRadius,
    circumference,
    pocketCenters,
    xRange: { min: xMin, max: xMax },
    outerSurfaces,
    endCaps: [
      {
        kind: "disk-cap",
        role: "endcap",
        x: xMin,
        radius: shaftRadius,
        normalSign: -1,
      },
      {
        kind: "disk-cap",
        role: "endcap",
        x: xMax,
        radius: shaftRadius,
        normalSign: 1,
      },
    ],
    pockets,
  };
}

function shaftPointFromUnwrapped(x, s, shaftRadius, surfaceRadius) {
  const theta = s / shaftRadius;
  return [x, surfaceRadius * Math.cos(theta), surfaceRadius * Math.sin(theta)];
}

function unwrapToCylinderPoint(point, shaftRadius, radialLift = 0) {
  const [x, s, z] = point;
  const theta = s / shaftRadius;
  const radius = shaftRadius + z + radialLift;
  return [x, radius * Math.cos(theta), radius * Math.sin(theta)];
}

function boundaryToCylinderPoint(point, shaftRadius, radialLift = 0.05) {
  const [x, s] = point;
  const theta = s / shaftRadius;
  const radius = shaftRadius + radialLift;
  return [x, radius * Math.cos(theta), radius * Math.sin(theta)];
}

function makeLine(points, color, loop = false) {
  const geometry = new THREE.BufferGeometry();
  const flat = new Float32Array(points.length * 3);
  points.forEach((point, idx) => {
    flat[idx * 3 + 0] = point[0];
    flat[idx * 3 + 1] = point[1];
    flat[idx * 3 + 2] = point[2];
  });
  geometry.setAttribute("position", new THREE.BufferAttribute(flat, 3));
  const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.98 });
  return loop ? new THREE.LineLoop(geometry, material) : new THREE.Line(geometry, material);
}

function sampleCountForSpan(span, totalSpan, totalSamples, minimum) {
  const ratio = Math.max(1e-6, span) / Math.max(1e-6, totalSpan);
  return Math.max(minimum, Math.round(totalSamples * ratio));
}

function sanitizeBoundaryLoop(boundary) {
  if (boundary.length <= 1) {
    return boundary.slice();
  }

  const out = boundary.map(([x, s]) => [x, s]);
  const first = out[0];
  const last = out[out.length - 1];
  if (Math.abs(first[0] - last[0]) <= 1e-9 && Math.abs(first[1] - last[1]) <= 1e-9) {
    out.pop();
  }
  return out;
}

function buildLinearSamples(start, end, count) {
  if (count <= 1 || Math.abs(end - start) <= 1e-9) {
    return [start];
  }

  const out = [];
  for (let idx = 0; idx < count; idx += 1) {
    const t = idx / (count - 1);
    out.push(start + (end - start) * t);
  }
  return out;
}

function mergeSortedSamples(values, decimals = 6) {
  const ordered = [...values].sort((left, right) => left - right);
  const out = [];
  for (const value of ordered) {
    const rounded = Number(value.toFixed(decimals));
    if (out.length === 0 || Math.abs(rounded - out[out.length - 1]) > 1e-6) {
      out.push(rounded);
    }
  }
  return out;
}

function normalizeLoopOrientation(loop, clockwise) {
  const vectors = loop.map(([x, s]) => new THREE.Vector2(x, s));
  const isClockWise = THREE.ShapeUtils.isClockWise(vectors);
  if (isClockWise === clockwise) {
    return loop;
  }
  return loop.slice().reverse();
}

function unwrapBoundaryAroundCenter(boundary, centerS, seamStart, circumference) {
  let targetCenter = centerS;
  while (targetCenter < seamStart) {
    targetCenter += circumference;
  }
  while (targetCenter >= seamStart + circumference) {
    targetCenter -= circumference;
  }

  return sanitizeBoundaryLoop(boundary).map(([x, s]) => {
    const ds = wrapArcLength(s - centerS, circumference);
    return [x, targetCenter + ds];
  });
}

function mergeIntervals(intervals) {
  const ordered = intervals
    .filter(([start, end]) => end > start + 1e-9)
    .sort((left, right) => left[0] - right[0]);
  if (ordered.length === 0) {
    return [];
  }

  const merged = [ordered[0].slice()];
  for (let idx = 1; idx < ordered.length; idx += 1) {
    const [start, end] = ordered[idx];
    const last = merged[merged.length - 1];
    if (start <= last[1] + 1e-9) {
      last[1] = Math.max(last[1], end);
    } else {
      merged.push([start, end]);
    }
  }
  return merged;
}

function complementIntervals(intervals, start, end) {
  const merged = mergeIntervals(intervals);
  const out = [];
  let cursor = start;
  for (const [intervalStart, intervalEnd] of merged) {
    if (intervalStart > cursor + 1e-9) {
      out.push([cursor, intervalStart]);
    }
    cursor = Math.max(cursor, intervalEnd);
  }
  if (cursor < end - 1e-9) {
    out.push([cursor, end]);
  }
  return out;
}

function intersectLoopWithVerticalLine(loop, x) {
  const cleanLoop = sanitizeBoundaryLoop(loop);
  const sHits = [];

  for (let idx = 0; idx < cleanLoop.length; idx += 1) {
    const start = cleanLoop[idx];
    const end = cleanLoop[(idx + 1) % cleanLoop.length];
    const [x1, s1] = start;
    const [x2, s2] = end;

    if (Math.abs(x2 - x1) <= 1e-9) {
      continue;
    }

    const low = Math.min(x1, x2);
    const high = Math.max(x1, x2);
    if (!(x >= low && x < high)) {
      continue;
    }

    const t = (x - x1) / (x2 - x1);
    if (t < -1e-9 || t > 1 + 1e-9) {
      continue;
    }
    sHits.push(s1 + (s2 - s1) * t);
  }

  sHits.sort((left, right) => left - right);
  const intervals = [];
  for (let idx = 0; idx + 1 < sHits.length; idx += 2) {
    intervals.push([sHits[idx], sHits[idx + 1]]);
  }
  return intervals;
}

function outerSurfaceFreeIntervalsAtX(surface, x) {
  const blocked = [];
  for (const hole of surface.holes) {
    blocked.push(...intersectLoopWithVerticalLine(hole, x));
  }
  return complementIntervals(blocked, surface.sStart, surface.sEnd);
}

function buildOuterSurfaceXValues(surface, part) {
  const xSegments = Math.max(26, Math.round(part.params.meshAxialSamples / 5));
  return mergeSortedSamples([
    ...buildLinearSamples(surface.xMin, surface.xMax, xSegments + 1),
    ...surface.holes.flat().map(([x]) => x),
  ]);
}

function buildOuterStripBandVertices(xLeft, xRight, leftBand, rightBand, sSegments, part, radius) {
  const positions = [];
  const normals = [];

  for (let idx = 0; idx <= sSegments; idx += 1) {
    const t = sSegments === 0 ? 0 : idx / sSegments;
    const sLeft = leftBand[0] + (leftBand[1] - leftBand[0]) * t;
    const sRight = rightBand[0] + (rightBand[1] - rightBand[0]) * t;
    const leftPoint = shaftPointFromUnwrapped(xLeft, sLeft, part.shaftRadius, radius);
    const rightPoint = shaftPointFromUnwrapped(xRight, sRight, part.shaftRadius, radius);
    const leftTheta = sLeft / part.shaftRadius;
    const rightTheta = sRight / part.shaftRadius;

    positions.push(...leftPoint, ...rightPoint);
    normals.push(0, Math.cos(leftTheta), Math.sin(leftTheta));
    normals.push(0, Math.cos(rightTheta), Math.sin(rightTheta));
  }

  return { positions, normals };
}

// Display stage: tessellate each analytic surface separately to preserve hard breaks.
function meshTrimmedCylinderSurface(surface, part) {
  const params = part.params;
  const totalXSpan = part.xRange.max - part.xRange.min;
  const xSpan = surface.xMax - surface.xMin;
  const sSpan = surface.sMax - surface.sMin;
  const xSegments = surface.role === "outer"
    ? Math.max(80, Math.round(params.meshAxialSamples))
    : sampleCountForSpan(xSpan, totalXSpan, params.meshAxialSamples, 28);
  const sSegments = surface.role === "outer"
    ? Math.max(128, Math.round(params.meshAngularSamples))
    : sampleCountForSpan(sSpan, part.circumference, params.meshAngularSamples, 28);

  const rowStride = xSegments + 1;
  const vertexCount = (sSegments + 1) * rowStride;
  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
  const indices = [];

  let vertexIndex = 0;
  for (let sIdx = 0; sIdx <= sSegments; sIdx += 1) {
    const sT = sSegments === 0 ? 0 : sIdx / sSegments;
    const s = surface.sMin + (surface.sMax - surface.sMin) * sT;
    const theta = s / part.shaftRadius;
    const cosTheta = Math.cos(theta);
    const sinTheta = Math.sin(theta);
    for (let xIdx = 0; xIdx <= xSegments; xIdx += 1) {
      const xT = xSegments === 0 ? 0 : xIdx / xSegments;
      const x = surface.xMin + (surface.xMax - surface.xMin) * xT;
      const point = shaftPointFromUnwrapped(x, s, part.shaftRadius, surface.radius);

      positions[vertexIndex * 3 + 0] = point[0];
      positions[vertexIndex * 3 + 1] = point[1];
      positions[vertexIndex * 3 + 2] = point[2];
      normals[vertexIndex * 3 + 0] = 0;
      normals[vertexIndex * 3 + 1] = cosTheta;
      normals[vertexIndex * 3 + 2] = sinTheta;
      vertexIndex += 1;
    }
  }

  for (let sIdx = 0; sIdx < sSegments; sIdx += 1) {
    const sMid = surface.sMin + (surface.sMax - surface.sMin) * ((sIdx + 0.5) / sSegments);
    for (let xIdx = 0; xIdx < xSegments; xIdx += 1) {
      const xMid = surface.xMin + (surface.xMax - surface.xMin) * ((xIdx + 0.5) / xSegments);
      const includeCell = surface.role === "outer"
        ? !pointInsideAnyPocket(xMid, sMid, params, part.pocketCenters, part.circumference)
        : pointInsidePocketWorld(xMid, sMid, params, surface.centerS, part.circumference);
      if (!includeCell) {
        continue;
      }

      const a = sIdx * rowStride + xIdx;
      const b = a + 1;
      const c = a + rowStride;
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  geometry.setIndex(indices);
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

function meshOuterCylinderSurface(surface, part) {
  const xValues = buildOuterSurfaceXValues(surface, part);
  const positions = [];
  const normals = [];
  const indices = [];
  let vertexBase = 0;

  for (let stripIdx = 0; stripIdx < xValues.length - 1; stripIdx += 1) {
    const xLeft = xValues[stripIdx];
    const xRight = xValues[stripIdx + 1];
    const stripWidth = xRight - xLeft;
    if (stripWidth <= 1e-9) {
      continue;
    }

    const epsilon = Math.min(1e-5, stripWidth * 0.25);
    const leftEval = Math.min(xRight - 1e-9, xLeft + epsilon);
    const rightEval = Math.max(xLeft + 1e-9, xRight - epsilon);
    const midEval = 0.5 * (xLeft + xRight);

    let leftBands = outerSurfaceFreeIntervalsAtX(surface, leftEval);
    let rightBands = outerSurfaceFreeIntervalsAtX(surface, rightEval);
    const midBands = outerSurfaceFreeIntervalsAtX(surface, midEval);

    if (midBands.length === 0) {
      continue;
    }

    if (leftBands.length !== midBands.length) {
      leftBands = midBands.map((band) => [...band]);
    }
    if (rightBands.length !== midBands.length) {
      rightBands = midBands.map((band) => [...band]);
    }

    const bandCount = Math.min(leftBands.length, rightBands.length, midBands.length);
    for (let bandIdx = 0; bandIdx < bandCount; bandIdx += 1) {
      const leftBand = leftBands[bandIdx];
      const rightBand = rightBands[bandIdx];
      const width = Math.max(leftBand[1] - leftBand[0], rightBand[1] - rightBand[0]);
      if (width <= 1e-9) {
        continue;
      }

      const sSegments = sampleCountForSpan(width, part.circumference, part.params.meshAngularSamples, 3);
      const strip = buildOuterStripBandVertices(
        xLeft,
        xRight,
        leftBand,
        rightBand,
        sSegments,
        part,
        surface.radius,
      );

      positions.push(...strip.positions);
      normals.push(...strip.normals);

      for (let segIdx = 0; segIdx < sSegments; segIdx += 1) {
        const a = vertexBase + segIdx * 2;
        const b = a + 1;
        const c = a + 2;
        const d = a + 3;
        indices.push(a, c, b, b, c, d);
      }
      vertexBase += (sSegments + 1) * 2;
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(positions), 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(normals), 3));
  geometry.setIndex(indices);
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

function meshPocketFloorSurface(surface, part) {
  const boundary = sanitizeBoundaryLoop(surface.boundary);
  const shape = boundary.map(([x, s]) => new THREE.Vector2(x, s));
  const triangles = THREE.ShapeUtils.triangulateShape(shape, []);
  const positions = new Float32Array(boundary.length * 3);
  const normals = new Float32Array(boundary.length * 3);

  boundary.forEach(([x, s], idx) => {
    const point = shaftPointFromUnwrapped(x, s, part.shaftRadius, surface.radius);
    const theta = s / part.shaftRadius;
    positions[idx * 3 + 0] = point[0];
    positions[idx * 3 + 1] = point[1];
    positions[idx * 3 + 2] = point[2];
    normals[idx * 3 + 0] = 0;
    normals[idx * 3 + 1] = Math.cos(theta);
    normals[idx * 3 + 2] = Math.sin(theta);
  });

  const indices = [];
  triangles.forEach(([a, b, c]) => {
    indices.push(a, c, b);
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  geometry.setIndex(indices);
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

function meshEndCapSurface(surface, part) {
  const angularSegments = Math.max(96, Math.round(part.params.meshAngularSamples));
  const vertexCount = angularSegments + 2;
  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
  const indices = [];

  positions[0] = surface.x;
  positions[1] = 0;
  positions[2] = 0;
  normals[0] = surface.normalSign;
  normals[1] = 0;
  normals[2] = 0;

  for (let idx = 0; idx <= angularSegments; idx += 1) {
    const theta = (TWO_PI * idx) / angularSegments;
    const vertexIndex = idx + 1;
    positions[vertexIndex * 3 + 0] = surface.x;
    positions[vertexIndex * 3 + 1] = surface.radius * Math.cos(theta);
    positions[vertexIndex * 3 + 2] = surface.radius * Math.sin(theta);
    normals[vertexIndex * 3 + 0] = surface.normalSign;
    normals[vertexIndex * 3 + 1] = 0;
    normals[vertexIndex * 3 + 2] = 0;
  }

  for (let idx = 1; idx <= angularSegments; idx += 1) {
    if (surface.normalSign > 0) {
      indices.push(0, idx, idx + 1);
    } else {
      indices.push(0, idx + 1, idx);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  geometry.setIndex(indices);
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

function meshRadialWallSurface(surface, part) {
  const radialSegments = 1;
  const boundary = surface.boundary;
  const boundaryCount = boundary.length;
  const ringStride = radialSegments + 1;
  const vertexCount = boundaryCount * ringStride;
  const positions = new Float32Array(vertexCount * 3);
  const indices = [];

  let vertexIndex = 0;
  for (const [x, s] of boundary) {
    for (let radialIdx = 0; radialIdx <= radialSegments; radialIdx += 1) {
      const t = radialSegments === 0 ? 0 : radialIdx / radialSegments;
      const radius = surface.outerRadius + (surface.innerRadius - surface.outerRadius) * t;
      const point = shaftPointFromUnwrapped(x, s, part.shaftRadius, radius);
      positions[vertexIndex * 3 + 0] = point[0];
      positions[vertexIndex * 3 + 1] = point[1];
      positions[vertexIndex * 3 + 2] = point[2];
      vertexIndex += 1;
    }
  }

  for (let boundaryIdx = 0; boundaryIdx < boundaryCount - 1; boundaryIdx += 1) {
    for (let radialIdx = 0; radialIdx < radialSegments; radialIdx += 1) {
      const a = boundaryIdx * ringStride + radialIdx;
      const b = a + 1;
      const c = a + ringStride;
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

function meshParametricPocketPart(part) {
  const surfaces = part.outerSurfaces.map((outerSurface) => ({
    role: "outer",
    geometry: meshOuterCylinderSurface(outerSurface, part),
  }));

  for (const endCap of part.endCaps) {
    surfaces.push({
      role: "endcap",
      geometry: meshEndCapSurface(endCap, part),
    });
  }

  for (const pocket of part.pockets) {
    surfaces.push({
      role: "floor",
      geometry: meshPocketFloorSurface(pocket.floorSurface, part),
    });
    surfaces.push({
      role: "wall",
      geometry: meshRadialWallSurface(pocket.wallSurface, part),
    });
  }

  return surfaces;
}

function formatValue(value, decimals = 2) {
  return Number(value).toFixed(decimals).replace(/\.?0+$/, "");
}

function buildFinishMaterial(role, finish) {
  const presets = {
    matte: {
      outer: { color: 0xb8c3cb, metalness: 0.28, roughness: 0.72, envMapIntensity: 0.65 },
      pocket: { color: 0xa9b8c2, metalness: 0.24, roughness: 0.76, envMapIntensity: 0.6 },
      endcap: { color: 0xb4c0c8, metalness: 0.26, roughness: 0.7, envMapIntensity: 0.62 },
    },
    shiny: {
      outer: { color: 0xb8c3cb, metalness: 0.8, roughness: 0.16, envMapIntensity: 1.0 },
      pocket: { color: 0xa9b8c2, metalness: 0.74, roughness: 0.22, envMapIntensity: 1.0 },
      endcap: { color: 0xb4c0c8, metalness: 0.72, roughness: 0.2, envMapIntensity: 0.98 },
    },
  };

  const family = presets[finish] ?? presets.matte;
  return new THREE.MeshStandardMaterial(family[role]);
}

class CylindricalOblongPocketViewer extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._params = cloneParams(DEFAULT_PARAMS);
    this._inputs = new Map();
    this._isConnected = false;
    this._rebuildTimer = null;
    this._animationFrame = 0;
    this._resizeObserver = null;
    this._firstFrame = true;
    this._scene = null;
    this._renderer = null;
    this._camera = null;
    this._controls = null;
    this._modelGroup = null;
  }

  connectedCallback() {
    if (this._isConnected) {
      return;
    }
    this._isConnected = true;
    this._renderShell();
    this._initScene();
    this._bindControls();
    this._syncInputsFromParams();
    this._scheduleRebuild();
    this._startRenderLoop();
  }

  disconnectedCallback() {
    this._isConnected = false;
    window.clearTimeout(this._rebuildTimer);
    cancelAnimationFrame(this._animationFrame);
    this._resizeObserver?.disconnect();
    this._controls?.dispose();
    this._renderer?.dispose();
  }

  get params() {
    return cloneParams(this._params);
  }

  set params(nextParams) {
    this._params = { ...this._params, ...nextParams };
    this._syncInputsFromParams();
    this._scheduleRebuild();
  }

  _renderShell() {
    const sectionMarkup = CONTROL_SECTIONS.map(
      (section) => `
        <section class="panel-section">
          <div class="section-kicker">${section.title}</div>
          ${section.fields
            .map(
              ([key, label, min, max, step, unit]) => `
                <label class="control">
                  <span class="control-head">
                    <span>${label}</span>
                    <span class="unit">${unit}</span>
                  </span>
                  <div class="control-body">
                    <input data-kind="range" data-key="${key}" type="range" min="${min}" max="${max}" step="${step}">
                    <input data-kind="number" data-key="${key}" type="number" min="${min}" max="${max}" step="${step}">
                  </div>
                </label>
              `,
            )
            .join("")}
        </section>
      `,
    ).join("");

    this.shadowRoot.innerHTML = `
      <style>
        :host {
          display: block;
          color: #f7f5ed;
          --panel-bg: linear-gradient(180deg, rgba(14, 24, 33, 0.92), rgba(8, 15, 22, 0.96));
          --card-bg: linear-gradient(180deg, rgba(21, 35, 46, 0.88), rgba(10, 17, 24, 0.9));
          --accent: #ff8b3d;
          --accent-soft: rgba(255, 139, 61, 0.18);
          --steel: #aebcc7;
          --line: rgba(194, 210, 220, 0.14);
          --warn: #ffc857;
          font-family: "Avenir Next", "Segoe UI", sans-serif;
        }

        * {
          box-sizing: border-box;
        }

        .shell {
          min-height: 820px;
          padding: 24px;
          border-radius: 28px;
          background:
            radial-gradient(circle at top left, rgba(255, 139, 61, 0.18), transparent 30%),
            radial-gradient(circle at 75% 10%, rgba(74, 136, 188, 0.16), transparent 28%),
            linear-gradient(135deg, #091017, #101b25 48%, #0a0f14);
          border: 1px solid rgba(255, 255, 255, 0.06);
          box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.08),
            0 24px 70px rgba(0, 0, 0, 0.38);
        }

        .grid {
          display: grid;
          grid-template-columns: minmax(320px, 400px) minmax(0, 1fr);
          gap: 18px;
          min-height: 772px;
        }

        .panel {
          display: flex;
          flex-direction: column;
          gap: 16px;
          padding: 22px;
          background: var(--panel-bg);
          border-radius: 24px;
          border: 1px solid rgba(255, 255, 255, 0.06);
          backdrop-filter: blur(14px);
        }

        .eyebrow {
          display: inline-flex;
          align-self: flex-start;
          padding: 6px 10px;
          border-radius: 999px;
          color: #ffe5d1;
          background: rgba(255, 139, 61, 0.12);
          border: 1px solid rgba(255, 139, 61, 0.22);
          font-size: 12px;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }

        h2 {
          margin: 0;
          font-size: clamp(26px, 3vw, 38px);
          line-height: 0.95;
          font-weight: 700;
          letter-spacing: -0.04em;
        }

        .lede {
          margin: 0;
          color: #cbd7df;
          font-size: 14px;
          line-height: 1.5;
        }

        .panel-section,
        .stats,
        .toggles {
          padding: 16px;
          border-radius: 18px;
          background: var(--card-bg);
          border: 1px solid var(--line);
        }

        .section-kicker,
        .stats-title {
          display: block;
          margin-bottom: 12px;
          color: #ffd5ba;
          font-size: 12px;
          letter-spacing: 0.11em;
          text-transform: uppercase;
        }

        .stats-title {
          grid-column: 1 / -1;
          margin-bottom: 0;
        }

        .control {
          display: block;
          margin-bottom: 14px;
        }

        .control:last-child {
          margin-bottom: 0;
        }

        .control-head {
          display: flex;
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 8px;
          font-size: 13px;
          color: #eaf0f4;
        }

        .unit {
          color: #8ea3b2;
        }

        .control-body {
          display: grid;
          grid-template-columns: minmax(0, 1fr) 92px;
          gap: 10px;
          align-items: center;
        }

        input[type="range"] {
          width: 100%;
          accent-color: var(--accent);
        }

        input[type="number"] {
          width: 100%;
          padding: 9px 10px;
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.1);
          background: rgba(255, 255, 255, 0.05);
          color: #f8f8f3;
          font: inherit;
        }

        select {
          width: 100%;
          padding: 9px 10px;
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.1);
          background: rgba(255, 255, 255, 0.05);
          color: #f8f8f3;
          font: inherit;
        }

        .control-body.single {
          grid-template-columns: 1fr;
        }

        .toggles {
          display: grid;
          gap: 12px;
        }

        .toggle {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          color: #d7e0e7;
          font-size: 14px;
        }

        input[type="checkbox"] {
          inline-size: 18px;
          block-size: 18px;
          accent-color: var(--accent);
        }

        .stats {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 10px 14px;
        }

        .stat {
          display: grid;
          gap: 2px;
        }

        .stat dt {
          margin: 0;
          color: #8ca0af;
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.08em;
        }

        .stat dd {
          margin: 0;
          font-size: 17px;
          font-weight: 600;
          letter-spacing: -0.03em;
        }

        .viewport-wrap {
          position: relative;
          min-height: 772px;
          border-radius: 24px;
          overflow: hidden;
          background:
            radial-gradient(circle at top right, rgba(255, 139, 61, 0.15), transparent 26%),
            radial-gradient(circle at bottom left, rgba(87, 174, 255, 0.16), transparent 28%),
            linear-gradient(180deg, rgba(9, 17, 24, 0.8), rgba(8, 14, 20, 0.94));
          border: 1px solid rgba(255, 255, 255, 0.06);
        }

        .viewport {
          position: absolute;
          inset: 0;
        }

        .viewport-header,
        .status {
          position: absolute;
          z-index: 1;
          left: 18px;
          right: 18px;
          display: flex;
          justify-content: space-between;
          gap: 12px;
          pointer-events: none;
        }

        .viewport-header {
          top: 18px;
          align-items: flex-start;
        }

        .viewport-title {
          padding: 12px 14px;
          border-radius: 16px;
          background: rgba(8, 14, 20, 0.7);
          border: 1px solid rgba(255, 255, 255, 0.08);
          backdrop-filter: blur(8px);
        }

        .viewport-title strong,
        .status strong {
          display: block;
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          color: #ffcfad;
        }

        .viewport-title span,
        .status span {
          display: block;
          margin-top: 4px;
          color: #d6dfe6;
          font-size: 13px;
        }

        .status {
          bottom: 18px;
          align-items: flex-end;
        }

        .status > div {
          max-width: 420px;
          padding: 12px 14px;
          border-radius: 16px;
          background: rgba(8, 14, 20, 0.76);
          border: 1px solid rgba(255, 255, 255, 0.08);
          backdrop-filter: blur(8px);
        }

        .status.error span {
          color: var(--warn);
        }

        @media (max-width: 1080px) {
          .shell {
            min-height: auto;
            padding: 18px;
            border-radius: 24px;
          }

          .grid {
            grid-template-columns: 1fr;
            min-height: auto;
          }

          .viewport-wrap {
            min-height: 560px;
          }
        }

        @media (max-width: 640px) {
          .panel,
          .shell {
            padding: 16px;
          }

          .control-body,
          .stats {
            grid-template-columns: 1fr;
          }

          .viewport-wrap {
            min-height: 460px;
          }
        }
      </style>

      <div class="shell">
        <div class="grid">
          <div class="panel">
            <div class="eyebrow">Embedded Web Component</div>
            <h2>Radial Oblong Pocket Studio</h2>
            <p class="lede">
              Live browser view of the same wrapped oblong pocket logic used in the Python CNC app,
              rendered directly on a cylindrical mesh with constant radial depth.
            </p>

            ${sectionMarkup}

            <section class="toggles">
              <label class="toggle">
                <span>Show pocket boundaries</span>
                <input data-kind="toggle" data-key="showBoundaries" type="checkbox">
              </label>
              <label class="toggle">
                <span>Show spiral toolpath</span>
                <input data-kind="toggle" data-key="showToolpath" type="checkbox">
              </label>
            </section>

            <section class="panel-section">
              <div class="section-kicker">Surface Finish</div>
              <label class="control">
                <span class="control-head">
                  <span>Outer Shell</span>
                  <span class="unit">appearance</span>
                </span>
                <div class="control-body single">
                  <select data-kind="select" data-key="outerFinish">
                    ${FINISH_OPTIONS.map((option) => `<option value="${option.value}">${option.label}</option>`).join("")}
                  </select>
                </div>
              </label>
              <label class="control">
                <span class="control-head">
                  <span>Pocket Walls + Floor</span>
                  <span class="unit">appearance</span>
                </span>
                <div class="control-body single">
                  <select data-kind="select" data-key="pocketFinish">
                    ${FINISH_OPTIONS.map((option) => `<option value="${option.value}">${option.label}</option>`).join("")}
                  </select>
                </div>
              </label>
            </section>

            <section class="stats">
              <span class="stats-title">Live Metrics</span>
              <dl class="stat">
                <dt>Pockets</dt>
                <dd data-metric="pockets">-</dd>
              </dl>
              <dl class="stat">
                <dt>Spiral Levels</dt>
                <dd data-metric="spiralLevels">-</dd>
              </dl>
              <dl class="stat">
                <dt>Circumference Pitch</dt>
                <dd data-metric="pitch">-</dd>
              </dl>
              <dl class="stat">
                <dt>Pocket Span</dt>
                <dd data-metric="span">-</dd>
              </dl>
            </section>
          </div>

          <div class="viewport-wrap">
            <div class="viewport-header">
              <div class="viewport-title">
                <strong>3D Pocketed Shaft</strong>
                <span>Drag to orbit, wheel to zoom, tweak controls for live updates.</span>
              </div>
            </div>
            <div class="viewport" part="viewport"></div>
            <div class="status" data-status>
              <div>
                <strong>Status</strong>
                <span data-status-text>Ready.</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  _initScene() {
    const viewport = this.shadowRoot.querySelector(".viewport");
    this._scene = new THREE.Scene();
    this._scene.background = new THREE.Color(0x091018);

    this._camera = new THREE.PerspectiveCamera(38, 1, 0.1, 2000);
    this._renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this._renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 3));
    this._renderer.outputColorSpace = THREE.SRGBColorSpace;
    viewport.appendChild(this._renderer.domElement);

    this._controls = new OrbitControls(this._camera, this._renderer.domElement);
    this._controls.enableDamping = true;
    this._controls.minDistance = 12;
    this._controls.maxDistance = 1200;
    this._controls.target.set(this._params.axialCenter, 0, 0);

    const hemi = new THREE.HemisphereLight(0xb8d8ff, 0x0a0e13, 1.4);
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(42, 36, 22);
    const rim = new THREE.DirectionalLight(0xffa25e, 0.8);
    rim.position.set(-22, -18, -14);
    const fill = new THREE.DirectionalLight(0x8dcfff, 0.55);
    fill.position.set(0, 18, -36);
    this._scene.add(hemi, key, rim, fill);

    const grid = new THREE.GridHelper(220, 16, 0x27445d, 0x1b2b38);
    grid.rotation.z = Math.PI / 2;
    grid.position.y = -28;
    this._scene.add(grid);

    this._modelGroup = new THREE.Group();
    this._scene.add(this._modelGroup);

    this._resizeObserver = new ResizeObserver(() => this._resizeRenderer());
    this._resizeObserver.observe(viewport);
    this._resizeRenderer();
  }

  _bindControls() {
    const inputs = this.shadowRoot.querySelectorAll("[data-key]");
    inputs.forEach((input) => {
      const key = input.dataset.key;
      if (!this._inputs.has(key)) {
        this._inputs.set(key, {});
      }
      this._inputs.get(key)[input.dataset.kind] = input;

      if (input.dataset.kind === "toggle") {
        input.addEventListener("change", () => {
          this._params[key] = input.checked;
          this._scheduleRebuild();
        });
        return;
      }

      if (input.dataset.kind === "select") {
        input.addEventListener("change", () => {
          this._params[key] = input.value;
          this._scheduleRebuild();
        });
        return;
      }

      if (input.dataset.kind === "range") {
        input.addEventListener("input", () => {
          this._applyNumericInput(key, Number(input.value), true);
        });
        return;
      }

      input.addEventListener("change", () => {
        this._applyNumericInput(key, Number(input.value), false);
      });
    });
  }

  _applyNumericInput(key, rawValue, fromRange) {
    if (!Number.isFinite(rawValue)) {
      return;
    }

    const config = CONTROL_SECTIONS.flatMap((section) => section.fields).find(([fieldKey]) => fieldKey === key);
    if (!config) {
      return;
    }
    const [, , min, max, step] = config;
    const snapped = step >= 1 ? Math.round(rawValue) : rawValue;
    const clamped = Math.min(max, Math.max(min, snapped));
    this._params[key] = key === "patternCount" ? Math.round(clamped) : clamped;

    const pair = this._inputs.get(key);
    if (pair.range && (!fromRange || Number(pair.range.value) !== clamped)) {
      pair.range.value = String(clamped);
    }
    if (pair.number && (fromRange || Number(pair.number.value) !== clamped)) {
      pair.number.value = String(clamped);
    }

    this._scheduleRebuild();
  }

  _syncInputsFromParams() {
    this._inputs.forEach((pair, key) => {
      if (pair.range) {
        pair.range.value = String(this._params[key]);
      }
      if (pair.number) {
        pair.number.value = String(this._params[key]);
      }
      if (pair.toggle) {
        pair.toggle.checked = Boolean(this._params[key]);
      }
      if (pair.select) {
        pair.select.value = String(this._params[key]);
      }
    });
  }

  _scheduleRebuild() {
    if (!this._isConnected) {
      return;
    }
    window.clearTimeout(this._rebuildTimer);
    this._rebuildTimer = window.setTimeout(() => this._rebuildModel(), 20);
  }

  _clearModelGroup() {
    while (this._modelGroup.children.length > 0) {
      const child = this._modelGroup.children[0];
      this._modelGroup.remove(child);
      if (child.geometry) {
        child.geometry.dispose();
      }
      if (child.material) {
        if (Array.isArray(child.material)) {
          child.material.forEach((material) => material.dispose());
        } else {
          child.material.dispose();
        }
      }
    }
  }

  _rebuildModel() {
    const status = this.shadowRoot.querySelector("[data-status]");
    const statusText = this.shadowRoot.querySelector("[data-status-text]");

    try {
      validateGeometry(this._params);
      this._clearModelGroup();

      const cadPart = buildParametricPocketPart(this._params);
      const shaftRadius = cadPart.shaftRadius;
      const materials = {
        outer: buildFinishMaterial("outer", this._params.outerFinish),
        floor: buildFinishMaterial("pocket", this._params.pocketFinish),
        wall: buildFinishMaterial("pocket", this._params.pocketFinish),
        endcap: buildFinishMaterial("endcap", this._params.outerFinish),
      };

      meshParametricPocketPart(cadPart).forEach(({ role, geometry }) => {
        const mesh = new THREE.Mesh(geometry, materials[role] ?? materials.outer);
        this._modelGroup.add(mesh);
      });

      if (this._params.showBoundaries) {
        patternBoundaries(this._params).forEach((boundary) => {
          const points = boundary.map((point) => boundaryToCylinderPoint(point, shaftRadius));
          this._modelGroup.add(makeLine(points, 0xffa25e, true));
        });
      }

      if (this._params.showToolpath) {
        generateToolpathSegments(this._params).forEach((segment) => {
          const points = segment.map((point) => unwrapToCylinderPoint(point, shaftRadius, 0.03));
          this._modelGroup.add(makeLine(points, 0x5cc8ff, false));
        });
      }

      this._updateMetrics();
      this._frameView();
      status.classList.remove("error");
      statusText.textContent = `Parametric pocket shaft rebuilt as ${cadPart.outerSurfaces.length + cadPart.endCaps.length + cadPart.pockets.length * 2} analytic surfaces, then meshed for display.`;

      this.dispatchEvent(
        new CustomEvent("pocketchange", {
          detail: this.params,
        }),
      );
    } catch (error) {
      this._clearModelGroup();
      status.classList.add("error");
      statusText.textContent = error.message;
    }
  }

  _updateMetrics() {
    const count = Math.max(1, Math.round(this._params.patternCount));
    const circumference = Math.PI * this._params.shaftDiameter;
    const pitch = circumference / count;
    const span = estimateMaxCircumferentialSpan(this._params);
    const spiralLevels = spiralLevelCount(this._params);

    this.shadowRoot.querySelector('[data-metric="pockets"]').textContent = String(count);
    this.shadowRoot.querySelector('[data-metric="spiralLevels"]').textContent = String(spiralLevels);
    this.shadowRoot.querySelector('[data-metric="pitch"]').textContent = `${formatValue(pitch, 2)} mm`;
    this.shadowRoot.querySelector('[data-metric="span"]').textContent = `${formatValue(span, 2)} mm`;
  }

  _frameView() {
    const shaftRadius = 0.5 * this._params.shaftDiameter;
    const visibleLength = this._params.pocketLength + 2 * Math.max(6, 0.35 * this._params.pocketLength);
    const targetX = this._params.axialCenter;
    this._controls.target.set(targetX, 0, 0);

    if (this._firstFrame) {
      this._camera.position.set(
        targetX + visibleLength * 0.92,
        shaftRadius * 2.55,
        shaftRadius * 2.15,
      );
      this._firstFrame = false;
      return;
    }

    const currentDistance = this._camera.position.distanceTo(this._controls.target);
    const idealDistance = Math.max(visibleLength * 1.05, shaftRadius * 5.2);
    const distance = 0.65 * currentDistance + 0.35 * idealDistance;
    const direction = this._camera.position.clone().sub(this._controls.target).normalize();
    this._camera.position.copy(direction.multiplyScalar(distance).add(this._controls.target));
  }

  _resizeRenderer() {
    const viewport = this.shadowRoot.querySelector(".viewport");
    if (!viewport || !this._renderer || !this._camera) {
      return;
    }
    const width = Math.max(1, viewport.clientWidth);
    const height = Math.max(1, viewport.clientHeight);
    this._camera.aspect = width / height;
    this._camera.updateProjectionMatrix();
    this._renderer.setSize(width, height, false);
  }

  _startRenderLoop() {
    const tick = () => {
      if (!this._isConnected) {
        return;
      }
      this._animationFrame = requestAnimationFrame(tick);
      this._controls?.update();
      this._renderer?.render(this._scene, this._camera);
    };
    tick();
  }
}

if (!customElements.get("cylindrical-oblong-pocket-viewer")) {
  customElements.define("cylindrical-oblong-pocket-viewer", CylindricalOblongPocketViewer);
}
