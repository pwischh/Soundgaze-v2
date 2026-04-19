"use client";

import { useRef, useMemo, useCallback, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { CameraControls } from "@react-three/drei";
import * as THREE from "three";
import type { TrackPoint } from "../lib/api";

// ---------------------------------------------------------------------------
// Colors
// ---------------------------------------------------------------------------

const COLOR_DEFAULT  = new THREE.Color("#4a4a5a");
const COLOR_SELECTED = new THREE.Color("#FF2D2D");
const COLOR_NEIGHBOR = new THREE.Color("#FF6B35");

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface PointCloudViewerProps {
  points: TrackPoint[];
  knnIds: Set<number>;
  selectedId: number | null;
  onPointClick: (point: TrackPoint) => void;
}

// ---------------------------------------------------------------------------
// Circle texture for filled circular points
// ---------------------------------------------------------------------------

function makeCircleTexture(): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d")!;
  ctx.beginPath();
  ctx.arc(32, 32, 30, 0, Math.PI * 2);
  ctx.fillStyle = "white";
  ctx.fill();
  return new THREE.CanvasTexture(canvas);
}

// ---------------------------------------------------------------------------
// Auto-rotate — pauses while user interacts
// ---------------------------------------------------------------------------

function AutoRotate({
  controlsRef,
  isInteracting,
}: {
  controlsRef: React.RefObject<CameraControls>;
  isInteracting: React.RefObject<boolean>;
}) {
  useFrame((_, delta) => {
    if (!isInteracting.current && controlsRef.current) {
      controlsRef.current.azimuthAngle += 0.06 * delta;
    }
  });
  return null;
}

// ---------------------------------------------------------------------------
// Inner scene
// ---------------------------------------------------------------------------

function PointCloud({
  points,
  knnIds,
  selectedId,
  onHover,
  mouseNDC,
  hoveredRef,
}: Omit<PointCloudViewerProps, "onPointClick"> & {
  onHover: (point: TrackPoint | null) => void;
  mouseNDC: React.RefObject<{ x: number; y: number } | null>;
  hoveredRef: React.MutableRefObject<TrackPoint | null>;
}) {
  const meshRef = useRef<THREE.Points>(null);
  const { camera, size } = useThree();
  const tempVec = useMemo(() => new THREE.Vector3(), []);
  const circleTexture = useMemo(() => makeCircleTexture(), []);

  // Positions — only depend on points (method switch). xyz is already normalized
  // by the backend to [0,1]; we center to [-0.5, 0.5].
  const { positions, renderPoints } = useMemo(() => {
    const pos = new Float32Array(points.length * 3);
    points.forEach(({ xyz }, i) => {
      pos[i * 3]     = xyz[0] - 0.5;
      pos[i * 3 + 1] = xyz[1] - 0.5;
      pos[i * 3 + 2] = xyz[2] - 0.5;
    });
    return { positions: pos, renderPoints: points };
  }, [points]);

  // Colors — depend on selection/knn state, recalculate independently of positions
  const colors = useMemo(() => {
    const col = new Float32Array(points.length * 3);
    points.forEach(({ track_id }, i) => {
      const c = track_id === selectedId  ? COLOR_SELECTED
              : knnIds.has(track_id)     ? COLOR_NEIGHBOR
              : COLOR_DEFAULT;
      col[i * 3]     = c.r;
      col[i * 3 + 1] = c.g;
      col[i * 3 + 2] = c.b;
    });
    return col;
  }, [points, knnIds, selectedId]);

  // Screen-space hover: project every point to NDC each frame, find closest to cursor.
  useFrame(() => {
    const mouse = mouseNDC.current;
    if (!mouse) {
      hoveredRef.current = null;
      onHover(null);
      return;
    }

    const pxThresh = 20 / (size.height * 0.5);
    const threshSq = pxThresh * pxThresh;
    let bestIdx = -1;
    let bestDist = Infinity;

    for (let i = 0; i < renderPoints.length; i++) {
      tempVec.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
      tempVec.project(camera);
      if (tempVec.z > 1) continue;
      const dx = tempVec.x - mouse.x;
      const dy = tempVec.y - mouse.y;
      const dSq = dx * dx + dy * dy;
      if (dSq < threshSq && dSq < bestDist) {
        bestDist = dSq;
        bestIdx = i;
      }
    }

    const found = bestIdx >= 0 ? renderPoints[bestIdx] : null;
    hoveredRef.current = found;
    onHover(found);
  });

  const selectedPoint = useMemo(
    () => points.find((p) => p.track_id === selectedId) ?? null,
    [points, selectedId],
  );

  return (
    <>
      <points ref={meshRef} frustumCulled={false}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} />
          <bufferAttribute attach="attributes-color"    args={[colors, 3]}    />
        </bufferGeometry>
        <pointsMaterial
          vertexColors
          map={circleTexture}
          alphaTest={0.5}
          size={0.04}
          sizeAttenuation
          depthWrite={false}
          transparent
          opacity={0.9}
        />
      </points>

      {/* Selected point marker */}
      {selectedPoint && (
        <mesh position={[
          selectedPoint.xyz[0] - 0.5,
          selectedPoint.xyz[1] - 0.5,
          selectedPoint.xyz[2] - 0.5,
        ]}>
          <sphereGeometry args={[0.009, 16, 16]} />
          <meshBasicMaterial color="#FF2D2D" transparent opacity={1} />
        </mesh>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Public component
// ---------------------------------------------------------------------------

export default function PointCloudViewer({
  points,
  knnIds,
  selectedId,
  onPointClick,
}: PointCloudViewerProps) {
  const [hoveredPoint, setHoveredPoint] = useState<TrackPoint | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const mouseNDC = useRef<{ x: number; y: number } | null>(null);
  const hoveredRef = useRef<TrackPoint | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const cameraControlsRef = useRef<CameraControls>(null);
  const isInteracting = useRef(false);
  const resumeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    setMousePos({ x: e.clientX, y: e.clientY });
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    mouseNDC.current = {
      x:  ((e.clientX - rect.left)  / rect.width)  * 2 - 1,
      y: -((e.clientY - rect.top)   / rect.height) * 2 + 1,
    };
  }, []);

  const handleMouseLeave = useCallback(() => {
    mouseNDC.current = null;
  }, []);

  const handleClick = useCallback(() => {
    if (hoveredRef.current) onPointClick(hoveredRef.current);
  }, [onPointClick]);

  function pauseRotation() {
    isInteracting.current = true;
    if (resumeTimer.current) clearTimeout(resumeTimer.current);
  }

  function scheduleResume(delay = 300) {
    if (resumeTimer.current) clearTimeout(resumeTimer.current);
    resumeTimer.current = setTimeout(() => { isInteracting.current = false; }, delay);
  }

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full"
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
      onMouseDown={pauseRotation}
      onMouseUp={() => scheduleResume(300)}
      onWheel={() => { pauseRotation(); scheduleResume(400); }}
    >
      <Canvas
        camera={{ position: [0, 0, 1.2], fov: 60, near: 0.01, far: 100 }}
        style={{ background: "#08090c", width: "100%", height: "100%" }}
      >
        <PointCloud
          points={points}
          knnIds={knnIds}
          selectedId={selectedId}
          onHover={setHoveredPoint}
          mouseNDC={mouseNDC}
          hoveredRef={hoveredRef}
        />
        <AutoRotate controlsRef={cameraControlsRef} isInteracting={isInteracting} />
        <CameraControls
          ref={cameraControlsRef}
          dampingFactor={0.06}
          azimuthRotateSpeed={0.5}
          polarRotateSpeed={0.5}
          dollySpeed={0.3}
          minDistance={0.08}
          maxDistance={3.5}
        />
      </Canvas>

      {/* Hover tooltip */}
      {hoveredPoint && (
        <div
          className="pointer-events-none fixed z-50 px-2 py-1 bg-black border border-white/20 font-mono text-xs text-white whitespace-nowrap"
          style={{ left: mousePos.x + 14, top: mousePos.y - 32 }}
        >
          <span className="font-black">{hoveredPoint.title}</span>
          <span className="text-white/50 ml-1">— {hoveredPoint.artist}</span>
        </div>
      )}
    </div>
  );
}
