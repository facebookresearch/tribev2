import { useRef, useEffect, useMemo } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import useStore from '../stores/useStore'
import { activationsToColors } from '../utils/colorscale'

function BrainMesh() {
  const meshRef = useRef()
  const mesh = useStore((s) => s.mesh)
  const preds = useStore((s) => s.preds)
  const timestep = useStore((s) => s.timestep)
  const globalVmin = useStore((s) => s.globalVmin)
  const globalVmax = useStore((s) => s.globalVmax)
  const selectedRegion = useStore((s) => s.selectedRegion)
  const regionVertices = useStore((s) => s.regionVertices)

  const geometry = useMemo(() => {
    if (!mesh) return null
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(mesh.vertices, 3))
    geo.setIndex(new THREE.BufferAttribute(mesh.faces, 1))
    geo.computeVertexNormals()
    const colors = new Float32Array(mesh.nVertices * 3)
    for (let i = 0; i < mesh.nVertices; i++) {
      colors[i * 3] = 0.42
      colors[i * 3 + 1] = 0.42
      colors[i * 3 + 2] = 0.42
    }
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
    return geo
  }, [mesh])

  useEffect(() => {
    if (!geometry || !preds || !preds[timestep]) return
    const activations = preds[timestep]
    const colors = activationsToColors(activations, globalVmin, globalVmax)

    // Highlight selected region: make its vertices bright cyan outline
    if (selectedRegion && regionVertices && regionVertices[selectedRegion]) {
      const idxList = regionVertices[selectedRegion]
      for (const idx of idxList) {
        colors[idx * 3] = 0.3       // R
        colors[idx * 3 + 1] = 0.9   // G
        colors[idx * 3 + 2] = 1.0   // B (cyan)
      }
    }

    geometry.attributes.color.array.set(colors)
    geometry.attributes.color.needsUpdate = true
  }, [geometry, preds, timestep, globalVmin, globalVmax, selectedRegion, regionVertices])

  if (!geometry) return null

  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshStandardMaterial vertexColors side={THREE.DoubleSide} />
    </mesh>
  )
}

function CameraFocus({ controlsRef }) {
  const mesh = useStore((s) => s.mesh)
  const selectedRegion = useStore((s) => s.selectedRegion)
  const regionVertices = useStore((s) => s.regionVertices)
  const targetRef = useRef(new THREE.Vector3(0, 0, 0))
  const animRef = useRef(null)
  const { camera } = useThree()

  // Compute centroid when selection changes
  useEffect(() => {
    if (!selectedRegion || !mesh || !regionVertices?.[selectedRegion]) {
      targetRef.current.set(0, 0, 0)
      animRef.current = { start: performance.now(), from: null, to: new THREE.Vector3(0, 0, 0) }
      return
    }
    const idxList = regionVertices[selectedRegion]
    const verts = mesh.vertices
    let cx = 0, cy = 0, cz = 0
    for (const idx of idxList) {
      cx += verts[idx * 3]
      cy += verts[idx * 3 + 1]
      cz += verts[idx * 3 + 2]
    }
    cx /= idxList.length
    cy /= idxList.length
    cz /= idxList.length
    animRef.current = {
      start: performance.now(),
      from: null,
      to: new THREE.Vector3(cx, cy, cz),
    }
  }, [selectedRegion, mesh, regionVertices])

  // Smoothly animate camera target toward centroid
  useFrame(() => {
    const anim = animRef.current
    if (!anim || !controlsRef.current) return
    if (!anim.from) {
      anim.from = controlsRef.current.target.clone()
    }
    const elapsed = performance.now() - anim.start
    const t = Math.min(1, elapsed / 800) // 800ms ease
    const eased = 1 - Math.pow(1 - t, 3) // ease-out-cubic
    controlsRef.current.target.lerpVectors(anim.from, anim.to, eased)
    controlsRef.current.update()
    if (t >= 1) {
      animRef.current = null
    }
  })

  return null
}

export default function BrainViewer() {
  const controlsRef = useRef()

  return (
    <div className="w-full h-full bg-gray-900 rounded-lg overflow-hidden">
      <Canvas
        camera={{
          position: [-180, -30, 100],
          fov: 40,
          near: 1,
          far: 2000,
          up: [0, 0, 1], // brain's anatomical up is Z in fsaverage coords
        }}
        onCreated={({ camera }) => {
          camera.up.set(0, 0, 1)
          camera.lookAt(0, 0, 0)
        }}
      >
        <ambientLight intensity={0.7} />
        <directionalLight position={[100, -100, 150]} intensity={0.6} />
        <directionalLight position={[-100, 100, -100]} intensity={0.3} />
        <BrainMesh />
        <CameraFocus controlsRef={controlsRef} />
        <OrbitControls
          ref={controlsRef}
          enableDamping
          dampingFactor={0.12}
          rotateSpeed={0.6}
          zoomSpeed={0.7}
          panSpeed={0.5}
          minDistance={100}
          maxDistance={600}
          target={[0, 0, 0]}
          mouseButtons={{
            LEFT: THREE.MOUSE.ROTATE,
            MIDDLE: THREE.MOUSE.DOLLY,
            RIGHT: THREE.MOUSE.PAN,
          }}
          touches={{
            ONE: THREE.TOUCH.ROTATE,
            TWO: THREE.TOUCH.DOLLY_PAN,
          }}
        />
      </Canvas>
    </div>
  )
}
