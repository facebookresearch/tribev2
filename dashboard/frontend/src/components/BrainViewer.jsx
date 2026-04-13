import { useRef, useEffect, useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
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

  const geometry = useMemo(() => {
    if (!mesh) return null
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(mesh.vertices, 3))
    geo.setIndex(new THREE.BufferAttribute(mesh.faces, 1))
    geo.computeVertexNormals()
    const colors = new Float32Array(mesh.nVertices * 3).fill(0.3)
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
    return geo
  }, [mesh])

  useEffect(() => {
    if (!geometry || !preds || !preds[timestep]) return
    const activations = preds[timestep]
    // Use global min/max (1st/99th percentile) for consistent colorscale across time
    const colors = activationsToColors(activations, globalVmin, globalVmax)
    geometry.attributes.color.array.set(colors)
    geometry.attributes.color.needsUpdate = true
  }, [geometry, preds, timestep, globalVmin, globalVmax])

  if (!geometry) return null

  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshStandardMaterial vertexColors side={THREE.DoubleSide} />
    </mesh>
  )
}

export default function BrainViewer() {
  return (
    <div className="w-full h-full bg-gray-900 rounded-lg overflow-hidden">
      <Canvas camera={{ position: [0, 0, 250], fov: 45 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[100, 100, 100]} intensity={0.8} />
        <BrainMesh />
        <OrbitControls enableDamping dampingFactor={0.1} />
      </Canvas>
    </div>
  )
}
