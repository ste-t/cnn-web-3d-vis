import { OrbitControls } from "three/examples/jsm/Addons.js";

export function create_controls(camera, renderer) {
    const controls = new OrbitControls(camera, renderer.domElement);

    controls.enableDamping = true; // Smooth camera motion
    controls.dampingFactor = 0.05;

    controls.enableRotate = false;

    return controls;
}
