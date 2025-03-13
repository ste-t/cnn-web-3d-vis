import * as THREE from "three";

export function create_camera() {
    const camera = new THREE.PerspectiveCamera(
        45,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    );

    const distance = 150;
    const angle = 35;
    camera.position.set(
        distance,
        distance * Math.tan((angle / 180) * Math.PI),
        distance
    );

    return camera;
}
