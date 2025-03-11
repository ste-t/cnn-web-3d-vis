import * as THREE from "three";

export function create_camera() {
    // TODO: experiment with ortographic camera
    // const camera = new THREE.OrthographicCamera();
    const camera = new THREE.PerspectiveCamera(
        45,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    );
    camera.position.set(0, 0, 70);

    const distance = 150;
    const angle = 35;
    camera.position.set(
        distance / 1,
        distance * Math.tan((angle / 180) * Math.PI),
        distance
    );

    return camera;
}
