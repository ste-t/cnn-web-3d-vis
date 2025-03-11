import * as THREE from "three";

export function create_renderer() {
    const canvas = document.querySelector("#scene");

    if (!canvas) {
        console.error("Canvas element #scene not found");
        return;
    }

    const renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        antialias: true,
    });

    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(
        new THREE.Color().setHex(0x181b22, THREE.SRGBColorSpace)
    );

    return renderer;
}
