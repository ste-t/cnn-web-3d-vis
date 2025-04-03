import * as tf from "@tensorflow/tfjs";
import * as THREE from "three";

export function interpolate_colors(value) {
    // Clamp value between 0 and 1
    value = Math.min(1, Math.max(0, value));

    const start = { r: 22, g: 1, b: 50 };
    const end = { r: 131, g: 37, b: 252 };

    // const start = { r: 99, g: 47, b: 112 };
    // const end = { r: 205, g: 106, b: 132 };

    const r = Math.round(start.r + (end.r - start.r) * value) / 255.0;
    const g = Math.round(start.g + (end.g - start.g) * value) / 255.0;
    const b = Math.round(start.b + (end.b - start.b) * value) / 255.0;

    return [r, g, b];
}

// Assumes every cube has a side of 1
export function distribute_grid(
    instanced_mesh,
    settings = { shape, padding: 0, pos: [0, 0, 0] }
) {
    const cols = settings.shape[0];
    const rows = settings.shape[1];
    const padding = settings.padding;

    const width = cols + padding * (cols - 1);
    const height = rows + padding * (rows - 1);

    const _ = new THREE.Object3D();

    for (let i = 0; i < cols * rows; i++) {
        // Distribution
        let x = i % cols;
        let y = Math.floor(i / cols);

        // Padding
        x *= 1 + padding;
        y *= 1 + padding;

        // Center and move
        x -= width / 2 - settings.pos[0];
        y -= height / 2 - settings.pos[1];

        // Mirror vertically
        y = -y + padding;

        _.position.set(x, y, settings.pos[2]);

        _.updateMatrix();
        instanced_mesh.setMatrixAt(i, _.matrix);
    }
}
// Assumes every cube has a side of 1
export function distribute_3d_grid(
    instanced_mesh,
    settings = { shape, padding: 0, pos: [0, 0, 0] }
) {
    const cols = settings.shape[0];
    const rows = settings.shape[1];
    const layers = settings.shape[2];

    const padding = settings.padding;

    const width = cols + padding * (cols - 1);
    const height = rows + padding * (rows - 1);
    const depth = layers + padding * (layers - 1);

    const _ = new THREE.Object3D();

    for (let i = 0; i < cols * rows * layers; i++) {
        // Distribution
        let x = i % cols;
        let y = Math.floor((i / cols) % rows);
        let z = -Math.floor(i / (cols * rows));

        // Padding
        x *= 1 + padding;
        y *= 1 + padding;

        // Center and move
        x -= width / 2 - settings.pos[0];
        y -= height / 2 - settings.pos[1];

        // Mirror vertically
        y = -y + padding;

        _.position.set(x, y, z + settings.pos[2]);

        _.updateMatrix();
        instanced_mesh.setMatrixAt(i, _.matrix);
    }
}

export function instance_index_to_grid_coords(index, cols, rows) {
    let x = index % cols;
    let y = Math.floor((index / cols) % rows);

    return [x, y];
}

export function grid_coords_to_instance_index(coords, cols) {
    return coords[1] * cols + coords[0];
}

export function map_input_tensor_to_grid(tensor, grid_instanced_mesh) {
    tensor.data().then((weights) => {
        for (let i = 0; i < weights.length; i++) {
            const color = new THREE.Color();
            color.setRGB(
                ...interpolate_colors(weights[i]),
                THREE.SRGBColorSpace
            );

            grid_instanced_mesh.setColorAt(i, color);
        }
        grid_instanced_mesh.material.color.set(0xffffff);
        grid_instanced_mesh.instanceColor.needsUpdate = true;
    });
}

export function predict(flat_weights, model) {
    const pred = model
        .predict(tf.reshape(flat_weights, [1, 28, 28, 1]))
        .argMax(1)
        .dataSync();

    // console.log(pred);
    return pred;
}

export async function input_from_image(img_path) {
    return new Promise((resolve) => {
        const img = new Image(28, 28);
        img.src = img_path;

        img.onload = () => {
            const img_tensor = tf.browser
                .fromPixels(img, 1)
                .div(tf.scalar(255.0));

            resolve(img_tensor);
        };
    });
}

export function clamp(x, lower, upper) {
    return Math.min(Math.max(lower, x), upper);
}
