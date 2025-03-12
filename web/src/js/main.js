import * as tf from "@tensorflow/tfjs";

import * as THREE from "three";
import { create_scene } from "./components/scene";
import { create_camera } from "./components/camera";
import { create_renderer } from "./components/renderer";
import { create_controls } from "./components/controls";

import { distribute_grid } from "./helpers";
import { map_input_weights_to_grid } from "./helpers";
import { instance_index_to_grid_coords } from "./helpers";
import { grid_coords_to_instance_index } from "./helpers";
import { interpolate_colors } from "./helpers";
import { predict } from "./helpers";
import { input_from_image } from "./helpers";

const model = await tf.loadLayersModel("/model/model.json");
//! TODO: stop the input_flattened madness
let input_flattened;

/* 3D scene */
const scene = create_scene();
const camera = create_camera();
const renderer = create_renderer();
const controls = create_controls(camera, renderer);

function gen_parallelepiped_geometry(depth) {
    return new THREE.BoxGeometry(1, 1, depth);
}
const geometry = {
    cube: new THREE.BoxGeometry(1, 1, 1),
    sphere: new THREE.SphereGeometry(0.5),
};
const default_color = new THREE.Color().setRGB(
    ...interpolate_colors(0),
    THREE.SRGBColorSpace
);
const material = new THREE.MeshBasicMaterial({ color: default_color });
const zeroed_material = new THREE.MeshBasicMaterial({
    color: default_color,
});

let gap = 0.5;
const layer_grids = {
    input: new THREE.InstancedMesh(geometry.cube, material, 28 * 28),
    conv1: new THREE.InstancedMesh(geometry.cube, material, 26 * 26),
    pool1: new THREE.InstancedMesh(geometry.cube, material, 13 * 13),
    conv2: new THREE.InstancedMesh(geometry.cube, material, 11 * 11),
    dense1: new THREE.InstancedMesh(geometry.cube, material, 70),
    dense2: new THREE.InstancedMesh(geometry.cube, material, 10),
};
const fake_depth = {
    conv1: new THREE.InstancedMesh(
        gen_parallelepiped_geometry(13),
        zeroed_material,
        26 * 26
    ),
    pool1: new THREE.InstancedMesh(
        gen_parallelepiped_geometry(13),
        zeroed_material,
        13 * 13
    ),
    conv2: new THREE.InstancedMesh(
        gen_parallelepiped_geometry(27),
        zeroed_material,
        11 * 11
    ),
};

const grids = new THREE.Group();
grids.add(layer_grids.input);
grids.add(layer_grids.conv1);
grids.add(fake_depth.conv1);
grids.add(layer_grids.pool1);
grids.add(fake_depth.pool1);
grids.add(layer_grids.conv2);
grids.add(fake_depth.conv2);
grids.add(layer_grids.dense1);
grids.add(layer_grids.dense2);
scene.add(grids);

const origin = new THREE.Mesh(
    // new THREE.SphereGeometry((gap * Math.SQRT2) / 2),
    new THREE.BoxGeometry(gap, gap, gap),
    new THREE.MeshBasicMaterial({ color: 0xf00000 })
);
origin.geometry.translate(-0.5, -0.5, 0);
// scene.add(origin);

distribute_grid(layer_grids.input, {
    shape: [28, 28],
    padding: gap,
    pos: [0, 0, 0],
});
distribute_grid(layer_grids.conv1, {
    shape: [26, 26],
    padding: gap,
    pos: [0, 0, -30],
});
distribute_grid(fake_depth.conv1, {
    shape: [26, 26],
    padding: gap,
    pos: [0, 0, -31 - 13 / 2 + gap],
});
distribute_grid(layer_grids.pool1, {
    shape: [13, 13],
    padding: gap,
    pos: [0, 0, -73],
});
distribute_grid(fake_depth.pool1, {
    shape: [13, 13],
    padding: gap,
    pos: [0, 0, -74 - 13 / 2 + gap],
});
distribute_grid(layer_grids.conv2, {
    shape: [11, 11],
    padding: gap,
    pos: [0, 0, -116],
});
distribute_grid(fake_depth.conv2, {
    shape: [11, 11],
    padding: gap,
    pos: [0, 0, -117],
    pos: [0, 0, -117 - 27 / 2 + gap],
});
distribute_grid(layer_grids.dense1, {
    shape: [1, 70],
    padding: gap,
    pos: [0, 0, -174],
});
distribute_grid(layer_grids.dense2, {
    shape: [1, 10],
    padding: gap * 4,
    pos: [0, 0, -204],
});

function paint() {
    raycaster.setFromCamera(pointer, camera);
    const intersection = raycaster.intersectObject(layer_grids.input, false);

    if (intersection.length > 0) {
        const index = intersection[0].instanceId;

        input_flattened[index] = 1;
        layer_grids.input.setColorAt(
            index,
            new THREE.Color().setRGB(
                ...interpolate_colors(1),
                THREE.SRGBColorSpace
            )
        );

        const grid_coords = instance_index_to_grid_coords(index, 28, 28);

        const surroundings = [
            [-1, 1],
            [0, 1],
            [1, 1],
            [1, 0],
            [1, -1],
            [0, -1],
            [-1, -1],
            [-1, 0],
        ];

        surroundings.forEach((offset) => {
            const coords = [
                grid_coords[0] + offset[0],
                grid_coords[1] + offset[1],
            ];
            const index = grid_coords_to_instance_index(coords, 28);
            if (index >= 28 * 28) return;
            if (!(-1 < coords[0] && coords[0] < 28)) return;

            let color = new THREE.Color();
            layer_grids.input.getColorAt(index, color);

            input_flattened[index] += 0.3;
            input_flattened[index] = Math.min(1, input_flattened[index]);

            layer_grids.input.setColorAt(
                index,
                new THREE.Color().setRGB(
                    ...interpolate_colors(input_flattened[index]),
                    THREE.SRGBColorSpace
                )
            );
        });

        layer_grids.input.instanceColor.needsUpdate = true;
        update_grids();
    }
}

window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

const keys = {
    "mouse-primary": false,
    "mouse-move": false,
};

window.addEventListener("mousedown", (event) => {
    keys["mouse-primary"] = event.buttons === 1;
    if (event.buttons === 1) do_features_animation = false;
});
window.addEventListener("pointerdown", (event) => {
    keys["mouse-primary"] = ["touch", "pen"].includes(event.pointerType);
    if (event.pointerType !== "mouse") do_features_animation = false;
});

window.addEventListener("pointerup", (event) => {
    if (event.button === 0) {
        keys["mouse-primary"] = false;
        do_features_animation = true;
    }
});

let do_features_animation = true;
window.addEventListener("keypress", (event) => {
    if (event.code === "Space") do_features_animation = !do_features_animation;
    if (event.code === "KeyC") {
        input_flattened = tf.zeros([28, 28]).dataSync();
        map_input_weights_to_grid(input_flattened, layer_grids.input);
        update_grids();
    }
    if (event.code === "KeyE") eraser_mode = !eraser_mode;
});
window.addEventListener("keydown", (event) => {
    keys[event.code] = true;
});
window.addEventListener("keyup", (event) => {
    keys[event.code] = false;
});

let pointer = new THREE.Vector2();
window.addEventListener("pointermove", (event) => {
    keys["mouse-move"] = true;
    pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
    pointer.y = -(event.clientY / window.innerHeight) * 2 + 1;
});

const raycaster = new THREE.Raycaster();
function update() {
    if (keys["mouse-move"]) {
        if (keys["mouse-primary"]) {
            paint();

            keys["mouse-move"] = false;
        }
    }
}

let angle = 0;
function animate() {
    requestAnimationFrame(animate);

    update();

    controls.update(); // Needed after any manual camera transformation
    renderer.render(scene, camera);
}
animate();

/* Example image */
input_flattened = await input_from_image("digits/7.png");
map_input_weights_to_grid(input_flattened, layer_grids.input);

/* Tensorflow model */
const layers = model.layers;
console.log("Layers:", layers);

const layer_outputs = layers.map((layer) => layer.output);
const intermediate_model = tf.model({
    inputs: model.input,
    outputs: layer_outputs,
});

let interval1;
let interval2;
let kernel1_index = 0;
let kernel2_index = 0;
function update_grids() {
    const activations = intermediate_model.predict(
        tf.reshape(input_flattened, [1, 28, 28, 1])
    );

    // const shapes_list = activations.map((activation) => activation.shape);
    // console.log("Layer shapes:", shapes_list);

    // activations.forEach((activation, i) => {
    //     console.log(`Layer ${i} output:`, activation.shape);
    // });

    let features1 = tf.squeeze(activations[1], [0]);
    features1 = tf.split(features1, 14, -1);
    features1 = features1.map((f) => f.squeeze(-1));

    let features1_pooled = tf.squeeze(activations[2], [0]);
    features1_pooled = tf.split(features1_pooled, 14, -1);
    features1_pooled = features1_pooled.map((f) => f.squeeze(-1));

    let features2 = tf.squeeze(activations[3], [0]);
    features2 = tf.split(features2, 28, -1);
    features2 = features2.map((f) => f.squeeze(-1));

    function swap_feature1() {
        if (do_features_animation) kernel1_index++;
        if (kernel1_index > 13) kernel1_index = 0;

        map_input_weights_to_grid(
            features1[kernel1_index].dataSync(),
            layer_grids.conv1
        );
        map_input_weights_to_grid(
            features1_pooled[kernel1_index].dataSync(),
            layer_grids.pool1
        );
    }
    swap_feature1();
    const dt = 800;
    clearInterval(interval1);
    interval1 = setInterval(swap_feature1, dt);

    function swap_feature2() {
        if (do_features_animation) kernel2_index++;
        if (kernel2_index > 27) kernel2_index = 0;

        map_input_weights_to_grid(
            features2[kernel2_index].dataSync(),
            layer_grids.conv2
        );
    }
    swap_feature2();
    clearInterval(interval2);
    interval2 = setInterval(swap_feature2, dt / 2);

    let normalized_dense1 = activations[6].div(activations[6].max());
    map_input_weights_to_grid(normalized_dense1.dataSync(), layer_grids.dense1);

    let normalized_output = activations[7].div(activations[7].max());
    map_input_weights_to_grid(normalized_output.dataSync(), layer_grids.dense2);

    // Display prediction
    document.getElementById("text").textContent =
        "This is a " + tf.argMax(activations[7].dataSync()).dataSync();
}

update_grids();
