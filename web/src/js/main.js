import * as tf from "@tensorflow/tfjs";

import {
    asyncBufferFromUrl,
    parquetMetadataAsync,
    parquetReadObjects,
} from "hyparquet";

import * as THREE from "three";
import { create_scene } from "./components/scene";
import { create_camera } from "./components/camera";
import { create_renderer } from "./components/renderer";
import { create_controls } from "./components/controls";

import {
    clamp,
    distribute_grid,
    map_input_tensor_to_grid,
    instance_index_to_grid_coords,
    grid_coords_to_instance_index,
    interpolate_colors,
    input_from_image,
} from "./helpers";

const model = await tf.loadLayersModel("/model/model.json");
let input;

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
};
const default_color = new THREE.Color().setRGB(
    ...interpolate_colors(0),
    THREE.SRGBColorSpace
);
const material = new THREE.MeshStandardMaterial({ color: default_color });
const zeroed_material = new THREE.MeshStandardMaterial({
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

// const pointLight = new THREE.PointLight(0xffffff, 1, 0, 0);
// pointLight.position.set(-200, 200, 200); // Set light position
// scene.add(pointLight);

const key_light = new THREE.DirectionalLight(0xffffff, 1);
key_light.position.set(10, 50, 40);
scene.add(key_light);

const fill_light = new THREE.DirectionalLight(0xffffff, 0.4);
fill_light.position.copy(key_light);
fill_light.position.x *= -1;
scene.add(fill_light);

const back_light = new THREE.DirectionalLight(0xffffff, 0.4);
back_light.position.set(0, 30, -230);
scene.add(back_light);

const ambientLight = new THREE.AmbientLight(0xffffff, 1); // Soft, general light
scene.add(ambientLight);

const parquet_file = await asyncBufferFromUrl({ url: "test.parquet" });
const parquet_metadata = await parquetMetadataAsync(parquet_file);
async function rnd_example() {
    const rnd_index = Math.floor(
        Math.random() * Number(parquet_metadata.num_rows)
    );

    const data = await parquetReadObjects({
        file: parquet_file,
        columns: ["image"],
        rowStart: rnd_index,
        rowEnd: rnd_index + 1,
        utf8: false,
    });

    const blob = new Blob([data[0].image.bytes], { type: "image/png" }); // Assuming the image is PNG
    const url = URL.createObjectURL(blob);

    input = await input_from_image(url);
    map_input_tensor_to_grid(input, layer_grids.input);
    update_grids();
}

function paint() {
    raycaster.setFromCamera(pointer, camera);
    const intersection = raycaster.intersectObject(layer_grids.input, false);

    if (intersection.length > 0) {
        const index = intersection[0].instanceId;
        const grid_coords = instance_index_to_grid_coords(index, 28, 28);

        const brush = tf.tensor([
            [[0.3], [0.3], [0.3]],
            [[0.3], [1.0], [0.3]],
            [[0.3], [0.3], [0.3]],
        ]);

        const neighbors = [
            [-1, -1],
            [0, -1],
            [1, -1],

            [-1, 0],
            [0, 0],
            [1, 0],

            [-1, 1],
            [0, 1],
            [1, 1],
        ];
        const indices = [];

        neighbors.forEach((offset) => {
            indices.push([
                clamp(grid_coords[1] + offset[1], 0, 27),
                clamp(grid_coords[0] + offset[0], 0, 27),
                0,
            ]);
        });

        input = tf
            .scatterND(indices, brush.flatten(), input.shape)
            .add(input)
            .minimum(1);

        indices.forEach((index) => {
            const weight = input
                .slice([index[0], index[1], 0], [1, 1, 1])
                .dataSync()[0];

            const instance_index = grid_coords_to_instance_index(
                [index[1], index[0]],
                28
            );

            layer_grids.input.setColorAt(
                instance_index,
                new THREE.Color().setRGB(
                    ...interpolate_colors(weight),
                    THREE.SRGBColorSpace
                )
            );
        });

        layer_grids.input.instanceColor.needsUpdate = true;
        update_grids();
    }
}

function toggle_animation() {
    do_features_animation = !do_features_animation;
}
function clear_input() {
    input = tf.zeros([28, 28, 1]);
    map_input_tensor_to_grid(input, layer_grids.input);
    update_grids();
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
    map_input_tensor_to_grid(input, layer_grids.input);
});

let do_features_animation = true;
window.addEventListener("keypress", (event) => {
    if (event.code === "Space") toggle_animation();
    if (event.code === "KeyC") clear_input();
    if (event.code === "KeyR") rnd_example();
    // if (event.code === "KeyE") eraser_mode = !eraser_mode;
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

document.querySelector("#random").addEventListener("click", rnd_example);
document.querySelector("#pause").addEventListener("click", toggle_animation);
document.querySelector("#clear").addEventListener("click", clear_input);

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
input = await input_from_image("digits/7.png");
map_input_tensor_to_grid(input, layer_grids.input);

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
    // const activations = intermediate_model.predict(
    //     tf.reshape(input_flattened, [1, 28, 28, 1])
    // );

    const activations = intermediate_model.predict(
        tf.reshape(input, [1, 28, 28, 1])
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

        map_input_tensor_to_grid(features1[kernel1_index], layer_grids.conv1);
        map_input_tensor_to_grid(
            features1_pooled[kernel1_index],
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

        map_input_tensor_to_grid(features2[kernel2_index], layer_grids.conv2);
    }
    swap_feature2();
    clearInterval(interval2);
    interval2 = setInterval(swap_feature2, dt / 2);

    let normalized_dense1 = activations[6].div(activations[6].max());
    map_input_tensor_to_grid(normalized_dense1, layer_grids.dense1);

    let normalized_output = activations[7].div(activations[7].max());
    map_input_tensor_to_grid(normalized_output, layer_grids.dense2);

    // Display prediction
    document.getElementById("text").textContent =
        "This is a " + tf.argMax(activations[7].dataSync()).dataSync();
}

update_grids();
