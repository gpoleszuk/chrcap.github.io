// Load model (weights and biases) from a JSON file
const model = {
    weights: [], // Replace with your model weights (from model.json)
    biases: []   // Replace with your model biases
};

// Sigmoid activation function
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// Perform a forward pass
function forwardPass(inputs) {
    let activations = inputs;
    for (let layer = 0; layer < model.weights.length; layer++) {
        let weightedSum = [];
        const weights = model.weights[layer];
        const biases = model.biases[layer];
        for (let j = 0; j < weights.length; j++) {
            let sum = biases[j];
            for (let i = 0; i < weights[j].length; i++) {
                sum += activations[i] * weights[j][i];
            }
            weightedSum.push(sigmoid(sum));
        }
        activations = weightedSum;
    }
    return activations;
}

// Resample the canvas to 28x28 grayscale
function getCanvasData(canvas) {
    const ctx = canvas.getContext("2d");
    const imageData = ctx.getImageData(0, 0, 280, 280);
    const data = new Array(28 * 28).fill(0);
    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            let sum = 0;
            for (let i = 0; i < 10; i++) {
                const idx = 4 * ((y * 10 + i) * 280 + x * 10);
                sum += imageData.data[idx]; // Red channel
            }
            data[y * 28 + x] = 1 - sum / 2550; // Normalize and invert
        }
    }
    return data;
}

// Setup drawing
const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d", willReadFrequently="true");
ctx.lineWidth = 10;
ctx.lineCap = "round";
let drawing = false;

canvas.addEventListener("mousedown", () => (drawing = true));
canvas.addEventListener("mouseup", () => (drawing = false));
canvas.addEventListener("mousemove", (e) => {
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
});

document.getElementById("clearCanvas").addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById("prediction").textContent = "";
});

// Perform inference
canvas.addEventListener("mouseup", () => {
    const inputs = getCanvasData(canvas);
    const outputs = forwardPass(inputs);
    const prediction = outputs.indexOf(Math.max(...outputs));
    document.getElementById("prediction").textContent = `${prediction} (Confidence: ${(Math.max(...outputs) * 100).toFixed(2)}%)`;
});

