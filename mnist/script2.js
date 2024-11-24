const model = {};

// Load the model JSON file
fetch("model.json")
    .then((response) => response.json())
    .then((data) => {
        model.weights = data.weights;
        model.biases = data.biases;
        console.log("Model loaded successfully.");
    })
    .catch((error) => console.error("Error loading model:", error));

// Sigmoid activation function
const sigmoid = (x) => 1.0 / (1.0 + Math.exp(-x));

// Perform a forward pass
function forwardPass(inputs) {
    let activations = inputs;
    for (let layer = 0; layer < model.weights.length; layer++) {
        let weightedSum = [];
        const weights = model.weights[layer];
        const biases = model.biases[layer];
        //console.log("Weights: " + weights[0][0]);
        //console.log("Biases: " + biases[0]);
        //let testWeights = 0.0; 
        //for (let k = 0; k < weights.length; k++) {
        //    for (let j = 0; j < weights[0].length; j++) {
        //       testWeights += weights[k][j];
        //    }
        //}
        //console.log("Sum: " + testWeights);
        //console.log("Biases len: " + biases.length);
        //console.log("Weights len: " + weights[2].length);
        for (let j = 0; j < weights.length; j++) {
            let sum = biases[j];
            for (let i = 0; i < weights[j].length; i++) {
                sum = sum * 1.0 + activations[i] * weights[j][i];
                //console.log("j: " + j + " i: " + i + " act[i]: " + activations[i] + " wgt[j][i]: " + weights[j][i] + " sum: " + sum);
            }
            weightedSum.push(sigmoid(sum));
        }
        activations = weightedSum;
    }
    return activations;
}

// Setup drawing
const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });

ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;

canvas.addEventListener("mousedown", () => (drawing = true));
canvas.addEventListener("mouseup", () => (drawing = false));
canvas.addEventListener("mousemove", (e) => {
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / (rect.width / canvas.width));
    const y = Math.floor((e.clientY - rect.top) / (rect.height / canvas.height));
    //ctx.lineWidth = 4;
    //ctx.lineCap = "round";
    ctx.fillStyle = "black";
    ctx.fillRect(x, y, 3, 3); // Draw a pixel
});

document.getElementById("clearCanvas").addEventListener("click", () => {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById("prediction").textContent = "";
});

// Perform inference
document.getElementById("drawingCanvas").addEventListener("mouseup", () => {
    const imageData = ctx.getImageData(0, 0, 28, 28).data;
    const inputs = [];
    for (let i = 0; i < imageData.length; i += 4) {
        // inputs.push(1 - imageData[i] / 255); // Normalize and invert
        inputs.push(1.0 - imageData[i] / 255.0);
    }
    //console.log(inputs);
    const outputs = forwardPass(inputs);
    const prediction = outputs.indexOf(Math.max(...outputs));
    //for(let i = 0; i < outputs.length; i++) {
    //    console.log("i: " + i + " " + outputs[i]*100.0 + "\n");
    //}
    document.getElementById("prediction").textContent = `${prediction} (Confidence: ${(Math.max(...outputs) * 100).toFixed(2)}%)`;
});

