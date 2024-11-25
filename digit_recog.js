const model = {};
const localDataKey = "digitExamples";
const storedData = JSON.parse(localStorage.getItem(localDataKey)) || { images: [], labels: [] };
updateExampleCount();
document.getElementById("version").textContent = "0.0.2";

// Load the model JSON file
fetch("model.json")
    .then((response) => response.json())
    .then((data) => {
        model.weights = data.weights;
        model.biases = data.biases;
        console.log("Model loaded successfully.");
    })
    .catch((error) => console.error("Error loading model:", error));

// Load model (weights and biases) from a JSON file

//const model = {
//    weights: [], // Replace with your model weights (from model.json)
//    biases: []   // Replace with your model biases
//};

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
                sum = sum * 1.0 + activations[i] * weights[j][i];
            }
            weightedSum.push(sigmoid(sum));
        }
        activations = weightedSum;
    }
    return activations;
}

// Resample the canvas to 28x28 grayscale
function getCanvasData(canvas) {
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const imageData = ctx.getImageData(0, 0, 280, 280);
    const data = new Array(28 * 28).fill(0);
    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            let sum = 0;
            for (let i = 0; i < 10; i++) {
                const idx = 4 * ((y * 10 + i) * 280 + x * 10);
                sum += imageData.data[idx + 3]; // Alpha channel   xRed channel
            }
            data[y * 28 + x] = parseInt(sum/10.0); // Normalize and invert
            //data[y * 28 + x] = parseInt(sum/2550.0); // Normalize and invert
        }
    }
    
    let char = "\n";
    let counter = 0;
    console.clear();
    for (let i = 0; i < data.length; i++) {
        char += ((data[i]>100)?(data[i] + " "):((data[i]>10)?(" " + data[i] + " "):("  " + data[i] + " ")));
        counter++;
        char += (counter % 28)?"":"\n";
        //char += data[i] + " ";
        //char += (data[i]>0)?" X":" .";
    }
    console.log(char);
    console.log(data.length);

    //console.log(data);
    return data;
}

// Setup drawing
const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
ctx.lineWidth = 20;
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

//--------------------


//Touch Input
// Since the project to which I was supposedly contributing
// was a modern web app, I needed to support smartphones and
// tablets. This meant adding touch controls to supplement 
// the mouse controls. Keeping it simple, I only used one 
// touch at a time (sorry, multitouch). For starters, 
// I utilized three touch event counterparts to the mouse 
// events from earlier:

//touchstart – to toggle drawing mode “on”
//touchend – to toggle drawing mode “off”
//touchmove – to track finger position, used in drawing
// Because I wanted to play around with event dispatching, 
// I used these touch events to trigger their mouse event
// counterparts and do the appropriate conversions 
// (touch position to mouse position, etc).

// Set up touch events for mobile, etc
canvas.addEventListener("touchstart", function (e) {
        mousePos = getTouchPos(canvas, e);
  var touch = e.touches[0];
  var mouseEvent = new MouseEvent("mousedown", {
    clientX: touch.clientX,
    clientY: touch.clientY
  });
  canvas.dispatchEvent(mouseEvent);
}, false);

canvas.addEventListener("touchend", function (e) {
  var mouseEvent = new MouseEvent("mouseup", {});
  canvas.dispatchEvent(mouseEvent);
}, false);

canvas.addEventListener("touchmove", function (e) {
  var touch = e.touches[0];
  var mouseEvent = new MouseEvent("mousemove", {
    clientX: touch.clientX,
    clientY: touch.clientY
  });
  canvas.dispatchEvent(mouseEvent);
}, false);

// Get the position of a touch relative to the canvas
function getTouchPos(canvasDom, touchEvent) {
  var rect = canvasDom.getBoundingClientRect();
  return {
    x: touchEvent.touches[0].clientX - rect.left,
    y: touchEvent.touches[0].clientY - rect.top
  };
}
//Unfortunately, all was not well. An issue arose from a conflict with built-in browser gestures. When I moved my finger on the canvas, I wanted the page to stay still so I could draw. However, if my page had horizontal or vertical scrolling (and it did) the page would move along with my finger and making proper drawing impossible. After some frustration, I stumbled upon the solution: preventing scrolling on document.body if the target of a touch event is the canvas.

// Prevent scrolling when touching the canvas
document.body.addEventListener("touchstart", function (e) {
  if (e.target == canvas) {
    e.preventDefault();
  }
}, false);
document.body.addEventListener("touchend", function (e) {
  if (e.target == canvas) {
    e.preventDefault();
  }
}, false);
document.body.addEventListener("touchmove", function (e) {
  preventPullToRefresh = false;
  if (e.target == canvas) {
    e.preventDefault();
  }
}, false);
//And viola, a mobile-friendly e-signature!



//---------------------
document.getElementById("clearCanvas").addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById("prediction").textContent = "";
    document.getElementById("messages").textContent = "";
});

// Perform inference
canvas.addEventListener("mouseup", () => {
    const inputs = getCanvasData(canvas);
    const outputs = forwardPass(inputs);
    const prediction = outputs.indexOf(Math.max(...outputs));
    document.getElementById("prediction").textContent = `${prediction} (Confidence: ${(Math.max(...outputs) * 100).toFixed(2)}%)`;
    document.getElementById("correctLabel").value = `${prediction}`;
});

function updateExampleCount() {
    document.getElementById("exampleCount").textContent = storedData.images.length;
}

document.getElementById("storeExample").addEventListener("click", () => {
    const canvas = document.getElementById("drawingCanvas");
    //const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let grayScaleData = [];
    //for (let i = 0; i < imgData.length; i += 4) {
    //    grayScaleData.push(1.0 - imgData[i] / 255); // Normalize to [0, 1]
    //}
    grayScaleData = getCanvasData(canvas);
    const label = parseInt(document.getElementById("correctLabel").value, 10);
    storedData.images.push(grayScaleData);
    storedData.labels.push(label);
    localStorage.setItem(localDataKey, JSON.stringify(storedData));
    updateExampleCount();
    const timestamp = Date.now();
    document.getElementById("messages").textContent = "Example stored " + timestamp;
    //alert("Example stored!");
});

document.getElementById("downloadData").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(storedData)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const timestamp = Date.now();
    a.href = url;
    a.download = "user_" + timestamp + "_mnist_data.json";
    a.click();
    URL.revokeObjectURL(url);
    storedData.images = [];
    storedData.labels = [];
    localStorage.removeItem(localDataKey);
    updateExampleCount();
});


