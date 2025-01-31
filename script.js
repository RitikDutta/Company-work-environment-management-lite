// script.js

// Wait for the DOM and face-api.js to load
window.addEventListener('DOMContentLoaded', async () => {
  const status = document.getElementById('status');
  
  // Load face-api.js models
  status.innerText = 'Loading face-api.js models, please wait...';
  try {
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri('./models'),
      faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
      faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
      // Optional: Load face expression model
      // faceapi.nets.faceExpressionNet.loadFromUri('./models')
    ]);
    status.innerText = 'Models loaded! Initializing webcam...';
  } catch (error) {
    console.error('Error loading models:', error);
    status.innerText = 'Error loading models. Check the console for details.';
    return;
  }

  // Access the webcam
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    const video = document.getElementById('video');
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      video.play();
      status.innerText = 'Webcam initialized! Detecting faces...';
      detectFaces();
    };
  } catch (error) {
    console.error('Error accessing webcam:', error);
    status.innerText = 'Error accessing webcam. Check permissions and console for details.';
  }
});

// Function to detect faces in real-time
async function detectFaces() {
  const video = document.getElementById('video');
  const canvas = document.getElementById('overlay');
  const ctx = canvas.getContext('2d');

  // Adjust canvas size to match video
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // Define display size
  const displaySize = { width: video.videoWidth, height: video.videoHeight };
  faceapi.matchDimensions(canvas, displaySize);

  // Use requestAnimationFrame for smoother detection
  async function runDetection() {
    const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptors(); // Optional: for face recognition

    // Resize detections to match display size
    const resizedDetections = faceapi.resizeResults(detections, displaySize);

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw detections
    faceapi.draw.drawDetections(canvas, resizedDetections);
    faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
    // Optional: Draw face expressions
    // faceapi.draw.drawFaceExpressions(canvas, resizedDetections);

    // Continue the loop
    requestAnimationFrame(runDetection);
  }

  runDetection();
}



async function registerFace(label) {
  const video = document.getElementById("video");

  // Detect a face in the current video frame
  const detection = await faceapi
    .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks()
    .withFaceDescriptor();

  if (!detection) {
    console.error("No face detected. Please ensure the face is clearly visible.");
    return;
  }

  // detection.descriptor is a Float32Array of length 128
  const descriptor = detection.descriptor;
  
  // You can store it in your own data structure
  // e.g. localStorage, an array in memory, or even a server DB
  // For now, let's just store in memory
  knownFaceDescriptors.push({
    label,
    descriptor
  });

  console.log(`Registered face for ${label}`);
}





async function createFaceMatcher() {
  // Convert your array of { label, descriptor } to face-api's LabeledFaceDescriptors
  const labeledDescriptors = [];

  // Suppose each label has multiple descriptors stored
  // e.g., { label: "Alice", descriptor: Float32Array }
  // Group them by label, so that each label can have multiple descriptor arrays
  const descriptorMap = {};

  knownFaceDescriptors.forEach(item => {
    if (!descriptorMap[item.label]) {
      descriptorMap[item.label] = [];
    }
    descriptorMap[item.label].push(item.descriptor);
  });

  for (const label in descriptorMap) {
    labeledDescriptors.push(
      new faceapi.LabeledFaceDescriptors(
        label,
        descriptorMap[label] // array of Float32Array
      )
    );
  }

  // FaceMatcher is a utility that helps find the best match
  // 0.6 is a good distance threshold for a decent balance
  faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);
  console.log("FaceMatcher created!");
}



async function recognizeFaces() {
  const video = document.getElementById("video");
  const canvas = document.getElementById("overlay");
  const displaySize = { width: video.width, height: video.height };

  faceapi.matchDimensions(canvas, displaySize);

  // Continuously detect
  requestAnimationFrame(recognizeFaces);

  // Detect all faces in the current frame
  const detections = await faceapi
    .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks()
    .withFaceDescriptors();

  const resizedDetections = faceapi.resizeResults(detections, displaySize);

  // Clear canvas
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // For each detected face, find best match
  resizedDetections.forEach((detection) => {
    const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
    const box = detection.detection.box;
    const text = bestMatch.toString(); // e.g. "Alice (distance: 0.48)"

    // Draw the bounding box
    ctx.strokeStyle = "#00FF00";
    ctx.lineWidth = 2;
    ctx.strokeRect(box.x, box.y, box.width, box.height);

    // Draw the label
    ctx.font = "16px Arial";
    ctx.fillStyle = "#00FF00";
    ctx.fillText(text, box.x, box.y - 5);
  });
}
