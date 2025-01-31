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
