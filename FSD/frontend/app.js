// Compose routes /api/predict through Nginx to the backend's /predict endpoint.
const API_BASE_URL = window.FSD_API_BASE_URL || "/api";
const form = document.querySelector("#upload-form");
const fileInput = document.querySelector("#image-file");
const dropZone = document.querySelector("#drop-zone");
const selectedFile = document.querySelector("#selected-file");
const submitButton = document.querySelector("#submit-button");
const statusBadge = document.querySelector("#result-status");
const emptyState = document.querySelector("#empty-state");
const resultContent = document.querySelector("#result-content");
const resultImage = document.querySelector("#result-image");
const resultDimensions = document.querySelector("#result-dimensions");
const laneCount = document.querySelector("#lane-count");
const downloadLink = document.querySelector("#download-link");
const errorMessage = document.querySelector("#error-message");
const MAX_BYTES = 10 * 1024 * 1024;

function setFile(file) {
  if (!file) return;
  const allowed = ["image/jpeg", "image/png", "image/webp"];
  if (!allowed.includes(file.type)) {
    showError("Choose a JPG, PNG, or WebP image.");
    fileInput.value = "";
    return;
  }
  if (file.size > MAX_BYTES) {
    showError("The image must be 10 MB or smaller.");
    fileInput.value = "";
    return;
  }
  errorMessage.hidden = true;
  selectedFile.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  selectedFile.hidden = false;
  submitButton.disabled = false;
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
  statusBadge.className = "status";
  statusBadge.innerHTML = "<i></i> ERROR";
}

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
}
dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  setFile(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;

  submitButton.disabled = true;
  submitButton.querySelector("span").textContent = "Detecting lanes…";
  statusBadge.className = "status busy";
  statusBadge.innerHTML = "<i></i> PROCESSING";
  errorMessage.hidden = true;

  const payload = new FormData();
  payload.append("file", file);
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, { method: "POST", body: payload });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Prediction failed. Check that the API and model are ready.");

    const imageUrl = `data:image/png;base64,${result.annotated_image_png_base64}`;
    resultImage.src = imageUrl;
    downloadLink.href = imageUrl;
    resultDimensions.textContent = `${result.width} × ${result.height} px`;
    const count = Array.isArray(result.lanes) ? result.lanes.length : 0;
    laneCount.textContent = `${count} lane${count === 1 ? "" : "s"} returned`;
    emptyState.hidden = true;
    resultContent.hidden = false;
    statusBadge.className = "status ready";
    statusBadge.innerHTML = "<i></i> COMPLETE";
  } catch (error) {
    showError(error.message || "Could not connect to the prediction service.");
  } finally {
    submitButton.disabled = !fileInput.files[0];
    submitButton.querySelector("span").textContent = "Detect lane markings";
  }
});
