const slider = document.getElementById("threshold-slider");
const input = document.getElementById("threshold-input");

function update_conf_threshold(value) {
  fetch("/api/config/threshold", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({threshold: value})
    });
}
slider.addEventListener("input", () => {
    input.value = slider.value;
    update_conf_threshold(parseFloat(slider.value));
});

// slider update
input.addEventListener("change", () => {
    let val = parseFloat(input.value);
    if (isNaN(val)) val = 0; // invalid numbers
    if (val < 0) val = 0;
    if (val > 1) val = 1;
    input.value = val.toFixed(2);
    slider.value = val.toFixed(2);
    update_conf_threshold(val.toFixed(2));
});
const modelSelect = document.getElementById("model-select");
const changeModelButton = document.getElementById("change-model");

//get available models from server
async function fetchModels() {
    try {
        const res = await fetch("/dashboard/config/get_models");
        const data = await res.json();
        modelSelect.innerHTML = "";
        data.models.forEach(model => {
            const opt = document.createElement("option");
            opt.value = model;
            opt.textContent = model;
            modelSelect.appendChild(opt);
        });
    } catch(err) {
        console.error("Failed to fetch models:", err);
        modelSelect.innerHTML = "<option>Error loading models</option>";
    }
}

// Switch model
changeModelButton.addEventListener("click", async () => {
    const selectedModel = modelSelect.value;
    showStatusMessage("Changing model to " + selectedModel);
    try {
        const res = await fetch("/api/config/model", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({model: selectedModel})
        });
        const data = await res.json();
        // this will pop up window, maybe add progress bar?
        if (data.ok) {
            showStatusMessage(`Model changed to: ${data.selected_model}`);
        } else {
            showStatusMessage(`Failed to change model: ${data.error}`, false);
        }
    } catch(err) {
        console.error("Error changing model:", err);
        showStatusMessage("Error changing model", false);
    }
});

// initial model fetching
fetchModels();

// language setting
const languageSelect = document.getElementById("language-select");
const changeLanguageButton = document.getElementById("change-language");

// Switch language
changeLanguageButton.addEventListener("click", async () => {
    const selectedLang = languageSelect.value;
    try {
        const res = await fetch("/api/config/language", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({language: selectedLang})
        });
        const data = await res.json();
        if (data.ok) {
            showStatusMessage(`Language changed to: ${data.language}`);
        } else {
            showStatusMessage(`Failed to change language: ${data.error || "unknown error"}`, false);
        }
    } catch(err) {
        console.error("Error changing language:", err);
        showStatusMessage("Error changing language");
    }
});
