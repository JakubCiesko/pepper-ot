const ws = new WebSocket(`ws://${location.host}/dashboard/events`);

const detectionsContainer = document.getElementById("detections-content");
const annotatedImage = document.getElementById("annotated-image");

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);

    if (data.type === "sentence") {
        displayPepperSentence(data.text);
        return;
    }

    // Clear previous data
    detectionsContainer.innerHTML = "";
    if (data.objects && data.objects.length > 0) {
    data.objects.forEach(obj => {
        const div = document.createElement("div");
        div.className = "obj mb-2 p-2 rounded";

        // Use color from backend colors dict
        const labelColor = data.colors && data.colors[obj.label]
            ? `rgb(${data.colors[obj.label].join(",")})`
            : "#ddd";

        div.style.backgroundColor = labelColor;
        div.style.color = "black";

        div.innerHTML = `<strong>${obj.label}</strong> (${obj.confidence.toFixed(2)}) - bbox: [${obj.bbox.map(x => x.toFixed(1)).join(", ")}]`;
        detectionsContainer.appendChild(div);
    });
    } else {
        detectionsContainer.innerHTML = `<p class="text-gray-500">No objects detected</p>`;
    }
    // image
    if (data.image) {
        annotatedImage.src = `data:image/jpeg;base64,${data.image}`;
        //annotatedImage.classList.remove("hidden");
    }
    //else {
    //    annotatedImage.classList.add("hidden");
    //}
};
