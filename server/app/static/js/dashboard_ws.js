const ws = new WebSocket(`ws://${location.host}/dashboard/events`);

const detectionsContainer = document.getElementById("detections-content");
const annotatedImage = document.getElementById("annotated-image");
const memoryContainer = document.getElementById("memory-content");
const sceneGraphContainer = document.getElementById("scene-graph-content");

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);

    if (data.type === "sentence") {
        displayPepperSentence(data.text);
        return;
    }

    // Clear previous data
    detectionsContainer.innerHTML = "";
    if (memoryContainer) memoryContainer.innerHTML = "";
    if (sceneGraphContainer) sceneGraphContainer.innerHTML = "";
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
        detectionsContainer.innerHTML = `<p class="text-slate-500">No objects detected</p>`;
    }
    // image
    if (data.image) {
        annotatedImage.src = `data:image/jpeg;base64,${data.image}`;
        //annotatedImage.classList.remove("hidden");
    }
    //else {
    //    annotatedImage.classList.add("hidden");
    //}
    if (memoryContainer) {
        if (data.memory && data.memory.length > 0) {
            data.memory.forEach(track => {
                const div = document.createElement("div");
                div.className = "mb-2 p-2 rounded border border-slate-800 bg-slate-950";
                div.innerHTML = `<strong>${track.label}</strong> #${track.id} · hits ${track.hits} · conf ${track.confidence.toFixed(2)}`;
                memoryContainer.appendChild(div);
            });
        } else {
            memoryContainer.innerHTML = `<p class="text-slate-500">No tracked objects yet...</p>`;
        }
    }

    if (sceneGraphContainer) {
        if (data.scene_graph && data.scene_graph.length > 0) {
            data.scene_graph.forEach(edge => {
                const div = document.createElement("div");
                div.className = "mb-2 p-2 rounded border border-slate-800 bg-slate-950";
                div.textContent = `${edge.sub} ${edge.rel} ${edge.obj}`;
                sceneGraphContainer.appendChild(div);
            });
        } else {
            sceneGraphContainer.innerHTML = `<p class="text-slate-500">No relations yet...</p>`;
        }
    }
};
