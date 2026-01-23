// collapsable utility
function toggleSection(id) {
    const content = document.getElementById(id);
    const arrow = document.getElementById(id.replace('-content','-arrow'));
    if (content.classList.contains('hidden')) {
        content.classList.remove('hidden');
        arrow.textContent = '▲';
    } else {
        content.classList.add('hidden');
        arrow.textContent = '▼';
    }
}
function showStatusMessage(msg, success = true) {
    const el = document.getElementById("status-message");
    el.textContent = msg;
    el.classList.remove("hidden");
    el.classList.toggle("text-green-600", success);
    el.classList.toggle("text-red-600", !success);
    setTimeout(() => el.classList.add("hidden"), 3000); // hide after 3s
}

const sentencesContainer = document.getElementById("sentences-content");

function displayPepperSentence(text) {
    if (!text || !sentencesContainer) return;

    // Remove placeholder text
    if (sentencesContainer.children.length === 1 &&
        sentencesContainer.children[0].textContent.includes("No messages")) {
        sentencesContainer.innerHTML = "";
    }

    // Clear any previous sentence → keep only the newest
    sentencesContainer.innerHTML = "";

    const div = document.createElement("div");
    div.className =
        "bg-blue-100 border border-blue-300 p-3 rounded shadow-sm animate-fade-in";
    div.textContent = text;

    sentencesContainer.appendChild(div);
}
