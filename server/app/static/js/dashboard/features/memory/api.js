import { requestJson } from "../../core/http.js";

export async function doMemoryRequest(url, options = {}) {
	return requestJson(url, options, "Request failed");
}

export async function fetchMemory() {
	return requestJson("/api/v1/memory", {}, "Failed to refresh memory");
}
