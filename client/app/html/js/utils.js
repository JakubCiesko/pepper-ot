(function (window) {
  function asObject(value) {
    if (!value) {
      return {};
    }
    if (typeof value === "string") {
      try {
        return JSON.parse(value);
      } catch (e) {
        return {};
      }
    }
    return value;
  }

  function escapeHtml(value) {
    return String(value === undefined || value === null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function parseQueryParams() {
    var result = {};
    var query = String(window.location.search || "").replace(/^\?/, "");
    if (!query) {
      return result;
    }
    var parts = query.split("&");
    var i;
    for (i = 0; i < parts.length; i += 1) {
      var part = parts[i];
      if (!part) {
        continue;
      }
      var pair = part.split("=");
      var key = decodeURIComponent(pair[0] || "");
      var value = decodeURIComponent(pair.length > 1 ? pair.slice(1).join("=") : "");
      if (!key) {
        continue;
      }
      result[key] = value;
    }
    return result;
  }

  window.PepperMemoryUtils = {
    asObject: asObject,
    escapeHtml: escapeHtml,
    parseQueryParams: parseQueryParams
  };
})(window);
