(function (window) {
  var state = window.PepperMemoryState;
  var utils = window.PepperMemoryUtils;

  function setServiceStatus(text, mode) {
    var node = document.getElementById("service-status");
    node.className = "status " + (mode || "warn");
    node.textContent = text;
  }

  function showQaError(text) {
    var node = document.getElementById("qa-error");
    if (!text) {
      node.style.display = "none";
      node.textContent = "";
      return;
    }
    node.style.display = "block";
    node.textContent = text;
  }

  function renderObjects(payload) {
    var host = document.getElementById("objects");
    var labels = payload.object_labels || [];
    var counts = payload.label_counts || {};
    var html = "";
    var i;

    if (!labels.length) {
      host.innerHTML = '<span class="muted">No objects remembered.</span>';
      return;
    }

    for (i = 0; i < labels.length; i += 1) {
      var label = labels[i];
      var count = counts[label] || 0;
      html += '<span class="chip">' + utils.escapeHtml(label) + ' (' + utils.escapeHtml(count) + ')</span>';
    }
    host.innerHTML = html;
  }

  function renderEdgeList(hostId, edges, formatter, emptyText) {
    var host = document.getElementById(hostId);
    var html = "";
    var i;

    if (!edges || !edges.length) {
      host.innerHTML = '<li class="muted">' + utils.escapeHtml(emptyText) + '</li>';
      return;
    }

    for (i = 0; i < edges.length; i += 1) {
      var edge = edges[i] || {};
      html += '<li>' + utils.escapeHtml(formatter(edge)) + '</li>';
    }
    host.innerHTML = html;
  }

  function renderGraph(payload) {
    var host = document.getElementById("graph");
    var svg = payload.graph_svg || "";
    if (!svg) {
      host.innerHTML = "No scene graph yet.";
      return;
    }
    host.innerHTML = svg;
  }

  function bindQuestionButtons() {
    var buttons = document.querySelectorAll(".qa-btn");
    var i;
    for (i = 0; i < buttons.length; i += 1) {
      buttons[i].onclick = window.PepperMemoryRender._questionClickHandler;
    }
  }

  function refreshButtonAvailability() {
    var buttons = document.querySelectorAll(".qa-btn");
    var disabled = !state.appState.serviceReady;
    var i;
    for (i = 0; i < buttons.length; i += 1) {
      buttons[i].disabled = disabled;
    }
  }

  function renderQA(payload) {
    var host = document.getElementById("qa");
    var pairs = payload.pregenerated_qa || [];
    var html = "";
    var i;

    if (!pairs.length) {
      host.innerHTML = "No pregenerated questions yet.";
      return;
    }

    for (i = 0; i < pairs.length; i += 1) {
      var item = pairs[i] || {};
      var q = String(item.question || "").trim();
      var a = String(item.answer || "").trim();
      if (!q) {
        continue;
      }
      html += '<div class="qa-item">';
      html += '<button class="qa-btn" data-question="' + utils.escapeHtml(q) + '">' + utils.escapeHtml(q) + '</button>';
      if (a) {
        html += '<div class="qa-answer">' + utils.escapeHtml(a) + '</div>';
      }
      html += '</div>';
    }

    if (!html) {
      host.innerHTML = "No pregenerated questions yet.";
      return;
    }

    host.innerHTML = html;
    bindQuestionButtons();
    refreshButtonAvailability();
  }

  function render(payload) {
    payload = utils.asObject(payload);
    state.appState.payload = payload;
    state.appState.language = String(payload.ui_language || "en").trim().toLowerCase();

    renderObjects(payload);
    renderEdgeList(
      "attributes",
      payload.attributes || [],
      function (edge) {
        var sub = edge.sub || "object";
        var rel = edge.rel || "attribute";
        return sub + ": " + rel;
      },
      "No attributes remembered."
    );
    renderEdgeList(
      "relationships",
      payload.relationships || [],
      function (edge) {
        return (edge.sub || "?") + " " + (edge.rel || "related_to") + " " + (edge.obj || "?");
      },
      "No relationships remembered."
    );
    renderGraph(payload);
    renderQA(payload);
  }

  function setQuestionClickHandler(handlerFn) {
    if (typeof handlerFn === "function") {
      window.PepperMemoryRender._questionClickHandler = handlerFn;
    }
    bindQuestionButtons();
  }

  window.PepperMemoryRender = {
    _questionClickHandler: function () {},
    setServiceStatus: setServiceStatus,
    showQaError: showQaError,
    renderObjects: renderObjects,
    renderEdgeList: renderEdgeList,
    renderGraph: renderGraph,
    renderQA: renderQA,
    refreshButtonAvailability: refreshButtonAvailability,
    setQuestionClickHandler: setQuestionClickHandler,
    render: render
  };
})(window);
