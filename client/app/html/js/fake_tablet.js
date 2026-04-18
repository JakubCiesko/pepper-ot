(function (window) {
  var state = window.PepperMemoryState;
  var utils = window.PepperMemoryUtils;
  var render = window.PepperMemoryRender;

  function initializeFakeModeFromQuery() {
    var params = utils.parseQueryParams();
    state.appState.fakeMode = String(params.fake_tablet || "") === "1";
    if (!state.appState.fakeMode) {
      return;
    }
    var pollMs = parseInt(params.poll_ms || "500", 10);
    if (!isFinite(pollMs) || pollMs < 100) {
      pollMs = 500;
    }
    state.appState.fakePollIntervalMs = pollMs;
  }

  function pollFakePayload() {
    var request = new XMLHttpRequest();
    request.open("GET", "/payload.json?_ts=" + String(Date.now()), true);
    request.onreadystatechange = function () {
      if (request.readyState !== 4) {
        return;
      }
      if (request.status < 200 || request.status >= 300) {
        render.setServiceStatus("Fake tablet payload unavailable", "warn");
        return;
      }
      try {
        var payload = JSON.parse(request.responseText || "{}");
        render.render(payload);
      } catch (error) {
        render.setServiceStatus("Fake tablet payload parse failed", "warn");
      }
    };
    request.send(null);
  }

  function startFakeTabletPolling() {
    render.setServiceStatus("Fake tablet mode", "ok");
    render.refreshButtonAvailability();
    pollFakePayload();
    setInterval(pollFakePayload, state.appState.fakePollIntervalMs);
  }

  window.PepperMemoryFake = {
    initializeFakeModeFromQuery: initializeFakeModeFromQuery,
    pollFakePayload: pollFakePayload,
    startFakeTabletPolling: startFakeTabletPolling
  };
})(window);
