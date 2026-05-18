(function (window) {
  var state = window.PepperMemoryState;
  var render = window.PepperMemoryRender;

  function currentLanguage() {
    return state.appState.language === "cs" ? "cs" : "en";
  }

  function resolvePepperService(session) {
    var candidates = ["PepperGroundedClient", "peppergroundedclient"];

    function tryCandidate(index) {
      if (index >= candidates.length) {
        state.appState.service = null;
        state.appState.serviceReady = false;
        render.setServiceStatus("Robot service unavailable", "err");
        render.refreshButtonAvailability();
        return;
      }
      session.service(candidates[index]).then(
        function (service) {
          state.appState.service = service;
          state.appState.serviceReady = true;
          render.setServiceStatus("Robot service connected", "ok");
          render.refreshButtonAvailability();
        },
        function () {
          tryCandidate(index + 1);
        }
      );
    }

    tryCandidate(0);
  }
  // Connect to the Python NAOqi service through qi.js. In fake-tablet mode there
  // is no robot service, so this connection path is skipped.
  function ensureServiceConnected() {
    if (state.appState.fakeMode) {
      return;
    }
    if (state.appState.serviceReady || state.appState.connecting) {
      return;
    }
    if (typeof QiSession !== "function") {
      render.setServiceStatus("qi.js unavailable", "err");
      render.refreshButtonAvailability();
      return;
    }

    state.appState.connecting = true;
    render.setServiceStatus("Connecting to robot service...", "warn");
    QiSession(
      function (session) {
        state.appState.connecting = false;
        state.appState.session = session;
        resolvePepperService(session);
      },
      function () {
        state.appState.connecting = false;
        state.appState.session = null;
        state.appState.service = null;
        state.appState.serviceReady = false;
        render.setServiceStatus("Robot service disconnected", "warn");
        render.refreshButtonAvailability();
      }
    );
  }

  function onQuestionButtonClick(event) {
    var button = event && event.currentTarget;
    if (!button) {
      return;
    }
    var question = String(button.getAttribute("data-question") || "").trim();
    if (!question) {
      return;
    }
    if (!state.appState.serviceReady || !state.appState.service) {
      render.showQaError("Robot service unavailable. Try again.");
      return;
    }
    if (state.appState.cooldownByQuestion[question]) {
      return;
    }

    state.appState.cooldownByQuestion[question] = true;
    button.disabled = true;
    render.showQaError("");

    state.appState.service.answerCachedQuestion(currentLanguage(), question).then(
      function () {
        setTimeout(function () {
          state.appState.cooldownByQuestion[question] = false;
          render.refreshButtonAvailability();
        }, state.appState.buttonCooldownMs);
      },
      function (error) {
        render.showQaError("Failed to trigger robot answer. Try again.");
        render.setServiceStatus("Service call failed", "warn");
        if (error) {
          console.log("answerCachedQuestion failed", error);
        }
        setTimeout(function () {
          state.appState.cooldownByQuestion[question] = false;
          render.refreshButtonAvailability();
        }, state.appState.buttonCooldownMs);
      }
    );
  }

  render.setQuestionClickHandler(onQuestionButtonClick);

  window.PepperMemoryService = {
    currentLanguage: currentLanguage,
    resolvePepperService: resolvePepperService,
    ensureServiceConnected: ensureServiceConnected,
    onQuestionButtonClick: onQuestionButtonClick
  };
})(window);
