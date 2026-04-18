(function (window) {
  window.PepperMemoryState = {
    appState: {
      payload: {},
      language: "en",
      serviceReady: false,
      service: null,
      session: null,
      connecting: false,
      fakeMode: false,
      fakePollIntervalMs: 500,
      buttonCooldownMs: 600,
      cooldownByQuestion: {}
    }
  };
})(window);
