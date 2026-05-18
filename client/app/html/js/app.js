(function (window) {

  var state = window.PepperMemoryState;
  var render = window.PepperMemoryRender;
  var service = window.PepperMemoryService;
  var fake = window.PepperMemoryFake;

  function bootstrap() {
    fake.initializeFakeModeFromQuery();

    window.PepperMemoryPage = {
      renderFromBridge: function (payload) {
        render.render(payload);
        if (!state.appState.fakeMode) {
          service.ensureServiceConnected();
        }
        return true;
      }
    };
    // Python waits for PepperMemoryPageReady and then calls this bridge with the
    // latest memory payload through ALTabletService.executeJS.
    window.PepperMemoryPageReady = true;

    if (state.appState.fakeMode) {
      fake.startFakeTabletPolling();
    } else {
      service.ensureServiceConnected();
      setInterval(service.ensureServiceConnected, 5000);
    }

    render.render({});
  }

  bootstrap();
})(window);
