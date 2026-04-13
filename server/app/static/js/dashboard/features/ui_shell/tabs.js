export function initTabGroup(groupName) {
  const buttons = Array.from(
    document.querySelectorAll(
      `[data-tab-group="${groupName}"][data-tab-target]`,
    ),
  );
  const panels = Array.from(
    document.querySelectorAll(`.tab-panel[data-tab-group="${groupName}"]`),
  );
  if (buttons.length === 0 || panels.length === 0) return;

  function activate(targetId) {
    panels.forEach((panel) => {
      panel.classList.toggle('hidden', panel.id !== targetId);
    });
    buttons.forEach((btn) => {
      const isActive = btn.getAttribute('data-tab-target') === targetId;
      btn.classList.toggle('active', isActive);
    });
  }

  buttons.forEach((btn) => {
    btn.addEventListener('click', () =>
      activate(btn.getAttribute('data-tab-target')),
    );
  });

  activate(buttons[0].getAttribute('data-tab-target'));
}
