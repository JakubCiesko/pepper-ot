export function initThemeToggle() {
  const toggles = [
    document.getElementById('theme-toggle'),
    document.getElementById('theme-toggle-mobile'),
  ].filter(Boolean);
  if (toggles.length === 0) return;

  const stored = localStorage.getItem('pepper_theme');
  if (stored === 'dark') {
    document.body.classList.remove('theme-light');
    document.body.classList.add('theme-dark');
  } else {
    document.body.classList.remove('theme-dark');
    document.body.classList.add('theme-light');
  }

  toggles.forEach((toggle) => {
    toggle.addEventListener('click', () => {
      document.body.classList.toggle('theme-dark');
      document.body.classList.toggle('theme-light');
      localStorage.setItem(
        'pepper_theme',
        document.body.classList.contains('theme-dark') ? 'dark' : 'light',
      );
    });
  });
}
