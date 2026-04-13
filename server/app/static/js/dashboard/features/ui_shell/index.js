import { initPageNavigation } from './navigation.js';
import { initSidebarToggle } from './sidebar.js';
import { initTabGroup } from './tabs.js';
import { initThemeToggle } from './theme.js';

export function initUiShell() {
  initTabGroup('live');
  initTabGroup('settings');
  initThemeToggle();
  initPageNavigation();
  initSidebarToggle();
}
