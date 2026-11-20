// The theme, and the sidebar's width, set before the page is first drawn. This one script is loaded
// in the head without `defer`, ahead of the stylesheets, so the root element
// carries a held theme by the time anything is painted: a user who holds
// Dark on a light system never sees the light page first, nor the other way
// round. With nothing held, the root carries no theme and the stylesheet
// follows the system through `prefers-color-scheme`, live.
//
// It reads the preferences the Account page keeps in this browser and
// touches nothing else; the token is never read here.

const THEME_KEY = "antumbra.dashboard.preferences";

/** Put a theme ("system", "light" or "dark"), a density and whether the
 *  sidebar is collapsed on the page. */
function applyTheme(theme, density, side) {
  const root = document.documentElement;
  const held = theme === "light" || theme === "dark";
  if (held) root.dataset.theme = theme;
  else delete root.dataset.theme;
  if (density === "compact") root.dataset.density = "compact";
  else delete root.dataset.density;
  if (side === "collapsed") root.dataset.side = "collapsed";
  else delete root.dataset.side;
  // What the browser itself draws (the canvas behind the page before its
  // style applies, scrollbars, form controls) follows the same choice.
  const scheme = document.querySelector('meta[name="color-scheme"]');
  if (scheme) scheme.content = held ? theme : "light dark";
}

/** The kept theme and density, or the defaults where nothing was kept or
 *  storage is refused. */
function keptTheme() {
  try {
    const kept = JSON.parse(localStorage.getItem(THEME_KEY) || "{}") || {};
    return { theme: kept.theme, density: kept.density, side: kept.side };
  } catch {
    return {};
  }
}

{
  const { theme, density, side } = keptTheme();
  applyTheme(theme, density, side);
}

// Another tab of this browser changed the preferences: this one follows.
window.addEventListener("storage", (event) => {
  if (event.key !== THEME_KEY && event.key !== null) return;
  const { theme, density, side } = keptTheme();
  applyTheme(theme, density, side);
  document.dispatchEvent(new CustomEvent("antumbra:preferences"));
});
