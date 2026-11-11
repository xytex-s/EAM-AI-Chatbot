// theme.js - runs in <head> before the page is drawn, so there's no flash
// of the wrong colours. The ONLY thing this site stores in the browser is the
// light/dark choice. Chat text is never stored.
(function () {
  try {
    var saved = localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") {
      document.documentElement.setAttribute("data-theme", saved);
    }
  } catch (e) {
    // Private browsing or blocked storage: just follow the device setting.
  }
})();
