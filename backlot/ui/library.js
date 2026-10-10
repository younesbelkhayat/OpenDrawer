import { el, fmtAgo, getJSON, subscribe, thumbURL } from "/ui/lib.js";

const grid = document.getElementById("grid");
const THEME_KEY = "backlot.theme";
let currentTheme = "dark";
try { currentTheme = localStorage.getItem(THEME_KEY) === "light" ? "light" : "dark"; } catch { /* private mode */ }

function applyTheme(theme) {
  currentTheme = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = currentTheme;
  try { localStorage.setItem(THEME_KEY, currentTheme); } catch { /* ignore */ }
}

function renderThemeToggle() {
  const next = currentTheme === "light" ? "dark" : "light";
  return el("button", {
    class: "theme-toggle",
    type: "button",
    title: next === "light" ? "Thème clair" : "Thème sombre",
    "aria-label": next === "light" ? "Passer au thème clair" : "Passer au thème sombre",
    "aria-pressed": currentTheme === "light" ? "true" : "false",
    onclick: () => {
      applyTheme(next);
      const replacement = renderThemeToggle();
      document.querySelector(".theme-toggle").replaceWith(replacement);
    },
  }, el("span", { class: "theme-toggle-icon", "aria-hidden": "true" }, currentTheme === "light" ? "☾" : "☀"));
}

applyTheme(currentTheme);
document.getElementById("liveBadge").before(renderThemeToggle());

function miniRail(states) {
  const rail = el("div", { class: "mini-rail" });
  for (const s of states) {
    const cls = s.status === "completed" ? "d"
      : s.status === "in_progress" ? "a"
      : s.status === "awaiting_human" ? "w" : "";
    rail.append(el("i", { class: cls, title: `${s.name}: ${s.status}` }));
  }
  return rail;
}

function card(p) {
  const poster = el("div", { class: "lib-poster" });
  if (p.poster) {
    poster.append(el("img", { src: thumbURL(p.project_id, p.poster, 640), loading: "lazy", alt: "" }));
  } else {
    poster.append(el("span", { class: "lp-txt" }, "PAS ENCORE D'IMAGE"));
  }
  if (p.live && p.active_stage) {
    poster.append(el("span", { class: "lp-live" },
      el("span", { class: "dot" }),
      p.awaiting_human ? "◈ À VALIDER" : `EN COURS · ${p.active_stage.toUpperCase()}`));
  } else if (p.awaiting_human) {
    poster.append(el("span", { class: "lp-live" }, "◈ À VALIDER"));
  }

  const meta = el("div", { class: "lb-meta" },
    el("span", { class: "chip" }, p.pipeline_type === "coloring-book" ? "livre" : "page"),
    p.difficulty ? el("span", { class: "chip" }, { easy: "facile", medium: "moyen", hard: "difficile" }[p.difficulty] || p.difficulty) : null,
    p.page_count ? el("span", { class: "chip" }, `${p.page_count} pages`) : null,
    p.qa_status ? el("span", { class: `chip${p.qa_status === "pass" ? "" : " warn"}` }, `QA ${p.qa_status}`) : null,
    el("span", { class: "when" }, fmtAgo(p.last_activity)),
  );

  const staticSuffix = new URLSearchParams(location.search).has("static") ? "?static=1" : "";
  return el("a", { class: `lib-card${p.live ? " live-card" : ""}`, href: `/p/${p.project_id}${staticSuffix}`, style: "text-decoration:none;color:inherit" },
    poster,
    el("div", { class: "lib-body" },
      el("h3", {}, (p.title || p.project_id).toUpperCase()),
      meta,
      p.stage_states.length ? miniRail(p.stage_states) : null,
    ),
  );
}

async function render() {
  const projects = await getJSON("/api/projects");
  document.getElementById("count").textContent = `${projects.length} projet(s)`;
  const liveCount = projects.filter((p) => p.live).length;
  const badge = document.getElementById("liveBadge");
  badge.classList.toggle("idle", liveCount === 0);
  document.getElementById("liveText").textContent = liveCount ? `${liveCount} EN COURS` : "INACTIF";
  grid.innerHTML = "";
  document.getElementById("empty").style.display = projects.length ? "none" : "block";
  for (const p of projects) grid.append(card(p));
}

render().catch(console.error);
if (!new URLSearchParams(location.search).has("static")) {
  subscribe("/api/library/events", () => render().catch(console.error));
}
