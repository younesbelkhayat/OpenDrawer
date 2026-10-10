// Backlot project board — renders BoardState and stays live via SSE.
// Shows a coloring project step by step (photo → cartoon → traits → zones →
// page) or the pages of a coloring book.

import { STAGE_ICONS, el, fmtAgo, fmtClock, getJSON, mediaURL, subscribe, thumbURL } from "/ui/lib.js";

const projectId = decodeURIComponent(location.pathname.split("/p/")[1] || "");
const app = document.getElementById("app");
const modal = document.getElementById("modal");

const THEME_KEY = "backlot.theme";
let currentTheme = "dark";
try { currentTheme = localStorage.getItem(THEME_KEY) === "light" ? "light" : "dark"; } catch { /* private mode */ }
let state = null;
let selectedStage = null;

const STAGE_LABELS = {
  intake: "photo",
  cartoon: "cartoon",
  line_art: "traits",
  regions: "zones",
  page: "page",
  collect: "photos",
  pages: "pages",
  assemble: "livre",
};
const STATUS_FR = {
  pending: "à venir",
  in_progress: "en cours",
  awaiting_human: "à valider",
  completed: "terminé",
  failed: "échec",
  skipped: "ignoré",
};
const DIFFICULTY_FR = { easy: "facile", medium: "moyen", hard: "difficile" };

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
    onclick: () => { applyTheme(next); render(); },
  }, el("span", { class: "theme-toggle-icon", "aria-hidden": "true" }, currentTheme === "light" ? "☾" : "☀"));
}

applyTheme(currentTheme);

// ---------------------------------------------------------------------------
// header + rail
// ---------------------------------------------------------------------------

function renderSlate(s) {
  const chips = [
    el("span", { class: "chip" }, s.pipeline.pipeline_type),
    s.difficulty ? el("span", { class: "chip" }, `difficulté ${DIFFICULTY_FR[s.difficulty] || s.difficulty}`) : null,
    s.style ? el("span", { class: "chip" }, s.style) : null,
    s.commercial_safe ? el("span", { class: "chip" }, "commercial-safe") : null,
    s.book_id ? el("a", { class: "chip", href: `/p/${encodeURIComponent(s.book_id)}` }, `livre ${s.book_id}`) : null,
  ];
  const awaiting = s.stages.find((x) => x.status === "awaiting_human");
  const inProgress = s.stages.find((x) => x.status === "in_progress");
  const stalled = s.stages.find((x) => x.stalled);
  let liveEl;
  if (awaiting) {
    liveEl = el("span", { class: "live" }, el("span", { class: "dot" }), "◈ À VALIDER");
  } else if (stalled) {
    liveEl = el("span", { class: "live", style: "color:var(--red)" },
      el("span", { class: "dot", style: "background:var(--red);animation:none" }), "⚠ BLOQUÉ ?");
  } else if (s.live || inProgress) {
    liveEl = el("span", { class: "live" }, el("span", { class: "dot" }), "EN COURS");
  } else {
    liveEl = el("span", { class: "live idle" }, el("span", { class: "dot" }),
      `INACTIF${s.last_activity ? " · " + fmtAgo(s.last_activity).toUpperCase() : ""}`);
  }
  return el("header", { class: "slate" },
    el("div", { class: "clapper" }),
    el("div", {},
      el("a", { class: "wordmark", href: "/", style: "text-decoration:none" }, "OpenDrawer · Backlot"),
      el("h1", {}, s.title),
    ),
    ...chips,
    el("div", { class: "spacer" }),
    renderThemeToggle(),
    liveEl,
  );
}

function stageSub(st) {
  if (st.status === "awaiting_human") return "à valider";
  if (st.status === "in_progress" && st.stalled) return `aucune activité depuis ${st.stalled_minutes} min`;
  if (st.status === "in_progress") return "en cours";
  if (st.status === "failed") return st.error ? String(st.error).slice(0, 60) : "échec";
  if (st.timestamp) return fmtClock(st.timestamp) + (st.gated && st.human_approved ? " · validé" : "");
  return "";
}

function renderRail(s) {
  const rail = el("nav", { class: "rail" });
  let pendingIndex = 1;
  for (const st of s.stages) {
    const cls = st.status === "completed" ? "done"
      : st.status === "in_progress" ? (st.stalled ? "active stalled" : "active")
      : st.status === "awaiting_human" ? "await"
      : st.status === "failed" ? "failed" : "";
    const icon = STAGE_ICONS[st.status] || String(pendingIndex);
    if (!STAGE_ICONS[st.status]) pendingIndex += 1;
    rail.append(el("div", {
      class: `stage ${cls}${selectedStage === st.name ? " selected" : ""}`,
      onclick: () => { selectedStage = selectedStage === st.name ? null : st.name; render(); },
    },
      el("span", { class: "line" }),
      el("span", { class: "node" }, icon),
      el("span", { class: "name" }, STAGE_LABELS[st.name] || st.name),
      el("span", { class: "sub", style: "white-space:pre-line" }, stageSub(st)),
    ));
  }
  return rail;
}

function renderDrawer(s) {
  if (!selectedStage) return null;
  const st = s.stages.find((x) => x.name === selectedStage);
  if (!st) return null;
  const body = el("div", { class: "drawer-body" });
  let shown = false;
  for (const name of st.produces || []) {
    const artifact = s.artifacts[name];
    if (!artifact) continue;
    shown = true;
    body.append(el("div", { class: "d-label" }, name), el("pre", {}, JSON.stringify(artifact, null, 2)));
  }
  if (!shown) body.append(el("div", { class: "hint" }, st.status === "pending" ? "Cette étape n'a pas encore tourné." : "Aucun artefact trouvé pour cette étape."));
  return el("div", { class: "drawer" },
    el("div", { class: "drawer-head" },
      el("h3", {}, `${STAGE_LABELS[st.name] || st.name} — ${STATUS_FR[st.status] || st.status}`),
      st.gate_skipped ? el("span", { class: "gate-chip" }, "⚑ VALIDATION SAUTÉE") : null,
      st.versions > 1 ? el("span", { class: "ver-chip" }, `v${st.versions}`) : null,
      el("span", { class: "close", onclick: () => { selectedStage = null; render(); } }, "FERMER ✕"),
    ),
    body);
}

function renderAwaitingNotice(s) {
  const awaiting = s.stages.find((x) => x.status === "awaiting_human");
  if (!awaiting) return null;
  const what = {
    cartoon: "Choisissez le cartoon à garder (ou demandez une nouvelle série).",
    page: "Vérifiez la page finale avant de la déclarer prête.",
    pages: "Vérifiez l'ensemble des pages du livre.",
    assemble: "Vérifiez le livre assemblé.",
  }[awaiting.name] || "Une validation est attendue.";
  return el("div", { class: "notice" },
    el("span", { style: "font-size:calc(16px * var(--fs-scale))" }, "◈"),
    el("span", {}, el("b", {}, `Étape « ${STAGE_LABELS[awaiting.name] || awaiting.name} » en attente. `), what,
      " Répondez à l'agent dans le chat, ou lancez ", el("code", {}, `python -m opendrawer resume ${s.project_id}`), "."));
}

// ---------------------------------------------------------------------------
// coloring page steps
// ---------------------------------------------------------------------------

function openLightbox(pid, path, label) {
  modal.innerHTML = "";
  modal.append(el("div", { class: "lightbox", onclick: (e) => e.stopPropagation() },
    el("div", { class: "lb-head" },
      el("span", {}, label || path),
      el("a", { href: mediaURL(pid, path), target: "_blank", rel: "noopener" }, "ouvrir l'original ↗"),
      el("span", { class: "close", onclick: closeModal }, "FERMER ✕")),
    el("img", { src: mediaURL(pid, path), alt: label || "" })));
  modal.classList.add("open");
}

function closeModal() { modal.classList.remove("open"); modal.innerHTML = ""; }
modal.addEventListener("click", closeModal);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

function stepImage(pid, img) {
  const badges = [];
  if (img.selected) badges.push(el("span", { class: "badge pick" }, "✓ retenu"));
  if (img.score != null) badges.push(el("span", { class: "badge" }, `score ${Number(img.score).toFixed(2)}`));
  return el("figure", { class: `step-img${img.selected ? " selected" : ""}`, onclick: () => openLightbox(pid, img.path, img.label) },
    el("img", { src: thumbURL(pid, img.path, 640), loading: "lazy", alt: img.label || "" }),
    badges.length ? el("div", { class: "badges" }, badges) : null,
    el("figcaption", {}, img.label || ""));
}

function palette(colors) {
  if (!colors || !colors.length) return null;
  return el("div", { class: "swatches" }, colors.map((c) => el("i", { style: `background:${c}`, title: c })));
}

function renderSteps(s) {
  if (!s.steps || !s.steps.length) return null;
  const stageStatus = Object.fromEntries(s.stages.map((x) => [x.name, x.status]));
  const wrap = el("div", { class: "steps" });
  for (const step of s.steps) {
    const status = stageStatus[step.stage] || "pending";
    if (!step.images.length && status === "pending") continue;
    const head = el("div", { class: "panel-head" },
      el("h2", {}, step.title),
      el("span", { class: `step-status ${status}` }, STATUS_FR[status] || status),
      step.generating ? el("span", { class: "step-status in_progress" }, "génération…") : null);
    const body = el("div", { class: "panel-body" });
    if (step.images.length) {
      body.append(el("div", { class: `step-grid${step.stage === "page" ? " single" : ""}` }, step.images.map((i) => stepImage(s.project_id, i))));
    } else {
      body.append(el("div", { class: "hint" }, status === "in_progress" ? "En cours…" : "Rien encore."));
    }
    const sw = palette(step.palette);
    if (sw) body.append(sw);
    if (step.facts && step.facts.length) body.append(el("ul", { class: "facts" }, step.facts.map((f) => el("li", {}, f))));
    if (step.files && step.files.length) {
      body.append(el("div", { class: "files" }, step.files.map((f) =>
        el("a", { href: mediaURL(s.project_id, f.path), target: "_blank", rel: "noopener", class: "chip" }, f.label))));
    }
    wrap.append(el("section", { class: "panel step" }, head, body));
  }
  return wrap;
}

function renderQA(s) {
  const qa = s.qa;
  if (!qa) return null;
  const icon = { pass: "✓", warn: "!", fail: "✕" };
  return el("div", { class: "panel" },
    el("div", { class: "panel-head" }, el("h2", {}, "Contrôle qualité"),
      el("span", { class: `step-status qa-${qa.status}` }, qa.status)),
    el("div", { class: "panel-body" },
      el("ul", { class: "qa-list" }, (qa.checks || []).map((c) =>
        el("li", { class: `qa-${c.status}` }, el("b", {}, icon[c.status] || "·"), " ", c.message)))));
}

// ---------------------------------------------------------------------------
// coloring book
// ---------------------------------------------------------------------------

function renderBook(s) {
  const book = s.book;
  if (!book) return null;
  const grid = el("div", { class: "book-grid" });
  for (const page of book.pages) {
    const poster = page.poster
      ? el("img", { src: thumbURL(page.poster_project, page.poster, 640), loading: "lazy", alt: "" })
      : el("span", { class: "lp-txt" }, STATUS_FR[page.status] || page.status);
    grid.append(el("a", { class: `book-page status-${page.status}`, href: `/p/${encodeURIComponent(page.project_id)}` },
      el("div", { class: "bp-poster" }, poster),
      el("div", { class: "bp-meta" },
        el("b", {}, `Page ${page.index}`),
        el("span", { class: `step-status ${page.status}` }, STATUS_FR[page.status] || page.status)),
      page.note ? el("div", { class: "hint" }, page.note) : null));
  }
  const sections = [el("div", { class: "panel" },
    el("div", { class: "panel-head" }, el("h2", {}, "Pages du livre"), el("span", { class: "meta" }, `${book.pages.length} page(s)`)),
    el("div", { class: "panel-body" }, grid))];
  if (book.contact_sheet) {
    sections.push(el("div", { class: "panel" },
      el("div", { class: "panel-head" }, el("h2", {}, "Sommaire"), el("span", { class: "meta" }, book.output_dir || "")),
      el("div", { class: "panel-body" }, stepImage(s.project_id, { path: book.contact_sheet, label: "sommaire" }))));
  }
  return el("div", { class: "steps" }, sections);
}

function renderFoundImages(s) {
  if ((s.steps && s.steps.some((x) => x.images.length)) || s.book || !s.images.length) return null;
  return el("div", { class: "panel" },
    el("div", { class: "panel-head" }, el("h2", {}, "Images trouvées"), el("span", { class: "meta" }, "sans checkpoint")),
    el("div", { class: "panel-body" }, el("div", { class: "step-grid" }, s.images.map((i) => stepImage(s.project_id, { path: i.path, label: i.path })))));
}

// ---------------------------------------------------------------------------
// side rail: decisions + activity
// ---------------------------------------------------------------------------

function renderDecisions(s) {
  const decisions = ((s.artifacts.decision_log || {}).decisions) || [];
  if (!decisions.length) return null;
  // A decision revised mid-run keeps (category, subject): show the latest one.
  const current = new Map();
  decisions.forEach((d, i) => {
    const key = `${d.category || "decision"}::${d.subject || ""}`;
    const prev = current.get(key);
    current.set(key, { d, order: i, revised: prev ? prev.revised + 1 : 0 });
  });
  const body = el("div", { class: "panel-body" });
  for (const { d, revised } of [...current.values()].sort((a, b) => b.order - a.order).slice(0, 8)) {
    const opt = (d.options_considered || []).find((o) => (o.option_id ?? o.label) === d.selected);
    body.append(el("div", { class: "decision" },
      el("div", { class: "d-cat" }, d.category || "decision", revised ? el("span", { class: "d-revised" }, " · révisée") : null),
      el("div", { class: "d-pick" }, `${d.subject || ""} `, el("span", { class: "arrow" }, "→"), ` ${(opt && opt.label) || d.selected || ""}`),
      d.reason ? el("div", { class: "d-why" }, d.reason) : null));
  }
  return el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("h2", {}, "Décisions")), body);
}

function renderActivity(s) {
  const events = s.events || [];
  if (!events.length) return null;
  const open = new Map();
  const rows = [];
  for (const ev of events) {
    if (ev.depth) continue; // selector → provider nesting: show the outer call
    if (ev.event === "start") open.set(ev.tool, ev);
    else { open.delete(ev.tool); rows.push(ev); }
  }
  for (const ev of open.values()) rows.push(ev);
  rows.sort((a, b) => String(a.ts).localeCompare(String(b.ts)));
  const body = el("div", { class: "panel-body" });
  for (const ev of rows.slice(-12).reverse()) {
    const status = ev.event === "finish"
      ? el("span", { class: `status ${ev.success === false ? "err" : "ok"}` },
        `${ev.success === false ? "✕" : "✓"}${ev.duration_s != null ? ` ${Number(ev.duration_s).toFixed(1)}s` : ""}`)
      : ev.event === "error" ? el("span", { class: "status err" }, "✕")
      : el("span", { class: "status run" }, "● en cours");
    body.append(el("div", { class: "act-row" },
      el("span", { class: "t" }, fmtClock(ev.ts)), el("span", { class: "tool" }, ev.tool || ""), status));
  }
  return el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("h2", {}, "Activité")), body);
}

// ---------------------------------------------------------------------------
// page assembly
// ---------------------------------------------------------------------------

function render() {
  if (!state) return;
  const s = state;
  document.title = `Backlot — ${s.title}`;
  app.innerHTML = "";
  app.append(renderSlate(s), renderRail(s));
  for (const section of [renderDrawer(s), renderAwaitingNotice(s)]) if (section) app.append(section);

  const main = el("div", { class: "main-col" });
  for (const section of [renderSteps(s), renderBook(s), renderFoundImages(s)]) if (section) main.append(section);
  const aside = el("aside", {});
  for (const section of [renderQA(s), renderDecisions(s), renderActivity(s)]) if (section) aside.append(section);
  const hasAside = aside.childNodes.length > 0;
  app.append(el("div", { class: `board${hasAside ? "" : " solo"}` }, main, hasAside ? aside : null));
}

function normalize(s) {
  s.pipeline = s.pipeline || { pipeline_type: "unknown", stages: [] };
  s.stages = Array.isArray(s.stages) ? s.stages : [];
  s.artifacts = s.artifacts || {};
  s.steps = Array.isArray(s.steps) ? s.steps : [];
  for (const step of s.steps) step.images = Array.isArray(step.images) ? step.images : [];
  s.images = Array.isArray(s.images) ? s.images : [];
  s.events = Array.isArray(s.events) ? s.events : [];
  return s;
}

async function refresh() {
  state = normalize(await getJSON(`/api/project/${encodeURIComponent(projectId)}/state`));
  render();
}

refresh().catch((err) => {
  app.innerHTML = "";
  app.append(el("div", { class: "empty", style: "margin-top:80px" },
    el("div", { class: "big" }, "PROJET INTROUVABLE"), el("div", {}, String(err))));
});
if (!new URLSearchParams(location.search).has("static")) {
  subscribe(`/api/project/${encodeURIComponent(projectId)}/events`, () => refresh().catch(console.error));
}
