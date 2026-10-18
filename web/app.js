// Интерфейс проверки: загрузка видео -> опрос статуса -> вердикт, плеер, карточки эпизодов; раздел «Как это работает».
const $ = (id) => document.getElementById(id);
const fmt = (s) => {
  s = Math.max(0, s || 0);
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}.${Math.floor((s % 1) * 10)}`;
};
const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const VERIFIED = {
  confirmed: ["ok", "подтверждено повторной проверкой"],
  skipped_deadline: ["warn", "без повторной проверки (лимит времени)"],
};
let pollTimer = null, tickTimer = null, startedAt = 0;
let current = null; // job_id, который сейчас показан или опрашивается

function show(id, on = true) { $(id).hidden = !on; }

// ---------- навигация: #job=<id> — результат, #about — методика, пусто — загрузка ----------

function route() {
  const about = location.hash === "#about";
  show("view-check", !about); show("view-about", about);
  document.querySelectorAll("nav a[data-view]").forEach((a) => a.classList.toggle("on", (a.dataset.view === "about") === about));
  if (about) return loadTypes();
  const m = location.hash.match(/job=([\w-]+)/);
  if (m && m[1] !== current) { setCurrent(m[1]); show("upload", false); poll(m[1]); }
  else if (!m && !current) reset();
}

function setCurrent(jobId) {
  current = jobId;
  // «Проверка видео» в меню возвращает к текущему результату, а не к пустой форме
  document.querySelectorAll('nav a[data-view="check"], a.brand').forEach((a) => (a.href = jobId ? `#job=${jobId}` : "#"));
}

function reset() {
  clearTimeout(pollTimer); clearInterval(tickTimer); pollTimer = null; tickTimer = null;
  setCurrent(null);
  if (location.hash.startsWith("#job=")) history.replaceState(null, "", location.pathname);
  show("upload"); show("status", false); show("result", false); show("again", false);
  $("file").value = ""; $("status-time").textContent = "";
  const v = $("video"); v.removeAttribute("src"); v.load();
}

// ---------- загрузка и статус ----------

function parse(text) { try { return JSON.parse(text); } catch { return {}; } }

function upload(file) {
  $("upload-error").hidden = true;
  if (!/\.(mov|mp4|m4v)$/i.test(file.name)) {
    $("upload-error").textContent = "Нужен видеофайл в формате .mov";
    $("upload-error").hidden = false;
    return;
  }
  show("upload", false); show("status"); show("result", false); show("again", false);
  $("status-file").textContent = file.name;
  setStage("Загрузка файла…", 0);
  startTimer(Date.now());
  const form = new FormData();
  form.append("file", file);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/ui-api/analyze");
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) setStage(`Загрузка файла… ${Math.round((100 * e.loaded) / e.total)}%`, (0.25 * e.loaded) / e.total);
  };
  xhr.onload = () => {
    const body = parse(xhr.responseText);
    if (xhr.status !== 200) {
      const msg = body.detail || (xhr.status === 413 ? "файл слишком большой" : `сервер ответил ${xhr.status}`);
      return fail(msg, true);
    }
    history.replaceState(null, "", `#job=${body.job_id}`);
    setCurrent(body.job_id);
    poll(body.job_id);
  };
  xhr.onerror = () => fail("нет связи с сервером", true);
  xhr.send(form);
}

function setStage(text, frac) {
  $("status-stage").textContent = text;
  if (frac != null) $("bar-fill").style.width = `${Math.round(Math.min(1, frac) * 100)}%`;
}

function startTimer(t0) {
  startedAt = t0;
  clearInterval(tickTimer);
  const tick = () => ($("status-time").textContent = clock((Date.now() - startedAt) / 1000));
  tick(); tickTimer = setInterval(tick, 1000);
}

function fail(msg, backToUpload = false) {
  clearInterval(tickTimer); clearTimeout(pollTimer); pollTimer = null; tickTimer = null;
  if (backToUpload) {
    // файл не принят — ошибка прямо под формой, можно сразу выбрать другой
    setCurrent(null);
    show("status", false); show("upload");
    $("file").value = "";
    $("upload-error").textContent = `Ошибка: ${msg}`;
    $("upload-error").hidden = false;
    return;
  }
  show("status"); show("again");
  $("status-stage").innerHTML = `<span class="error">Ошибка: ${esc(msg)}</span>`;
  $("bar-fill").style.width = "0";
}

async function poll(jobId, retries = 0) {
  pollTimer = null;
  let job;
  try {
    const r = await fetch(`/ui-api/jobs/${jobId}`, { cache: "no-store" });
    if (r.status === 404) return fail("задание не найдено — возможно, оно удалено по сроку хранения");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    job = await r.json();
  } catch (e) {
    // временный обрыв связи: продолжаем опрашивать, задание на сервере не теряется
    setStage(`Нет связи с сервером, повтор… (${retries + 1})`);
    pollTimer = setTimeout(() => poll(jobId, retries + 1), Math.min(10000, 2000 * (retries + 1)));
    return;
  }
  show("status");
  $("status-file").textContent = job.filename || "";
  if (!tickTimer && job.status === "processing") startTimer(Date.now() - (job.elapsed_sec || 0) * 1000);
  if (job.status === "processing") {
    setStage(job.stage || "Анализ видео", 0.25 + 0.75 * (job.progress || 0));
    pollTimer = setTimeout(() => poll(jobId), 1500);
  } else if (job.status === "failed") {
    fail(job.error || "обработка не удалась");
  } else {
    clearInterval(tickTimer); tickTimer = null;
    $("status-time").textContent = "";
    setStage(`Готово: обработка заняла ${Math.round(job.processing_sec ?? job.elapsed_sec ?? 0)} с`, 1);
    show("again");
    render(jobId, job);
  }
}

// ---------- результат ----------

function render(jobId, job) {
  show("upload", false); show("result");
  const found = job.verdict === "violation";
  const n = job.violations.length;
  $("verdict").className = `verdict ${found ? "bad" : "ok"}`;
  $("verdict-title").textContent = found ? "Обнаружены признаки возможных нарушений" : "Нарушение не обнаружено";
  $("verdict-meta").textContent = [
    job.duration_sec ? `видео ${clock(job.duration_sec)}` : "",
    found ? `эпизодов: ${n}` : "",
  ].filter(Boolean).join(" · ");

  const video = $("video");
  show("video-note", false);
  video.onerror = () => show("video-note");
  video.src = `/ui-api/jobs/${jobId}/video`;

  const tl = $("timeline");
  tl.innerHTML = "";
  show("timeline", found); show("timeline-hint", found);
  const dur = job.duration_sec || 1;
  job.violations.forEach((x, i) => {
    const m = document.createElement("button");
    m.className = "mark";
    m.style.left = `${Math.min(100, (100 * x.timecode_sec) / dur)}%`;
    m.title = `${fmt(x.timecode_sec)} — ${x.type}`;
    m.setAttribute("aria-label", m.title);
    m.onclick = () => focusEpisode(i, x.timecode_sec);
    tl.appendChild(m);
  });

  const list = $("violations");
  list.innerHTML = "";
  if (!found) {
    list.innerHTML = `<div class="card empty"><h3>Эпизодов, похожих на нарушения, не найдено</h3>
      <p class="muted">Модель проверила видео по всем типам из <a href="#about">перечня</a>. Результат — подсказка:
      при сомнениях просмотрите запись.</p></div>`;
  }
  job.violations.forEach((x, i) => list.appendChild(card(x, i)));
}

function seek(t) {
  const video = $("video");
  video.currentTime = Math.max(0, t - 1);
  video.play().catch(() => {});
}

function focusEpisode(i, t) {
  seek(t);
  document.querySelectorAll(".violation").forEach((c, k) => c.classList.toggle("active", k === i));
  const el = $(`ep-${i}`);
  if (el) el.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function card(x, i) {
  const el = document.createElement("article");
  el.className = "card violation";
  el.id = `ep-${i}`;
  const badge = VERIFIED[x.verified];
  const frames = [...(x.evidence_frames || [])].sort((a, b) => a.timestamp_sec - b.timestamp_sec);
  const sources = x.sources || [];
  el.innerHTML = `
    <header>
      <button class="time" title="Перейти к моменту">${fmt(x.timecode_sec)}</button>
      <div>
        <h3>${esc(x.type)}</h3>
        <div class="badges">${badge ? `<span class="badge ${badge[0]}">${badge[1]}</span>` : ""}
          ${x.confidence != null ? `<span class="badge plain">уверенность модели ${Math.round(x.confidence * 100)}%</span>` : ""}</div>
      </div>
    </header>
    <dl>
      ${x.observation ? `<dt>Что видно на видео</dt><dd>${esc(x.observation)}</dd>` : ""}
      <dt>Почему это может быть нарушением</dt><dd>${esc(x.why || x.description)}</dd>
      <dt>Правило и источник</dt>
      <dd>${sources.length ? `<ul class="sources">${sources.map((s) => `
        <li><a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.document)}</a>${s.clause ? `, ${esc(s.clause)}` : ""}
          ${s.basis ? `<span class="basis">${esc(s.basis)}</span>` : ""}
          ${s.quote ? `<blockquote>«${esc(s.quote)}»</blockquote>` : ""}</li>`).join("")}</ul>`
        : `<span class="muted">Точный пункт не определён: эпизод отмечен как иное возможное нарушение и требует оценки проверяющим.</span>`}</dd>
    </dl>
    <div class="frames">${frames.map((f) => `
      <figure>
        <img src="data:image/jpeg;base64,${f.image_base64}" alt="Кадр ${fmt(f.timestamp_sec)}" loading="lazy">
        <figcaption><button class="link" data-t="${f.timestamp_sec}">▶ ${fmt(f.timestamp_sec)}</button>
          ${Math.abs(f.timestamp_sec - x.timecode_sec) < 0.3 ? "· момент события" : f.timestamp_sec < x.timecode_sec ? "· до" : "· после"}</figcaption>
      </figure>`).join("")}</div>`;
  el.querySelector(".time").onclick = () => focusEpisode(i, x.timecode_sec);
  el.querySelectorAll("figcaption .link").forEach((b) => (b.onclick = () => seek(Number(b.dataset.t) + 1)));
  el.querySelectorAll("img").forEach((img) => (img.onclick = () => zoom(img)));
  return el;
}

function zoom(img) {
  const d = $("zoom");
  d.querySelector("img").src = img.src;
  d.querySelector("img").alt = img.alt;
  d.showModal();
}

// ---------- раздел «Как это работает» ----------

let typesLoaded = false;
async function loadTypes() {
  document.querySelectorAll(".origin").forEach((e) => (e.textContent = location.origin));
  if (typesLoaded) return;
  try {
    const r = await fetch("/ui-api/types");
    if (!r.ok) throw new Error(r.status);
    const types = await r.json();
    const docs = new Map();
    $("types").innerHTML = Object.entries(types).map(([id, t]) => {
      (t.sources || []).forEach((s) => docs.set(s.url + "|" + s.document, s));
      return `<div class="type">
        <h3>${esc(t.name)}</h3>
        <p>${esc(t.why)}</p>
        ${(t.sources || []).length ? `<p class="small">${t.sources.map((s) => `${esc(s.document)}${s.clause ? `, ${esc(s.clause)}` : ""}
          ${s.basis ? `<span class="basis">${esc(s.basis)}</span>` : ""}`).join("<br>")}</p>`
          : `<p class="small muted">Для эпизодов вне перечня пункт не подставляется: их оценивает проверяющий.</p>`}
      </div>`;
    }).join("");
    $("docs").innerHTML = [...new Map([...docs.values()].map((s) => [s.url, s])).values()].map((s) =>
      `<li><a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.document)}</a> — adilet.zan.kz</li>`).join("");
    typesLoaded = true;
  } catch {
    $("types").innerHTML = `<p class="error">Не удалось загрузить перечень типов.</p>`;
  }
}

// ---------- события ----------

$("file").onchange = (e) => e.target.files[0] && upload(e.target.files[0]);
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));
$("again").onclick = () => { reset(); };
window.addEventListener("hashchange", route);
route();
