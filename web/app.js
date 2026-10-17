// Мини-интерфейс: загрузка видео -> опрос статуса -> вердикт, плеер, карточки нарушений.
const $ = (id) => document.getElementById(id);
const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}.${Math.round((s % 1) * 10)}`;
let pollTimer = null;

function show(id, on = true) { $(id).hidden = !on; }

function reset() {
  clearTimeout(pollTimer);
  history.replaceState(null, "", location.pathname);
  show("upload"); show("status", false); show("result", false); show("again", false);
  $("file").value = "";
}

function upload(file) {
  $("upload-error").hidden = true;
  if (!/\.(mov|mp4|m4v)$/i.test(file.name)) {
    $("upload-error").textContent = "Нужен видеофайл .mov";
    $("upload-error").hidden = false;
    return;
  }
  show("upload", false); show("status"); show("result", false);
  $("status-file").textContent = file.name;
  $("status-stage").textContent = "Загрузка файла…";
  const form = new FormData();
  form.append("file", file);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/ui-api/analyze");
  xhr.upload.onprogress = (e) => { if (e.lengthComputable) setBar(0.3 * e.loaded / e.total); };
  xhr.onload = () => {
    if (xhr.status !== 200) return fail(JSON.parse(xhr.responseText || "{}").detail || "Ошибка загрузки");
    const { job_id } = JSON.parse(xhr.responseText);
    history.replaceState(null, "", `#job=${job_id}`);
    poll(job_id);
  };
  xhr.onerror = () => fail("Сеть недоступна");
  xhr.send(form);
}

function setBar(frac) { $("bar-fill").style.width = `${Math.round(frac * 100)}%`; }

function fail(msg) {
  show("status"); show("again");
  $("status-stage").textContent = `Ошибка: ${msg}`;
  setBar(0);
}

async function poll(jobId) {
  try {
    const r = await fetch(`/ui-api/jobs/${jobId}`);
    if (!r.ok) return fail(r.status === 404 ? "задание не найдено" : `HTTP ${r.status}`);
    const job = await r.json();
    show("status");
    $("status-file").textContent = job.filename || "";
    if (job.status === "processing") {
      $("status-stage").textContent = job.stage || "Анализ видео";
      setBar(0.3 + 0.7 * (job.progress || 0));
      pollTimer = setTimeout(() => poll(jobId), 1500);
    } else if (job.status === "failed") {
      fail(job.error || "обработка не удалась");
    } else {
      setBar(1);
      $("status-stage").textContent = `Готово за ${job.processing_sec ?? "?"} с`;
      show("again");
      render(jobId, job);
    }
  } catch (e) {
    pollTimer = setTimeout(() => poll(jobId), 3000);
  }
}

function render(jobId, job) {
  show("upload", false); show("result");
  const v = $("verdict");
  const found = job.verdict === "violation";
  v.className = `verdict ${found ? "bad" : "ok"}`;
  v.textContent = found
    ? `Обнаружены признаки возможных нарушений: ${job.violations.length}`
    : "Нарушение не обнаружено";

  const video = $("video");
  video.src = `/ui-api/jobs/${jobId}/video`;
  const tl = $("timeline");
  tl.innerHTML = "";
  const dur = job.duration_sec || 1;
  job.violations.forEach((x, i) => {
    const m = document.createElement("button");
    m.className = "mark";
    m.style.left = `${Math.min(100, (100 * x.timecode_sec) / dur)}%`;
    m.title = `${fmt(x.timecode_sec)} — ${x.type}`;
    m.onclick = () => seek(x.timecode_sec, i);
    tl.appendChild(m);
  });

  const list = $("violations");
  list.innerHTML = "";
  if (!found) {
    list.innerHTML = `<div class="card muted">Модель не нашла на видео действий, похожих на нарушения из каталога.</div>`;
  }
  job.violations.forEach((x, i) => list.appendChild(card(x, i)));
}

function seek(t, i) {
  const video = $("video");
  video.currentTime = Math.max(0, t - 1);
  video.play().catch(() => {});
  document.querySelectorAll(".violation").forEach((c, k) => c.classList.toggle("active", k === i));
}

function card(x, i) {
  const el = document.createElement("article");
  el.className = "card violation";
  const verified = { confirmed: "подтверждено проверкой", not_required: "", skipped_deadline: "без проверки (лимит времени)" }[x.verified] || "";
  el.innerHTML = `
    <header>
      <button class="time">${fmt(x.timecode_sec)}</button>
      <div><h3>${esc(x.type)}</h3>
        <div class="meta">${x.type_id || ""}${verified ? " · " + verified : ""}${x.confidence != null ? " · уверенность " + x.confidence : ""}</div></div>
    </header>
    <p>${esc(x.description)}</p>
    <div class="frames">${(x.evidence_frames || []).map((f) =>
      `<figure><img src="data:image/jpeg;base64,${f.image_base64}" alt="кадр ${fmt(f.timestamp_sec)}"><figcaption>${fmt(f.timestamp_sec)}</figcaption></figure>`).join("")}</div>
    ${(x.sources || []).length ? `<details><summary>Нормативная база</summary><ul>${x.sources.map((s) =>
      `<li><a href="${s.url}" target="_blank" rel="noopener">${esc(s.document)}, ${esc(s.clause)}</a>
        <span class="basis">${esc(s.basis || "")}</span><blockquote>${esc(s.quote || "")}</blockquote></li>`).join("")}</ul></details>` : ""}`;
  el.querySelector(".time").onclick = () => seek(x.timecode_sec, i);
  el.querySelectorAll("img").forEach((img) => (img.onclick = () => zoom(img.src)));
  return el;
}

function zoom(src) {
  const d = $("zoom");
  d.querySelector("img").src = src;
  d.showModal();
}

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// --- события
$("file").onchange = (e) => e.target.files[0] && upload(e.target.files[0]);
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));
$("again").onclick = reset;

const m = location.hash.match(/job=([\w-]+)/);
if (m) { show("upload", false); poll(m[1]); }
