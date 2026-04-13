/**
 * main.js — HelpDesk AI · Clasificador de Tickets
 * Gestiona la interacción del formulario, llamadas a la API Flask
 * y la visualización animada de los resultados.
 */

// ── Constantes ────────────────────────────────────────────────────────────────
const API_CLASSIFY = "/api/classify";
const API_HEALTH   = "/api/health";

// ── Referencias al DOM ────────────────────────────────────────────────────────
const statusDot       = document.getElementById("statusDot");
const statusLabel     = document.getElementById("statusLabel");

const ticketIdInput   = document.getElementById("ticketId");
const btnRegenerate   = document.getElementById("btnRegenerate");
const subjectInput    = document.getElementById("subject");
const descriptionInput= document.getElementById("description");
const charCount       = document.getElementById("charCount");

const btnClassify     = document.getElementById("btnClassify");
const chips           = document.querySelectorAll(".chip");

const resultEmpty     = document.getElementById("resultEmpty");
const resultLoading   = document.getElementById("resultLoading");
const resultError     = document.getElementById("resultError");
const resultContent   = document.getElementById("resultContent");
const errorMsg        = document.getElementById("errorMsg");
const btnRetry        = document.getElementById("btnRetry");
const btnNewTicket    = document.getElementById("btnNewTicket");

const resultTicketId  = document.getElementById("resultTicketId");
const categoryCard    = document.getElementById("categoryCard");
const categoryIcon    = document.getElementById("categoryIcon");
const categoryName    = document.getElementById("categoryName");
const categoryLabel   = document.getElementById("categoryLabel");
const confidenceValue = document.getElementById("confidenceValue");
const mainConfBar     = document.getElementById("mainConfidenceBar");
const probList        = document.getElementById("probList");
const tokensUsed      = document.getElementById("tokensUsed");
const processedAt     = document.getElementById("processedAt");


// ── Generador de Ticket IDs ───────────────────────────────────────────────────

/**
 * Genera un ID de ticket con formato TKT-XXXXXXXX.
 * @returns {string}
 */
function generateTicketId() {
  const chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  let id = "TKT-";
  for (let i = 0; i < 8; i++) {
    id += chars[Math.floor(Math.random() * chars.length)];
  }
  return id;
}

function refreshTicketId() {
  ticketIdInput.value = generateTicketId();
}


// ── Health Check ──────────────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const res  = await fetch(API_HEALTH);
    const data = await res.json();

    if (data.status === "ok") {
      statusDot.className   = "status-dot ok";
      statusLabel.textContent = "Modelo listo";
    } else {
      statusDot.className   = "status-dot error";
      statusLabel.textContent = "Modelo no disponible";
    }
  } catch {
    statusDot.className   = "status-dot error";
    statusLabel.textContent = "Sin conexión";
  }
}


// ── Clasificación ─────────────────────────────────────────────────────────────

async function classifyTicket() {
  const subject     = subjectInput.value.trim();
  const description = descriptionInput.value.trim();
  const ticketId    = ticketIdInput.value.trim();

  // Validación mínima en el cliente
  if (!description) {
    descriptionInput.focus();
    descriptionInput.style.borderColor = "var(--red)";
    setTimeout(() => (descriptionInput.style.borderColor = ""), 1500);
    return;
  }

  // Cambiar UI al estado de carga
  setResultState("loading");
  setButtonLoading(true);

  try {
    const payload = { subject, description, ticket_id: ticketId };

    const res  = await fetch(API_CLASSIFY, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || `Error ${res.status}`);
    }

    renderResult(data);
    setResultState("result");

  } catch (err) {
    errorMsg.textContent = err.message;
    setResultState("error");
  } finally {
    setButtonLoading(false);
  }
}


// ── Renderizado del resultado ─────────────────────────────────────────────────

/**
 * Renderiza los datos de clasificación en el panel de resultados.
 * @param {Object} data  Respuesta JSON del endpoint /api/classify
 */
function renderResult(data) {
  const { ticket_id, category, confidence_pct, probabilities, meta, tokens_used, processed_at } = data;

  // Ticket ID y timestamp
  resultTicketId.textContent = ticket_id;

  // Categoría principal
  const color = meta?.color || "#22d3ee";
  categoryCard.style.setProperty("--category-color", color);
  categoryIcon.textContent           = meta?.icon  || "🎫";
  categoryName.textContent           = category;
  categoryName.style.color           = color;
  categoryLabel.textContent          = meta?.label || category;
  confidenceValue.textContent        = `${confidence_pct.toFixed(1)}%`;

  // Barra de confianza principal (animada)
  mainConfBar.style.width = "0%";
  requestAnimationFrame(() => {
    setTimeout(() => {
      mainConfBar.style.width = `${Math.min(confidence_pct, 100)}%`;
    }, 100);
  });

  // Lista de probabilidades
  probList.innerHTML = "";

  probabilities.forEach((item, idx) => {
    const isTop     = item.class === category;
    const barColor  = item.meta?.color || "#22d3ee";
    const pct       = (item.percentage).toFixed(2);

    const el = document.createElement("div");
    el.className = "prob-item";
    el.style.animationDelay = `${idx * 60}ms`;

    el.innerHTML = `
      <span class="prob-item__icon">${item.meta?.icon || "•"}</span>
      <div class="prob-item__bar-wrapper">
        <div class="prob-item__label" style="${isTop ? `color:${barColor};font-weight:600` : ""}">${item.class}</div>
        <div class="prob-item__bar-track">
          <div
            class="prob-item__bar-fill"
            style="background:${barColor}; opacity:${isTop ? 1 : 0.45}"
            data-pct="${pct}"
          ></div>
        </div>
      </div>
      <span class="prob-item__pct">${pct}%</span>
    `;
    probList.appendChild(el);
  });

  // Animar barras de probabilidad
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.querySelectorAll(".prob-item__bar-fill").forEach(bar => {
        bar.style.width = `${bar.dataset.pct}%`;
      });
    }, 200);
  });

  // Chips de análisis
  tokensUsed.textContent  = tokens_used ?? "—";
  processedAt.textContent = formatTimestamp(processed_at);
}


// ── Helpers de UI ─────────────────────────────────────────────────────────────

/**
 * Controla qué estado muestra el panel de resultados.
 * @param {"empty"|"loading"|"error"|"result"} state
 */
function setResultState(state) {
  resultEmpty.classList.add("hidden");
  resultLoading.classList.add("hidden");
  resultError.classList.add("hidden");
  resultContent.classList.add("hidden");

  if (state === "empty")   resultEmpty.classList.remove("hidden");
  if (state === "loading") resultLoading.classList.remove("hidden");
  if (state === "error")   resultError.classList.remove("hidden");
  if (state === "result")  resultContent.classList.remove("hidden");
}

function setButtonLoading(loading) {
  btnClassify.disabled = loading;
  btnClassify.classList.toggle("loading", loading);
}

/**
 * Formatea un timestamp ISO a hora local legible.
 * @param {string} iso
 * @returns {string}
 */
function formatTimestamp(iso) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString("es-GT", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch {
    return iso;
  }
}

function resetForm() {
  subjectInput.value      = "";
  descriptionInput.value  = "";
  charCount.textContent   = "0";
  refreshTicketId();
  setResultState("empty");
  descriptionInput.focus();
}


// ── Event Listeners ───────────────────────────────────────────────────────────

// Actualizar contador de caracteres
descriptionInput.addEventListener("input", () => {
  charCount.textContent = descriptionInput.value.length;
});

// Regenerar ticket ID
btnRegenerate.addEventListener("click", refreshTicketId);

// Clasificar al presionar el botón
btnClassify.addEventListener("click", classifyTicket);

// Clasificar con Ctrl+Enter en el textarea
descriptionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) classifyTicket();
});

// Chips de ejemplos rápidos
chips.forEach(chip => {
  chip.addEventListener("click", () => {
    descriptionInput.value  = chip.dataset.example;
    charCount.textContent   = descriptionInput.value.length;
    descriptionInput.scrollIntoView({ behavior: "smooth", block: "center" });
  });
});

// Reintentar en caso de error
btnRetry.addEventListener("click", classifyTicket);

// Nuevo ticket
btnNewTicket.addEventListener("click", resetForm);


// ── Inicialización ────────────────────────────────────────────────────────────

(function init() {
  refreshTicketId();
  checkHealth();

  // Re-verificar estado del modelo cada 30 segundos
  setInterval(checkHealth, 30_000);
})();