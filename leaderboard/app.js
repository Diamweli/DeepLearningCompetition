const CSV_URL = "./leaderboard.csv";
const UPLOAD_URL = "/api/upload";
const REFRESH_INTERVAL_MS = 120000;

const state = {
  rows: [],
  sortKey: "accuracy",
  sortDir: "desc",
  lastRects: new Map(),
  selectedFile: null,
};

const tableBody = document.getElementById("leaderboard-body");
const searchInput = document.getElementById("search-input");
const periodFilter = document.getElementById("period-filter");
const refreshButton = document.getElementById("refresh-button");
const lastUpdated = document.getElementById("last-updated");
const rowCount = document.getElementById("row-count");
const headerCells = document.querySelectorAll("thead th");

// Upload elements
const uploadToggle = document.getElementById("upload-toggle");
const uploadModal = document.getElementById("upload-modal");
const modalClose = document.getElementById("modal-close");
const uploadForm = document.getElementById("upload-form");
const teamNameInput = document.getElementById("team-name");
const predictionsFile = document.getElementById("predictions-file");
const dropZone = document.getElementById("drop-zone");
const fileInfo = document.getElementById("file-info");
const fileName = document.getElementById("file-name");
const fileRemove = document.getElementById("file-remove");
const uploadStatus = document.getElementById("upload-status");
const submitBtn = document.getElementById("submit-btn");
const submitText = document.getElementById("submit-text");
const submitSpinner = document.getElementById("submit-spinner");

// ─── CSV Parsing ────────────────────────────────────────

const parseCsv = (text) => {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length <= 1) {
    return [];
  }
  const [, ...data] = lines;
  return data
    .filter((line) => line.trim().length > 0)
    .map((line) => line.split(",").map((f) => f.trim()))
    .filter((parts) => parts.length >= 3 && parts[0].length > 0)
    .map(([team, accuracy, submittedAt]) => ({
      team,
      accuracy: Number.parseFloat(accuracy),
      submittedAt: new Date(submittedAt),
    }));
};

const formatDate = (date) => {
  if (Number.isNaN(date.getTime())) {
    return "—";
  }
  return new Intl.DateTimeFormat("fr-FR", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
};

const formatScore = (score) => {
  if (Number.isNaN(score)) {
    return "—";
  }
  return `${(score * 100).toFixed(2)}%`;
};

// ─── Filtering & Sorting ────────────────────────────────

const applyFilters = (rows) => {
  const query = searchInput.value.trim().toLowerCase();
  const period = periodFilter.value;
  const now = Date.now();
  let cutoff = 0;
  if (period === "24h") {
    cutoff = now - 24 * 60 * 60 * 1000;
  } else if (period === "7d") {
    cutoff = now - 7 * 24 * 60 * 60 * 1000;
  } else if (period === "30d") {
    cutoff = now - 30 * 24 * 60 * 60 * 1000;
  }
  return rows.filter((row) => {
    const matchQuery = !query || row.team.toLowerCase().includes(query);
    const matchPeriod =
      !cutoff || (row.submittedAt && row.submittedAt.getTime() >= cutoff);
    return matchQuery && matchPeriod;
  });
};

const sortRows = (rows) => {
  const dir = state.sortDir === "asc" ? 1 : -1;
  const key = state.sortKey;
  return [...rows].sort((a, b) => {
    let left = a[key];
    let right = b[key];
    if (key === "submitted_at") {
      left = a.submittedAt;
      right = b.submittedAt;
    }
    if (left instanceof Date && right instanceof Date) {
      return (left.getTime() - right.getTime()) * dir;
    }
    if (typeof left === "string" && typeof right === "string") {
      return left.localeCompare(right) * dir;
    }
    return (left - right) * dir;
  });
};

// ─── Badges ─────────────────────────────────────────────

const buildBadge = (rank) => {
  if (rank === 1) {
    return '<span class="badge gold">Top 1</span>';
  }
  if (rank === 2) {
    return '<span class="badge silver">Top 2</span>';
  }
  if (rank === 3) {
    return '<span class="badge bronze">Top 3</span>';
  }
  return "";
};

// ─── Table Rendering ────────────────────────────────────

const renderTable = () => {
  const filtered = applyFilters(state.rows);
  const sorted = sortRows(filtered);
  const maxScore = sorted.reduce((max, row) => Math.max(max, row.accuracy || 0), 0);

  const previousRects = state.lastRects;
  tableBody.innerHTML = "";

  const fragment = document.createDocumentFragment();
  sorted.forEach((row, index) => {
    const rank = index + 1;
    const tr = document.createElement("tr");
    tr.dataset.id = row.team;
    tr.innerHTML = `
      <td class="rank">${rank}</td>
      <td>${row.team} ${buildBadge(rank)}</td>
      <td class="score">${formatScore(row.accuracy)}</td>
      <td>${formatDate(row.submittedAt)}</td>
      <td>
        <div class="progress" aria-label="Progression">
          <span style="width: ${maxScore ? (row.accuracy / maxScore) * 100 : 0}%"></span>
        </div>
      </td>
    `;
    fragment.appendChild(tr);
  });

  tableBody.appendChild(fragment);
  rowCount.textContent = `Participants: ${sorted.length}`;

  const newRects = new Map();
  Array.from(tableBody.querySelectorAll("tr")).forEach((row) => {
    newRects.set(row.dataset.id, row.getBoundingClientRect());
  });

  Array.from(tableBody.querySelectorAll("tr")).forEach((row) => {
    const prev = previousRects.get(row.dataset.id);
    const next = newRects.get(row.dataset.id);
    if (!prev || !next) {
      return;
    }
    const deltaY = prev.top - next.top;
    if (deltaY) {
      row.style.transform = `translateY(${deltaY}px)`;
      row.style.transition = "transform 0s";
      requestAnimationFrame(() => {
        row.style.transition = "transform 300ms ease";
        row.style.transform = "";
      });
    }
  });

  state.lastRects = newRects;
};

// ─── Data Fetching ──────────────────────────────────────

const updateData = async () => {
  try {
    const errorMsg = document.getElementById("error-message");
    if (errorMsg) errorMsg.style.display = "none";

    const response = await fetch(`${CSV_URL}?t=${Date.now()}`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const text = await response.text();
    state.rows = parseCsv(text);
    lastUpdated.textContent = `Dernière mise à jour: ${new Date().toLocaleString("fr-FR")}`;
    renderTable();
  } catch (error) {
    console.error("Erreur lors du chargement des données:", error);
    const errorMsg = document.getElementById("error-message");
    if (errorMsg) {
      errorMsg.style.display = "block";
      errorMsg.textContent = `Impossible de charger le classement. Erreur: ${error.message}`;
    }
  }
};

// ─── Sorting ────────────────────────────────────────────

const handleSort = (event) => {
  const key = event.target.getAttribute("data-key");
  if (!key || key === "progress" || key === "rank") {
    return;
  }
  if (state.sortKey === key) {
    state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
  } else {
    state.sortKey = key === "submitted_at" ? "submitted_at" : key;
    state.sortDir = key === "team" ? "asc" : "desc";
  }
  renderTable();
};

// ─── Upload Modal ───────────────────────────────────────

const openModal = () => {
  uploadModal.style.display = "flex";
  uploadStatus.style.display = "none";
  document.body.style.overflow = "hidden";
};

const closeModal = () => {
  uploadModal.style.display = "none";
  document.body.style.overflow = "";
};

const updateSubmitButton = () => {
  const hasTeam = teamNameInput.value.trim().length > 0;
  const hasFile = state.selectedFile != null;
  submitBtn.disabled = !(hasTeam && hasFile);
};

const selectFile = (file) => {
  state.selectedFile = file;
  fileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  fileInfo.style.display = "flex";
  dropZone.style.display = "none";
  updateSubmitButton();
};

const removeFile = () => {
  state.selectedFile = null;
  predictionsFile.value = "";
  fileInfo.style.display = "none";
  dropZone.style.display = "block";
  updateSubmitButton();
};

const showUploadStatus = (type, html) => {
  uploadStatus.style.display = "block";
  uploadStatus.className = `upload-status ${type}`;
  uploadStatus.innerHTML = html;
};

const handleUpload = async (e) => {
  e.preventDefault();

  const team = teamNameInput.value.trim();
  const file = state.selectedFile;

  if (!team || !file) return;

  // Show loading
  submitBtn.disabled = true;
  submitText.textContent = "Évaluation en cours...";
  submitSpinner.style.display = "inline-block";
  uploadStatus.style.display = "none";

  const formData = new FormData();
  formData.append("team", team);
  formData.append("predictions", file);

  try {
    const response = await fetch(UPLOAD_URL, {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (response.ok && result.success) {
      showUploadStatus(
        "success",
        `<strong>${result.team}</strong> — Évaluation réussie !
        <span class="result-score">${result.accuracy_pct}</span>
        <span class="result-rank">Classement: ${result.rank}/${result.total_participants}</span>`
      );
      // Refresh the table
      await updateData();
    } else {
      showUploadStatus("error", `<strong>Erreur:</strong> ${result.error || "Erreur inconnue"}`);
    }
  } catch (err) {
    showUploadStatus(
      "error",
      `<strong>Erreur de connexion:</strong> Vérifiez que le serveur tourne avec <code>python leaderboard/server.py</code>`
    );
  }

  // Reset button
  submitBtn.disabled = false;
  submitText.textContent = "Évaluer et soumettre";
  submitSpinner.style.display = "none";
  updateSubmitButton();
};

// ─── Event Listeners ────────────────────────────────────

headerCells.forEach((cell) => {
  cell.addEventListener("click", handleSort);
});

searchInput.addEventListener("input", renderTable);
periodFilter.addEventListener("change", renderTable);
refreshButton.addEventListener("click", updateData);

// Upload events
uploadToggle.addEventListener("click", openModal);
modalClose.addEventListener("click", closeModal);
uploadModal.addEventListener("click", (e) => {
  if (e.target === uploadModal) closeModal();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeModal();
});

teamNameInput.addEventListener("input", updateSubmitButton);

dropZone.addEventListener("click", () => predictionsFile.click());
predictionsFile.addEventListener("change", (e) => {
  if (e.target.files[0]) selectFile(e.target.files[0]);
});
fileRemove.addEventListener("click", removeFile);

// Drag and drop
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0]);
});

uploadForm.addEventListener("submit", handleUpload);

// ─── Init ───────────────────────────────────────────────

updateData();
setInterval(updateData, REFRESH_INTERVAL_MS);
