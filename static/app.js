/* RepoJudge frontend. Plain JavaScript, no frameworks.
   All repository text is inserted with textContent so nothing from a judged
   repository is ever treated as HTML. */
(function () {
  "use strict";

  var form = document.getElementById("analyze-form");
  var urlInput = document.getElementById("repo-url");
  var reqInput = document.getElementById("requirements");
  var button = document.getElementById("analyze-button");

  var loadingEl = document.getElementById("loading");
  var stepsEl = document.getElementById("steps");
  var errorEl = document.getElementById("error");
  var errorTitle = document.getElementById("error-title");
  var errorMessage = document.getElementById("error-message");
  var resultEl = document.getElementById("result");

  var currentReport = null;
  var stepTimer = null;

  // Friendly titles for the error codes the API returns.
  var ERROR_TITLES = {
    gemma_timeout: "Gemma timed out",
    gemma_rate_limit: "Gemma is rate limited",
    gemma_unavailable: "Gemma is unavailable",
    repo_not_found: "Repository not found",
    private_repo: "This repository is private",
    no_readme: "This repository has no README",
    invalid_url: "That URL is not supported",
    github_rate_limit: "GitHub is rate limiting us"
  };

  // --- helpers ------------------------------------------------------------

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  // Friendly names for the engine statuses that are not plain words.
  var STATUS_LABELS = {
    not_checked: "not checked",
    not_applicable: "not applicable",
    "Unverified": "Unverified (no proof found in the code read)",
    "Not judged": "Not judged"
  };

  function statusClass(status) {
    var key = String(status || "").toLowerCase().replace(/\s+/g, "_");
    return "pill pill-" + key;
  }

  function statusPill(status) {
    var key = String(status || "");
    return el("span", statusClass(key), STATUS_LABELS[key] || key);
  }

  function evidenceCode(hit) {
    // A finding with no file to point at (for example "there is no .gitignore")
    // is shown as "none" rather than an invented location.
    var location = hit.file || "none";
    if (hit.file && hit.line) location += ":" + hit.line;
    var code = el("code", "evidence", location);
    if (hit.snippet) code.title = hit.snippet;
    return code;
  }

  // --- loading steps ------------------------------------------------------

  var STEP_IDS = ["fetch", "claims", "proof", "score"];

  function resetSteps() {
    Array.prototype.forEach.call(stepsEl.children, function (li) {
      li.classList.remove("active", "done");
    });
    stepsEl.children[0].classList.add("active");
  }

  function startSteps() {
    resetSteps();
    var index = 0;
    window.clearInterval(stepTimer);
    stepTimer = window.setInterval(function () {
      if (index >= STEP_IDS.length - 1) return;
      stepsEl.children[index].classList.remove("active");
      stepsEl.children[index].classList.add("done");
      index += 1;
      stepsEl.children[index].classList.add("active");
    }, 1100);
  }

  function finishSteps() {
    window.clearInterval(stepTimer);
    Array.prototype.forEach.call(stepsEl.children, function (li) {
      li.classList.remove("active");
      li.classList.add("done");
    });
  }

  // --- render -------------------------------------------------------------

  function renderBanners(banners) {
    var container = document.getElementById("banners");
    clear(container);
    (banners || []).forEach(function (text) {
      container.appendChild(el("div", "banner", text));
    });
  }

  function renderBreakdown(score) {
    var container = document.getElementById("breakdown");
    clear(container);
    var parts = [
      ["Claims", score.breakdown.claims],
      ["Engineering quality", score.breakdown.quality],
      ["Repository health", score.breakdown.health]
    ];
    parts.forEach(function (entry) {
      var label = entry[0];
      var part = entry[1] || { score: 0, max: 0 };
      var pct = part.max ? Math.max(0, Math.min(100, (part.score / part.max) * 100)) : 0;

      var row = el("div", "bar-row");
      var head = el("div", "bar-head");
      head.appendChild(el("span", null, label));
      head.appendChild(el("span", null, part.score + " / " + part.max));
      var track = el("div", "bar-track");
      var fill = el("div", "bar-fill");
      fill.style.width = pct + "%";
      track.appendChild(fill);
      row.appendChild(head);
      row.appendChild(track);
      if (part.reason) row.appendChild(el("div", "bar-note", part.reason));
      container.appendChild(row);
    });
  }

  function renderClaims(claims, summary) {
    var body = document.getElementById("claims-body");
    clear(body);
    var summaryText =
      summary.verified + " verified, " + summary.partial + " partial, " +
      summary.unverified + " unverified";
    if (summary.not_judged) {
      summaryText += ", " + summary.not_judged + " not judged";
    }
    summaryText += " (out of " + summary.total + ").";
    document.getElementById("claims-summary").textContent = summaryText;

    if (!claims.length) {
      var emptyRow = el("tr");
      var cell = el("td", "muted", "No claims were found in the README.");
      cell.colSpan = 5;
      emptyRow.appendChild(cell);
      body.appendChild(emptyRow);
      return;
    }

    claims.forEach(function (claim) {
      var tr = el("tr");
      tr.appendChild(el("td", null, claim.claim));
      tr.appendChild(el("td", "muted", claim.type || ""));

      var statusCell = el("td");
      statusCell.appendChild(statusPill(claim.status));
      tr.appendChild(statusCell);

      tr.appendChild(el("td", null, claim.reason || ""));

      var evidenceCell = el("td");
      var evidence = claim.evidence || [];
      if (evidence.length) {
        evidence.forEach(function (hit, index) {
          if (index > 0) evidenceCell.appendChild(document.createTextNode(" "));
          evidenceCell.appendChild(evidenceCode(hit));
        });
      } else {
        evidenceCell.appendChild(el("span", "muted", "none"));
      }
      tr.appendChild(evidenceCell);

      body.appendChild(tr);
    });
  }

  function renderCheckList(nodeId, items, includePoints) {
    var list = document.getElementById(nodeId);
    clear(list);
    (items || []).forEach(function (item) {
      var li = el("li", "check-item");
      li.appendChild(statusPill(item.status));

      var body = el("div");
      var title = item.name || item.id || "";
      if (includePoints) {
        title += " — " + item.earned + " / " + item.max + " points";
      }
      body.appendChild(el("div", "check-title", title));
      if (item.reason) body.appendChild(el("div", "check-reason", item.reason));

      var evidence = item.evidence || [];
      if (evidence.length) {
        var wrap = el("div", "check-evidence");
        evidence.forEach(function (hit) {
          wrap.appendChild(evidenceCode(hit));
        });
        body.appendChild(wrap);
      }

      li.appendChild(body);
      list.appendChild(li);
    });
  }

  function renderFiles(files) {
    files = files || {};
    var readList = document.getElementById("files-read");
    var skippedList = document.getElementById("files-skipped");
    clear(readList);
    clear(skippedList);

    var read = files.read || [];
    var skipped = files.skipped || [];
    document.getElementById("files-summary").textContent =
      "Read " + read.length + " file(s), skipped " + skipped.length +
      ". Every skipped file is listed with its reason.";

    if (!read.length) {
      readList.appendChild(el("li", "muted", "No files were read."));
    }
    read.forEach(function (entry) {
      var li = el("li", "file-item");
      var path = el("code", "file-path", entry.path);
      if (entry.reason) path.title = entry.reason;
      li.appendChild(path);
      var note = (entry.size || 0) + " bytes, " + (entry.lines || 0) + " lines";
      if (entry.truncated) note += ", read in part";
      li.appendChild(el("span", "file-note", " — " + note));
      if (entry.reason) li.appendChild(el("span", "file-reason", " · " + entry.reason));
      readList.appendChild(li);
    });

    if (!skipped.length) {
      skippedList.appendChild(el("li", "muted", "No files were skipped."));
    }
    skipped.forEach(function (entry) {
      var li = el("li", "file-item");
      li.appendChild(el("code", "file-path", entry.path));
      li.appendChild(el("span", "file-reason", " · " + (entry.reason || "")));
      skippedList.appendChild(li);
    });
  }

  function renderWarnings(warnings) {
    var card = document.getElementById("warnings-card");
    var list = document.getElementById("warnings-list");
    clear(list);
    if (!warnings || !warnings.length) {
      card.hidden = true;
      return;
    }
    warnings.forEach(function (text) {
      list.appendChild(el("li", null, text));
    });
    card.hidden = false;
  }

  function render(report) {
    currentReport = report;

    document.getElementById("total-score").textContent = report.score.total;
    document.getElementById("repo-name").textContent = report.repo.full_name;

    var link = document.getElementById("repo-link");
    link.href = report.repo.url;
    link.textContent = report.repo.url;

    document.getElementById("score-grade").textContent =
      "Grade " + report.score.grade + " · " + report.analysis_type +
      " analysis · " + report.repo.language;

    renderBanners(report.banners);
    renderBreakdown(report.score);
    renderClaims(report.claims, report.summary);
    renderCheckList("checks-list", report.quality_checks, false);
    renderCheckList("health-list", report.score.breakdown.health.items, true);
    renderFiles(report.files);
    renderWarnings(report.warnings);

    errorEl.hidden = true;
    resultEl.hidden = false;
  }

  // --- downloads ----------------------------------------------------------

  function downloadBlob(filename, text, mime) {
    var blob = new Blob([text], { type: mime || "text/plain;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  }

  function repoSlug() {
    if (!currentReport) return "report";
    return (currentReport.repo.full_name || "report").replace("/", "-");
  }

  document.getElementById("download-json").addEventListener("click", function () {
    if (!currentReport) return;
    downloadBlob("repojudge-" + repoSlug() + ".json",
      JSON.stringify(currentReport, null, 2), "application/json");
  });

  document.getElementById("download-md").addEventListener("click", function () {
    if (!currentReport) return;
    fetch("/report/" + currentReport.id + "/markdown")
      .then(function (response) {
        if (!response.ok) throw new Error("Markdown is not available.");
        return response.text();
      })
      .then(function (text) {
        downloadBlob("repojudge-" + repoSlug() + ".md", text, "text/markdown");
      })
      .catch(function (err) {
        showError("Download failed", err.message);
      });
  });

  // --- request flow -------------------------------------------------------

  function showError(title, message) {
    errorTitle.textContent = title;
    errorMessage.textContent = message;
    errorEl.hidden = false;
    resultEl.hidden = true;
  }

  function errorMessageFrom(payload, fallback) {
    if (payload && payload.error && payload.error.message) {
      return payload.error.message;
    }
    if (payload && payload.detail) {
      if (typeof payload.detail === "string") return payload.detail;
      if (Array.isArray(payload.detail) && payload.detail.length) {
        return payload.detail[0].msg || fallback;
      }
    }
    return fallback;
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();

    var repoUrl = urlInput.value.trim();
    if (!repoUrl) {
      showError("Missing URL", "Please enter a public GitHub repository URL.");
      return;
    }

    var requirements = reqInput.value
      .split("\n")
      .map(function (line) { return line.trim(); })
      .filter(function (line) { return line.length > 0; });

    var nocacheEl = document.getElementById("nocache");

    button.disabled = true;
    errorEl.hidden = true;
    resultEl.hidden = true;
    loadingEl.hidden = false;
    startSteps();

    fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        repo_url: repoUrl,
        requirements: requirements,
        nocache: nocacheEl ? nocacheEl.checked : false
      })
    })
      .then(function (response) {
        return response.json().then(function (payload) {
          return { ok: response.ok, status: response.status, payload: payload };
        });
      })
      .then(function (result) {
        finishSteps();
        if (!result.ok) {
          var code = result.payload && result.payload.error &&
            result.payload.error.code;
          var message = errorMessageFrom(result.payload,
            "The server returned an error (HTTP " + result.status + ").");
          if (code === "gemma_timeout") {
            message = message + " You can try again in a moment.";
          }
          showError(ERROR_TITLES[code] || "Could not analyze this repository",
            message);
          return;
        }
        render(result.payload);
      })
      .catch(function (err) {
        finishSteps();
        showError("Request failed",
          err && err.message ? err.message : "Could not reach the server.");
      })
      .finally(function () {
        button.disabled = false;
        loadingEl.hidden = true;
      });
  });
})();
