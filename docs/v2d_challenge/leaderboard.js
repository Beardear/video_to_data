/* Render V2D Challenge standings and Kaggle metric links. */
(function () {
  "use strict";

  // Generated from config/tracks.json by tools/build_site_leaderboard.py. Do not edit by hand.
  //
  // The page must never tell a visitor the leaderboard is unavailable. If the data host is
  // missing, unreachable, or serving something unparseable, the section falls back to this: the
  // real tabs, the real columns, and no rows. That is the same thing the page shows before the
  // first submission scores, which is the honest state and the one people already expect.
  /* FALLBACK BEGIN */
  var FALLBACK = {"note":"Fallback board: the data host was unreachable.","tracks":[{"key":"track_1","title":"Track 1 - Reconstruction","short_title":"Track 1","metrics":[{"key":"cd_h_cm","display":"CD-H","unit":"cm","higher_is_better":false,"competition":"v2d-challenge-track1-cd-h","available":true},{"key":"cd_o_cm","display":"CD-O","unit":"cm","higher_is_better":false,"competition":"v2d-challenge-track1-cd-o","available":true},{"key":"acc_h_cm","display":"ACC-H","unit":"cm","higher_is_better":false,"competition":"v2d-challenge-track1-acc-h","available":true},{"key":"acc_o_cm","display":"ACC-O","unit":"cm","higher_is_better":false,"competition":"v2d-challenge-track1-acc-o","available":true},{"key":"interpenetration_cm","display":"PEN","unit":"cm","higher_is_better":false,"competition":"v2d-challenge-track1-pen","available":true}],"rows":[],"incomplete":[]},{"key":"track_2_tier1","title":"Track 2 - Robotic Grounding (Tier 1: multi-view input)","short_title":"Track 2 \u00b7 Tier 1","metrics":[{"key":"add_auc","display":"AUC","unit":"","higher_is_better":true,"competition":"v2d-challenge-track2-tier1-auc","available":true},{"key":"spider_sr","display":"SP-SR","unit":"","higher_is_better":true,"competition":"v2d-challenge-track2-tier1-sp-sr","available":true},{"key":"maniptrans_sr","display":"MP-SR","unit":"","higher_is_better":true,"competition":"v2d-challenge-track2-tier1-mp-sr","available":true},{"key":"mppe_cm","display":"MPPE","unit":"cm","higher_is_better":false,"competition":"v2d-challenge-track2-tier1-mppe","available":true}],"rows":[],"incomplete":[]},{"key":"track_2_tier2","title":"Track 2 - Robotic Grounding (Tier 2: synthetically corrupted input)","short_title":"Track 2 \u00b7 Tier 2","metrics":[{"key":"add_auc","display":"AUC","unit":"","higher_is_better":true,"competition":"v2d-challenge-track2-tier2-auc","available":true},{"key":"spider_sr","display":"SP-SR","unit":"","higher_is_better":true,"competition":"v2d-challenge-track2-tier2-sp-sr","available":true},{"key":"maniptrans_sr","display":"MP-SR","unit":"","higher_is_better":true,"competition":"v2d-challenge-track2-tier2-mp-sr","available":true},{"key":"mppe_cm","display":"MPPE","unit":"cm","higher_is_better":false,"competition":"v2d-challenge-track2-tier2-mppe","available":true}],"rows":[],"incomplete":[]},{"key":"track_2_tier3","title":"Track 2 - Robotic Grounding (Tier 3: bring your own reconstruction)","short_title":"Track 2 \u00b7 Tier 3","metrics":[{"key":"add_auc","display":"AUC","unit":"","higher_is_better":true,"competition":"v2d-challenge-track2-tier3-auc","available":true},{"key":"spider_sr","display":"SP-SR","unit":"","higher_is_better":true,"competition":"v2d-challenge-track2-tier3-sp-sr","available":true},{"key":"maniptrans_sr","display":"MP-SR","unit":"","higher_is_better":true,"competition":"v2d-challenge-track2-tier3-mp-sr","available":true},{"key":"mppe_cm","display":"MPPE","unit":"cm","higher_is_better":false,"competition":"v2d-challenge-track2-tier3-mppe","available":true}],"rows":[],"incomplete":[]},{"key":"track_3","title":"Track 3 - Egocentric","short_title":"Track 3","metrics":[{"key":"add_auc","display":"AUC","unit":"","higher_is_better":true,"competition":"v2d-challenge-track3-auc","available":true},{"key":"spider_sr","display":"SP-SR","unit":"","higher_is_better":true,"competition":"v2d-challenge-track3-sp-sr","available":true},{"key":"maniptrans_sr","display":"MP-SR","unit":"","higher_is_better":true,"competition":"v2d-challenge-track3-mp-sr","available":true},{"key":"rpe_cm","display":"RPE","unit":"cm","higher_is_better":false,"competition":"v2d-challenge-track3-rpe","available":true},{"key":"mppe_cm","display":"MPPE","unit":"cm","higher_is_better":false,"competition":"v2d-challenge-track3-mppe","available":true},{"key":"cd_o_cm","display":"CD-O","unit":"cm","higher_is_better":false,"competition":"v2d-challenge-track3-cd-o","available":true}],"rows":[],"incomplete":[]}]};
  /* FALLBACK END */

  var MOUNT_ID = "v2d-leaderboard";
  var DEFAULT_SRC = "./leaderboard.json";
  var REFRESH_MS = 5 * 60 * 1000;

  var css = {
    mono: "var(--font-mono, ui-monospace, SFMono-Regular, Menlo, monospace)",
    border: "var(--border-color-base, #e4e4e4)",
    ink: "var(--text-color-primary, #1a1a1a)",
    muted: "var(--text-color-secondary, #6b6b6b)",
    brand: "var(--color-brand, #76b900)"
  };

  function el(tag, style, text) {
    var node = document.createElement(tag);
    if (style) node.setAttribute("style", style);
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function num(value) {
    if (value === null || value === undefined || value === "") return null;
    var parsed = typeof value === "number" ? value : Number(value);
    return isFinite(parsed) ? parsed : null;
  }

  function fmtScore(value) {
    var parsed = num(value);
    return parsed === null ? "-" : parsed.toFixed(4);
  }

  function fmtDate(value) {
    if (!value) return "";
    var parsed = new Date(String(value).replace(" ", "T") + (/[Zz+]/.test(value) ? "" : "Z"));
    if (isNaN(parsed.getTime())) return String(value);
    return parsed.toISOString().slice(0, 10);
  }

  /* Blanks always sort last, in both directions: a team that has not entered a leaderboard
     should never outrank one that has, and should never be mistaken for the worst entry. */
  function compare(a, b, key, direction) {
    var x = num(a.scores ? a.scores[key] : null);
    var y = num(b.scores ? b.scores[key] : null);
    var xMissing = x === null;
    var yMissing = y === null;
    if (xMissing && yMissing) return a.team.toLowerCase() < b.team.toLowerCase() ? -1 : 1;
    if (xMissing) return 1;
    if (yMissing) return -1;
    if (x === y) return a.team.toLowerCase() < b.team.toLowerCase() ? -1 : 1;
    return direction === "asc" ? x - y : y - x;
  }

  function canonicalRanks(track, metric) {
    var ranks = new Map();
    if (!metric) return ranks;
    track.rows.filter(function (row) {
      return row.is_baseline !== true && num(row.scores[metric.key]) !== null;
    }).sort(function (a, b) {
      return compare(a, b, metric.key, metric.higher_is_better === false ? "asc" : "desc");
    }).forEach(function (row, index) {
      ranks.set(row, index + 1);
    });
    return ranks;
  }

  function isCari4dBaseline(track, row) {
    return row.is_baseline === true && (
      (track.key === "track_1" && /^CARI4D\b/i.test(row.team)) ||
      (track.key === "track_2_tier3" && /\bCARI4D Reconstruction\b/i.test(row.team))
    );
  }

  function renderTable(track, state) {
    var available = track.metrics.filter(function (m) { return m.available; });
    var wrap = el("div", "overflow-x:auto;-webkit-overflow-scrolling:touch");
    var table = el(
      "table",
      "width:100%;border-collapse:collapse;font-size:15px;min-width:" +
        (320 + available.length * 130) + "px"
    );

    var thead = el("thead");
    var headRow = el("tr");
    var headStyle =
      "text-align:right;padding:12px 14px;border-bottom:1px solid " + css.border +
      ";font-family:" + css.mono + ";font-size:11px;letter-spacing:.08em;text-transform:uppercase;" +
      "color:" + css.muted + ";font-weight:400;white-space:nowrap";

    headRow.appendChild(el("th", headStyle + ";text-align:left;width:52px", "#"));
    headRow.appendChild(el("th", headStyle + ";text-align:left", "Team"));

    available.forEach(function (metric) {
      var th = el("th", headStyle + ";cursor:pointer;user-select:none");
      th.setAttribute("scope", "col");
      th.setAttribute("tabindex", "0");
      th.setAttribute("role", "columnheader");
      var active = state.sortKey === metric.key;
      th.setAttribute("aria-sort", active ? (state.sortDir === "asc" ? "ascending" : "descending") : "none");
      th.title = "Sort by " + metric.display + " (" +
        (metric.higher_is_better === false ? "lower is better" : "higher is better") +
        ") · Kaggle competition: " + metric.competition;

      var label = el("span", null, metric.display + (metric.unit ? " (" + metric.unit + ")" : ""));
      if (active) label.setAttribute("style", "color:" + css.ink);
      th.appendChild(label);
      var caret = el("span", "margin-left:6px;color:" + (active ? css.brand : "transparent"),
        state.sortDir === "asc" && active ? "▲" : "▼");
      th.appendChild(caret);

      function activate() {
        if (state.sortKey === metric.key) {
          state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
        } else {
          state.sortKey = metric.key;
          // First click on a column shows its best entries first. For a lower-is-better
          // metric such as RPE (cm) that is ascending, not descending.
          state.sortDir = metric.higher_is_better === false ? "asc" : "desc";
        }
        state.rerender();
      }
      th.addEventListener("click", activate);
      th.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activate(); }
      });
      headRow.appendChild(th);
    });

    headRow.appendChild(el("th", headStyle, "Last entry"));
    thead.appendChild(headRow);
    table.appendChild(thead);

    var rows = track.rows.slice().sort(function (a, b) {
      return compare(a, b, state.sortKey, state.sortDir);
    });

    var selectedMetric = available.filter(function (metric) { return metric.key === state.sortKey; })[0];
    var ranks = canonicalRanks(track, selectedMetric);
    var tbody = el("tbody");
    rows.forEach(function (row) {
      var baseline = row.is_baseline === true;
      var tr = el("tr", "border-bottom:1px solid " + css.border);
      var cell = "padding:14px;text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums";

      tr.appendChild(el("td", cell + ";text-align:left;color:" + css.muted + ";font-family:" + css.mono, ranks.has(row) ? ranks.get(row) : "—"));

      var teamCell = el("td", cell + ";text-align:left;white-space:normal");
      var teamName = track.key === "track_1" && isCari4dBaseline(track, row)
        ? "CARI4D (commercial) Baseline" : row.team;
      var name = el("div", "font-weight:" + (baseline ? "400" : "500") +
        ";color:" + (baseline ? css.muted : css.ink), teamName);
      if (isCari4dBaseline(track, row)) {
        var marker = el("a", "color:inherit;text-decoration:none", "*");
        marker.href = "#v2d-cari4d-baseline-note";
        marker.setAttribute("aria-label", "About the CARI4D baseline");
        var superscript = el("sup", "margin-left:2px");
        superscript.appendChild(marker);
        name.appendChild(superscript);
      }
      if (baseline) {
        name.appendChild(el("span",
          "display:inline-block;margin-left:8px;padding:2px 7px;border:1px solid " + css.border +
          ";border-radius:4px;font-size:11px;font-weight:400;color:" + css.muted,
          "Baseline"));
      }
      teamCell.appendChild(name);
      if (row.members && row.members.length) {
        teamCell.appendChild(
          el("div", "font-size:12px;color:" + css.muted + ";font-family:" + css.mono, row.members.join(", "))
        );
      }
      if (row.missing && row.missing.length) {
        var note = row.missing
          .map(function (key) {
            var found = track.metrics.filter(function (m) { return m.key === key; })[0];
            return found ? found.display : key;
          })
          .join(", ");
        teamCell.appendChild(
          el("div", "font-size:12px;color:" + css.muted, "no entry for " + note)
        );
      }
      tr.appendChild(teamCell);

      available.forEach(function (metric) {
        var value = num(row.scores ? row.scores[metric.key] : null);
        var strong = !baseline && state.sortKey === metric.key;
        var td = el(
          "td",
          cell + ";color:" + (baseline || value === null ? css.muted : css.ink) +
            (strong ? ";font-weight:600" : ""),
          fmtScore(value)
        );
        tr.appendChild(td);
      });

      tr.appendChild(el("td", cell + ";color:" + css.muted + ";font-size:13px", fmtDate(row.last_submission)));
      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  function renderEmpty(message, detail) {
    var card = el(
      "div",
      "border:1px solid " + css.border +
        ";padding:88px 40px;display:flex;flex-direction:column;align-items:center;gap:14px;text-align:center"
    );
    card.appendChild(el("div", "font-size:24px;font-weight:500", message));
    if (detail) card.appendChild(el("div", "font-size:16px;color:" + css.muted + ";max-width:52ch", detail));
    return card;
  }

  function render(mount, data, state) {
    mount.innerHTML = "";
    if (!data || !data.tracks || !data.tracks.length) {
      mount.appendChild(renderEmpty("No entries yet",
        "Standings appear here once submissions are scored. Track 2 is scored separately at "
        + "each of its three input tiers."));
      return;
    }

    var tracks = data.tracks.filter(function (t) { return t && typeof t === "object"; });
    tracks.forEach(function (t) {
      if (!Array.isArray(t.metrics)) t.metrics = [];
      if (!Array.isArray(t.rows)) t.rows = [];
      if (!t.title) t.title = t.key || "Track";
      t.rows.forEach(function (r) {
        if (!r.scores || typeof r.scores !== "object") r.scores = {};
        if (typeof r.team !== "string") r.team = String(r.team === undefined ? "" : r.team);
        if (!Array.isArray(r.members)) r.members = [];
      });
    });
    if (!tracks.length) {
      mount.appendChild(renderEmpty("No entries yet",
        "Standings appear here once submissions are scored. Track 2 is scored separately at "
        + "each of its three input tiers."));
      return;
    }
    if (state.track >= tracks.length) state.track = 0;

    var tabs = el("div", "display:flex;gap:24px;border-bottom:1px solid " + css.border + ";margin-bottom:8px;flex-wrap:wrap");
    tabs.setAttribute("role", "tablist");
    tracks.forEach(function (track, index) {
      var active = index === state.track;
      var tab = el(
        "button",
        "appearance:none;background:none;border:0;border-bottom:2px solid " +
          (active ? css.brand : "transparent") + ";padding:10px 2px;cursor:pointer;font:inherit;font-size:16px;color:" +
          (active ? css.ink : css.muted),
        track.short_title || track.title
      );
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-selected", active ? "true" : "false");
      tab.addEventListener("click", function () {
        state.track = index;
        var metrics = track.metrics.filter(function (m) { return m.available; });
        state.sortKey = metrics.length ? metrics[0].key : null;
        state.sortDir = metrics.length && metrics[0].higher_is_better === false ? "asc" : "desc";
        state.rerender();
      });
      tabs.appendChild(tab);
    });
    mount.appendChild(tabs);

    var track = tracks[state.track];
    if (track.short_title && track.short_title !== track.title) {
      mount.appendChild(el("div",
        "font-size:15px;color:" + css.muted + ";margin:14px 0 2px", track.title));
    }
    if (!state.sortKey) {
      var first = track.metrics.filter(function (m) { return m.available; })[0];
      state.sortKey = first ? first.key : null;
      if (first && first.higher_is_better === false) state.sortDir = "asc";
    }

    if (!track.rows.length) {
      mount.appendChild(renderEmpty("No entries yet for " + track.title,
        "Standings can take up to 24 hours to appear here. Your score is on the Kaggle "
        + "competition page as soon as it is computed."));
    } else {
      mount.appendChild(renderTable(track, state));
    }

    var footer = el("div", "display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-top:16px;font-size:13px;color:" + css.muted);
    footer.appendChild(
      el("div", null, data.generated_at ? "Updated " + new Date(data.generated_at).toUTCString() : "")
    );
    var links = el("div", "display:flex;gap:14px;flex-wrap:wrap");
    links.appendChild(el("span", null, "Links to live track leaderboards on Kaggle"));
    track.metrics.forEach(function (metric) {
      // Only link out when the slug really is a slug. The prefix already makes a
      // "javascript:" value inert, but refusing to build the URL at all means a typo in
      // config/tracks.json shows as a plain label rather than a link to nowhere.
      if (!/^[A-Za-z0-9._-]{1,120}$/.test(String(metric.competition || ""))) {
        links.appendChild(el("span", "color:" + css.muted, metric.display));
        return;
      }
      var link = el("a", "color:" + css.muted + ";text-decoration:underline", metric.display);
      link.href = "https://www.kaggle.com/competitions/" + encodeURIComponent(metric.competition) + "/leaderboard";
      link.target = "_blank";
      link.rel = "noopener";
      link.title = "Kaggle leaderboard for " + metric.display;
      links.appendChild(link);
    });
    footer.appendChild(links);
    mount.appendChild(footer);
    if (track.rows.some(function (row) { return isCari4dBaseline(track, row); })) {
      var note = el("p", "margin:12px 0 0;font-size:13px;line-height:1.6;color:" + css.muted,
        "* We use a commercial friendly recreation of CARI4D with retrained weights and some non-commercial modules replaced. Performance may differ slightly from the original implementation.");
      note.id = "v2d-cari4d-baseline-note";
      mount.appendChild(note);
    }

  }

  function boot() {
    var mount = document.getElementById(MOUNT_ID);
    if (!mount) return;
    var src = mount.getAttribute("data-src") || DEFAULT_SRC;
    var state = { track: 0, sortKey: null, sortDir: "desc", data: null };
    // The challenge page is driven by a template runtime (every node carries data-dc-tpl). It
    // re-renders the document from its template AFTER this script runs, which silently discarded
    // everything appended here: the mount came back empty and the section showed only its intro
    // text. Guard by writing through a flag and watching for the mount being emptied under us.
    var writing = false;

    function paint() {
      writing = true;
      try {
        render(mount, state.data, state);
      } finally {
        // Let the mutation records this write produced drain before re-arming.
        setTimeout(function () { writing = false; }, 0);
      }
    }

    state.rerender = paint;

    if (typeof MutationObserver === "function") {
      new MutationObserver(function () {
        if (writing) return;                        // our own write, not the framework's
        if (mount.childElementCount > 0) return;    // still populated, nothing to do
        if (!state.data) return;                    // nothing to restore yet
        paint();
      }).observe(mount, { childList: true });
    }

    writing = true;
    mount.appendChild(el("div", "padding:48px;text-align:center;color:" + css.muted, "Loading leaderboard…"));
    setTimeout(function () { writing = false; }, 0);

    function load() {
      // GitHub Pages serves cache-control: max-age=600, so bust it or a 15-minute refresh is
      // invisible for up to 10 of those minutes.
      fetch(src + (src.indexOf("?") === -1 ? "?" : "&") + "v=" + Date.now(), { cache: "no-store" })
        .then(function (response) {
          if (!response.ok) throw new Error("HTTP " + response.status);
          return response.json();
        })
        .then(function (data) {
          if (!data || !Array.isArray(data.tracks) || !data.tracks.some(function (track) {
            return Array.isArray(track.metrics) && track.metrics.some(function (metric) {
              return metric.available;
            });
          })) throw new Error("feed has no available metrics");
          // render() replaces the whole subtree, which drops focus and scroll position. The
          // board changes rarely, so only pay that cost when the payload actually differs.
          var next = JSON.stringify(data);
          if (state.raw === next) return;
          state.raw = next;
          state.data = data;
          state.rerender();
        })
        .catch(function (error) {
          if (state.data) return; // a failed refresh must not blank a table that is already up
          // Never surface the failure as page copy. A visitor cannot act on it, and "Leaderboard
          // unavailable" reads as broken where "no entries yet" reads as early. The console keeps
          // the real reason for whoever is debugging.
          if (window.console && console.warn) {
            console.warn("v2d leaderboard: falling back to the empty board (" + error.message + ")");
          }
          if (FALLBACK) {
            state.data = FALLBACK;
            paint();
          }
        });
    }

    load();
    setInterval(load, REFRESH_MS);
  }

  // Boot after load rather than DOMContentLoaded: the template runtime rebuilds the document
  // between the two, and booting early means painting into a subtree about to be thrown away.
  if (document.readyState === "complete") {
    boot();
  } else {
    window.addEventListener("load", function () { setTimeout(boot, 0); });
  }
})();
