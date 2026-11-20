/* Beyond Words — Studio Session, Trajectory and Enterprise & ROI tabs. */
(() => {
  const q = selector => document.querySelector(selector);
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const SVG_NS = "http://www.w3.org/2000/svg";
  const svg = (tag, attrs = {}) => {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    return node;
  };

  const EMOJI = { neutral: "😐", happy: "😊", sad: "😢", angry: "😠", fearful: "😨", surprised: "😮", disgusted: "🤢" };
  const DISSONANCE = {
    masked_sarcasm: { title: "Masked sarcasm", tone: "bad" },
    masked_distress: { title: "Masked distress", tone: "bad" },
    mismatch: { title: "Voice / word mismatch", tone: "warn" },
    congruent: { title: "Congruent", tone: "ok" },
    words_only: { title: "Words only", tone: "muted" },
  };
  const ACTION = {
    human_handover: { title: "Human handover", tone: "bad" },
    de_escalate: { title: "De-escalate", tone: "warn" },
    continue: { title: "Continue", tone: "ok" },
  };
  const fmtValence = value => (value === null || value === undefined ? "—" : (value > 0 ? "+" : "") + value.toFixed(2));

  /* ---------------- Tabs ---------------- */
  const tabs = [...document.querySelectorAll(".tab")];
  function showTab(name) {
    tabs.forEach(tab => tab.setAttribute("aria-selected", String(tab.dataset.tab === name)));
    document.querySelectorAll(".panel").forEach(panel => { panel.hidden = panel.dataset.panel !== name; });
    if (name === "trajectory") drawTrajectory();
    if (name === "enterprise") renderWebhook();
    try { history.replaceState(null, "", "#" + name); } catch { /* ignore */ }
  }
  tabs.forEach(tab => tab.addEventListener("click", () => showTab(tab.dataset.tab)));

  /* ---------------- Insight card (shared) ---------------- */
  function valenceBar(label, sub, value) {
    const row = el("div", "vbar");
    const head = el("div", "vbar-head");
    head.append(el("span", "vbar-label", label), el("small", "vbar-sub", sub), el("strong", "vbar-value", fmtValence(value)));
    const track = el("div", "vbar-track");
    track.append(el("span", "vbar-zero"));
    if (value !== null && value !== undefined) {
      const fill = el("span", "vbar-fill " + (value >= 0 ? "pos" : "neg"));
      const width = Math.abs(value) * 50;
      fill.style.width = width + "%";
      fill.style.left = value >= 0 ? "50%" : (50 - width) + "%";
      track.append(fill);
    }
    const scale = el("div", "vbar-scale");
    scale.append(el("span", "", "−1 negative"), el("span", "", "0"), el("span", "", "positive +1"));
    row.append(head, track, scale);
    return row;
  }

  function gauge(value) {
    const wrap = el("div", "gauge");
    const box = svg("svg", { viewBox: "0 0 120 70", "aria-hidden": "true" });
    box.append(svg("path", { d: "M10 60 A50 50 0 0 1 110 60", class: "gauge-track" }));
    if (value !== null && value !== undefined) {
      const angle = Math.PI * (1 - value / 100);
      const x = 60 + 50 * Math.cos(angle);
      const y = 60 - 50 * Math.sin(angle);
      box.append(svg("path", { d: `M10 60 A50 50 0 0 1 ${x.toFixed(2)} ${y.toFixed(2)}`, class: "gauge-fill" }));
    }
    wrap.append(box);
    const number = el("div", "gauge-num");
    number.append(el("strong", "", value === null || value === undefined ? "—" : String(value)), el("small", "", "/100"));
    wrap.append(number, el("span", "gauge-cap", "Discrepancy score"));
    return wrap;
  }

  function renderInsightCard(container, insight, extras = {}) {
    container.replaceChildren();
    container.hidden = false;
    const kind = DISSONANCE[insight.dissonance] || DISSONANCE.congruent;
    const action = ACTION[insight.action] || ACTION.continue;

    const top = el("div", "insight-top");
    const left = el("div", "insight-verdict");
    left.append(el("span", "eyebrow", "Turn insight"));
    const badge = el("span", "badge " + kind.tone, kind.title);
    left.append(badge, el("p", "insight-text", insight.dissonance_text || ""));
    if (insight.intent) left.append(el("small", "hint", "Intent: " + insight.intent));
    top.append(left, gauge(insight.discrepancy));
    container.append(top);

    const voiceSub = extras.voiceSource === "scripted" ? "scripted voice profile" : "emotion2vec+";
    container.append(
      valenceBar("Voice valence", voiceSub, insight.vocal_valence),
      valenceBar("Word valence", "Gemini, words only", insight.semantic_valence),
    );

    const bottom = el("div", "insight-bottom");
    const distress = el("div", "distress");
    const distressHead = el("div", "vbar-head");
    distressHead.append(el("span", "vbar-label", "Distress"), el("strong", "vbar-value", `${insight.distress}/100`));
    const meter = el("div", "distress-track");
    const fill = el("span", "distress-fill");
    fill.style.width = insight.distress + "%";
    fill.dataset.level = insight.distress >= 70 ? "high" : insight.distress >= 40 ? "mid" : "low";
    meter.append(fill);
    distress.append(distressHead, meter);
    const next = el("div", "next-action");
    next.append(el("span", "eyebrow", "Recommended action"), el("span", "badge " + action.tone, action.title),
      el("small", "hint", insight.action_text || ""));
    bottom.append(distress, next);
    container.append(bottom);
  }
  window.BW = { renderInsightCard };

  /* ---------------- Shared session state ---------------- */
  const state = { turns: [], last: null };
  try { state.turns = JSON.parse(localStorage.getItem("bw-trajectory") || "[]").slice(-20); } catch { /* ignore */ }
  const save = () => { try { localStorage.setItem("bw-trajectory", JSON.stringify(state.turns)); } catch { /* ignore */ } };

  function recordTurn(data, source) {
    if (!data.insights) return;
    state.last = { ...data, source };
    state.turns.push({
      transcript: data.transcript, distress: data.insights.distress, action: data.insights.action,
      dissonance: data.insights.dissonance, emotion: data.vocal_cue?.label || null, source,
      reply: data.reply_text || null,
    });
    state.turns = state.turns.slice(-20);
    save();
    drawTrajectory();
    renderWebhook();
  }
  document.addEventListener("bw:turn", event => recordTurn(event.detail, "live"));

  async function analyze(transcript, profile, options = {}) {
    const response = await fetch("/v1/analyze-text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript, vocal_profile: profile, ...options }),
      signal: AbortSignal.timeout(90000),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || "Analysis failed.");
    return data;
  }

  /* ---------------- Studio Session ---------------- */
  const BENCHMARKS = [
    {
      id: "sarcasm", title: "Masked Sarcasm & Dissonance", icon: "🤢",
      text: "Great service, wonderful job. Really, just fantastic.",
      profile: { label: "disgusted", scores: { disgusted: 0.7, angry: 0.2, neutral: 0.1 } },
      blurb: "Glowing words, disgusted voice.",
    },
    {
      id: "masked-distress", title: "Masked Distress", icon: "😊",
      text: "No no, it's fine. I lost the whole order, but I'll just figure it out myself, I guess.",
      profile: { label: "happy", scores: { happy: 0.6, neutral: 0.3, surprised: 0.1 } },
      blurb: "A real problem, laughed off.",
    },
    {
      id: "escalation", title: "Congruent Escalation", icon: "😠",
      text: "This is the third time my order is late. I want to talk to a manager right now.",
      profile: { label: "angry", scores: { angry: 0.8, sad: 0.2 } },
      blurb: "Angry words, angry voice.",
    },
    {
      id: "calm", title: "Calm Inquiry", icon: "😐",
      text: "Hi, could you tell me when my package will arrive?",
      profile: { label: "neutral", scores: { neutral: 0.8, happy: 0.2 } },
      blurb: "Baseline: nothing hidden.",
    },
  ];

  const benchList = q("#bench-list");
  const studioRun = q("#studio-run");
  const studioStatus = q("#studio-status");
  let studioBusy = false;

  BENCHMARKS.forEach(bench => {
    const button = el("button", "bench");
    button.type = "button";
    button.dataset.bench = bench.id;
    button.append(el("span", "bench-icon", bench.icon));
    const text = el("span", "bench-text");
    text.append(el("strong", "", bench.title), el("small", "", bench.blurb));
    button.append(text);
    button.addEventListener("click", () => runStudio(bench.text, bench.profile, bench.title));
    benchList.append(button);
  });

  studioRun.addEventListener("click", () => {
    const text = q("#studio-text").value.trim();
    if (!text) { studioStatus.textContent = "Type an utterance first."; return; }
    const voice = q("#studio-voice").value;
    const profile = voice ? { label: voice, scores: voice === "neutral" ? { neutral: 1 } : { [voice]: 0.85, neutral: 0.15 } } : null;
    runStudio(text, profile, "Custom line");
  });

  function profileChips(profile) {
    const row = el("div", "profile");
    Object.entries(profile.scores).sort((a, b) => b[1] - a[1]).forEach(([name, value]) => {
      row.append(el("span", "pchip", `${EMOJI[name] || ""} ${name} ${Math.round(value * 100)}%`));
    });
    return row;
  }

  async function runStudio(text, profile, title) {
    if (studioBusy) return;
    studioBusy = true;
    document.body.classList.add("studio-busy");
    benchList.querySelectorAll("button").forEach(button => { button.disabled = true; });
    studioRun.disabled = true;
    studioStatus.textContent = `Analysing “${title}” with Gemini…`;
    try {
      const data = await analyze(text, profile);
      const out = q("#studio-output");
      q("#studio-empty").hidden = true;
      out.hidden = false;
      out.replaceChildren();

      const said = el("article", "bubble you");
      const head = el("header");
      head.append(el("span", "who", "Customer said"), el("small", "", title));
      said.append(head, el("p", "", `“${data.transcript}”`));
      if (profile) {
        const voiceRow = el("div", "cue-row");
        voiceRow.append(el("span", "cue-label", "Scripted voice"), profileChips(profile));
        said.append(voiceRow);
      }
      out.append(said);

      const card = el("div", "insight-card");
      renderInsightCard(card, data.insights, { voiceSource: data.vocal_source });
      out.append(card);

      const replies = el("div", "reply-compare");
      const plain = el("div", "rc plain");
      plain.append(el("span", "eyebrow", "Text-only bot"), el("p", "", data.text_only_reply || "—"),
        el("small", "hint", "Reads the words only — no voice, no analysis."));
      const aware = el("div", "rc aware");
      aware.append(el("span", "eyebrow", "Beyond Words"), el("p", "", data.reply_text || "—"),
        el("small", "hint", "Hears the voice and the detected dissonance."));
      replies.append(plain, aware);
      out.append(replies);

      q("#studio-meta").textContent = `${data.resolved_model || "Gemini"} · ${(data.elapsed_ms / 1000).toFixed(1)} s`;
      studioStatus.textContent = "Analysis ready. Gemini word analysis and replies are live; the voice profile is scripted.";
      recordTurn(data, "benchmark");
    } catch (error) {
      studioStatus.textContent = `${error.message} Please retry.`;
    } finally {
      studioBusy = false;
      document.body.classList.remove("studio-busy");
      benchList.querySelectorAll("button").forEach(button => { button.disabled = false; });
      studioRun.disabled = false;
    }
  }

  /* ---------------- Trajectory ---------------- */
  const ARC = [
    { text: "This is ridiculous. My order is a week late and nobody has told me anything!",
      profile: { label: "angry", scores: { angry: 0.8, disgusted: 0.15, sad: 0.05 } } },
    { text: "Okay. So it ships tomorrow? I still really need it before Friday.",
      profile: { label: "sad", scores: { sad: 0.45, neutral: 0.45, angry: 0.1 } } },
    { text: "Alright, that works for me. Thanks for sorting it out.",
      profile: { label: "neutral", scores: { neutral: 0.55, happy: 0.45 } } },
  ];

  function trajectoryStats(turns) {
    if (!turns.length) return null;
    const first = turns[0].distress;
    const last = turns[turns.length - 1].distress;
    const adv = turns.length > 1 ? (last - first) / (turns.length - 1) : 0;
    return { first, last, adv, peak: Math.max(...turns.map(t => t.distress)), count: turns.length };
  }

  function drawTrajectory() {
    const chart = q("#traj-chart");
    const turns = state.turns;
    chart.replaceChildren();
    const W = 800, H = 300, L = 48, R = 24, T = 20, B = 40;
    const y = value => T + (1 - value / 100) * (H - T - B);
    const x = index => turns.length <= 1 ? (L + (W - R)) / 2 : L + index * ((W - L - R) / (turns.length - 1));

    [[70, 100, "zone-high", "Handover ≥70"], [40, 70, "zone-mid", "De-escalate 40–69"], [0, 40, "zone-low", "Calm <40"]]
      .forEach(([low, high, cls, label]) => {
        chart.append(svg("rect", { x: L, y: y(high), width: W - L - R, height: y(low) - y(high), class: cls }));
        const text = svg("text", { x: W - R - 6, y: y(high) + 14, class: "zone-label", "text-anchor": "end" });
        text.textContent = label;
        chart.append(text);
      });
    [0, 20, 40, 60, 80, 100].forEach(value => {
      chart.append(svg("line", { x1: L, x2: W - R, y1: y(value), y2: y(value), class: "grid" }));
      const label = svg("text", { x: L - 10, y: y(value) + 4, class: "axis", "text-anchor": "end" });
      label.textContent = value;
      chart.append(label);
    });

    const kpis = q("#traj-kpis");
    kpis.replaceChildren();
    const stats = trajectoryStats(turns);
    const kpi = (label, value, tone = "") => {
      const box = el("div", "kpi " + tone);
      box.append(el("span", "eyebrow", label), el("strong", "", value));
      kpis.append(box);
    };
    if (!stats) {
      kpi("Turns", "0"); kpi("Start", "—"); kpi("Current", "—"); kpi("ADV", "—"); kpi("Peak", "—");
      const empty = svg("text", { x: W / 2, y: H / 2, class: "axis", "text-anchor": "middle" });
      empty.textContent = "No turns yet";
      chart.append(empty);
      q("#traj-list").replaceChildren();
      return;
    }
    kpi("Turns", String(stats.count));
    kpi("Start distress", String(stats.first));
    kpi("Current distress", String(stats.last));
    kpi("ADV", `${stats.adv > 0 ? "+" : ""}${stats.adv.toFixed(0)} pts/turn`, stats.adv < 0 ? "good" : stats.adv > 0 ? "bad" : "");
    kpi("Peak", String(stats.peak));

    const points = turns.map((turn, index) => `${x(index)},${y(turn.distress)}`).join(" ");
    if (turns.length > 1) {
      const area = svg("polygon", { points: `${x(0)},${y(0)} ${points} ${x(turns.length - 1)},${y(0)}`, class: "area" });
      chart.append(area, svg("polyline", { points, class: "line" }));
    }
    turns.forEach((turn, index) => {
      const cx = x(index), cy = y(turn.distress);
      const level = turn.distress >= 70 ? "high" : turn.distress >= 40 ? "mid" : "low";
      chart.append(svg("circle", { cx, cy, r: 7, class: "pt " + level }));
      const value = svg("text", { x: cx, y: cy - 14, class: "pt-label", "text-anchor": "middle" });
      value.textContent = turn.distress;
      const tick = svg("text", { x: cx, y: H - 14, class: "axis", "text-anchor": "middle" });
      tick.textContent = `T${index + 1}`;
      chart.append(value, tick);
    });

    const list = q("#traj-list");
    list.replaceChildren();
    turns.forEach(turn => {
      const item = el("li");
      const head = el("div", "tl-head");
      head.append(el("span", "badge " + (turn.distress >= 70 ? "bad" : turn.distress >= 40 ? "warn" : "ok"), `Distress ${turn.distress}`),
        el("span", "pchip", `${EMOJI[turn.emotion] || "🔇"} ${turn.emotion || "no voice"}`),
        el("small", "hint", turn.source === "live" ? "Live recording" : turn.source === "simulated" ? "Simulation (scripted voice)" : "Benchmark (scripted voice)"));
      item.append(head, el("p", "", `“${turn.transcript}”`));
      if (turn.reply) item.append(el("p", "tl-reply", `↳ ${turn.reply}`));
      list.append(item);
    });
  }

  q("#traj-clear").addEventListener("click", () => {
    state.turns = [];
    save();
    drawTrajectory();
    q("#traj-status").textContent = "Cleared.";
  });

  q("#traj-sim").addEventListener("click", async () => {
    const button = q("#traj-sim");
    if (button.disabled) return;
    button.disabled = true;
    document.body.classList.add("traj-busy");
    state.turns = [];
    drawTrajectory();
    try {
      for (let i = 0; i < ARC.length; i++) {
        q("#traj-status").textContent = `Turn ${i + 1} of ${ARC.length}: analysing with Gemini…`;
        const data = await analyze(ARC[i].text, ARC[i].profile, { compare_text_only: false });
        recordTurn(data, "simulated");
      }
      const stats = trajectoryStats(state.turns);
      q("#traj-status").textContent = `Simulation complete: distress ${state.turns.map(t => t.distress).join(" → ")}, ADV ${stats.adv.toFixed(0)} pts/turn. Voices are scripted; Gemini analysis is live.`;
    } catch (error) {
      q("#traj-status").textContent = `${error.message} Please retry the simulation.`;
    } finally {
      button.disabled = false;
      document.body.classList.remove("traj-busy");
    }
  });

  /* ---------------- Enterprise: webhook dispatcher ---------------- */
  const SAMPLE = {
    transcript: "Great service, wonderful job. Really, just fantastic.",
    vocal_cue: { label: "disgusted" },
    insights: { distress: 50, dissonance: "masked_sarcasm", discrepancy: 80, vocal_valence: -0.65, semantic_valence: 0.95,
      action: "de_escalate", action_text: "Acknowledge the frustration, slow down, and offer a concrete next step.", intent: "complain about service" },
    reply_text: "It sounds like something didn't go the way it should have. Can you tell me what happened so I can help fix it?",
    source: "sample",
  };

  function payloadFor(target, turn) {
    const i = turn.insights;
    const priority = i.action === "human_handover" ? "urgent" : i.action === "de_escalate" ? "high" : "normal";
    const tags = ["beyondwords", i.dissonance, i.action, turn.vocal_cue?.label].filter(Boolean);
    const summary = `Distress ${i.distress}/100 · ${(DISSONANCE[i.dissonance] || {}).title || i.dissonance}` +
      (i.discrepancy != null ? ` · discrepancy ${i.discrepancy}/100` : "");
    if (target === "zendesk") {
      return { ticket: {
        subject: `[Beyond Words] ${(ACTION[i.action] || {}).title || i.action}: ${i.intent || "voice interaction"}`,
        comment: { body: `Customer said: "${turn.transcript}"\n\n${summary}\nRecommended: ${i.action_text}\n\nSuggested reply: ${turn.reply_text || "—"}`, public: false },
        priority, tags,
        custom_fields: [
          { id: "beyondwords_distress", value: i.distress },
          { id: "beyondwords_dissonance", value: i.dissonance },
          { id: "beyondwords_discrepancy", value: i.discrepancy },
          { id: "beyondwords_vocal_emotion", value: turn.vocal_cue?.label || null },
        ],
      } };
    }
    if (target === "salesforce") {
      return {
        Subject: `Beyond Words: ${i.intent || "voice interaction"}`,
        Description: `Customer said: "${turn.transcript}"\n${summary}\nSuggested reply: ${turn.reply_text || "—"}`,
        Origin: "Voice", Status: "New",
        Priority: priority === "urgent" ? "High" : priority === "high" ? "Medium" : "Low",
        BeyondWords_Distress__c: i.distress,
        BeyondWords_Dissonance__c: i.dissonance,
        BeyondWords_Discrepancy__c: i.discrepancy,
        BeyondWords_Vocal_Emotion__c: turn.vocal_cue?.label || null,
        BeyondWords_Recommended_Action__c: i.action,
      };
    }
    return {
      event: "beyondwords.turn.analyzed", created_at: new Date().toISOString(),
      transcript: turn.transcript, vocal_emotion: turn.vocal_cue?.label || null, insights: i,
      suggested_reply: turn.reply_text || null,
    };
  }

  function renderWebhook() {
    const turn = state.last?.insights ? state.last : SAMPLE;
    q("#hook-source").textContent = turn === SAMPLE
      ? "Showing a sample turn. Run a Companion turn or a benchmark to use real data."
      : `Built from your latest ${turn.source === "live" ? "live Companion turn" : turn.source === "simulated" ? "simulation turn" : "benchmark"}.`;
    q("#hook-payload").textContent = JSON.stringify(payloadFor(q("#hook-target").value, turn), null, 2);
  }
  q("#hook-target").addEventListener("change", renderWebhook);
  q("#hook-copy").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(q("#hook-payload").textContent);
      q("#hook-copy").textContent = "Copied ✓";
      setTimeout(() => { q("#hook-copy").textContent = "Copy payload"; }, 1500);
    } catch { q("#hook-copy").textContent = "Copy failed"; }
  });

  /* ---------------- Enterprise: ROI calculator ---------------- */
  const SLIDERS = [
    { id: "agents", label: "Agents", min: 10, max: 1000, step: 10, value: 100, fmt: v => v },
    { id: "calls", label: "Calls per agent per day", min: 10, max: 120, step: 5, value: 40, fmt: v => v },
    { id: "days", label: "Working days per year", min: 200, max: 365, step: 5, value: 250, fmt: v => v },
    { id: "esc", label: "Calls that escalate", min: 1, max: 30, step: 1, value: 10, fmt: v => v + "%" },
    { id: "avoid", label: "Escalations avoided with Beyond Words", min: 5, max: 60, step: 1, value: 25, fmt: v => v + "%" },
    { id: "cost", label: "Cost per escalation", min: 5, max: 100, step: 1, value: 25, fmt: v => "$" + v },
    { id: "price", label: "Platform cost per agent / month", min: 5, max: 150, step: 5, value: 30, fmt: v => "$" + v },
  ];
  const money = value => new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(value);
  const number = value => new Intl.NumberFormat("en-US").format(Math.round(value));

  const sliderBox = q("#roi-sliders");
  SLIDERS.forEach(spec => {
    const row = el("label", "slider");
    const head = el("span", "slider-head");
    const out = el("strong", "", spec.fmt(spec.value));
    head.append(el("span", "", spec.label), out);
    const input = el("input");
    Object.assign(input, { type: "range", min: spec.min, max: spec.max, step: spec.step, value: spec.value, id: "roi-" + spec.id });
    input.addEventListener("input", () => { out.textContent = spec.fmt(Number(input.value)); renderRoi(); });
    row.append(head, input);
    sliderBox.append(row);
  });

  function renderRoi() {
    const v = Object.fromEntries(SLIDERS.map(spec => [spec.id, Number(q("#roi-" + spec.id).value)]));
    const calls = v.agents * v.calls * v.days;
    const escalations = calls * v.esc / 100;
    const avoided = escalations * v.avoid / 100;
    const gross = avoided * v.cost;
    const platform = v.agents * v.price * 12;
    const net = gross - platform;
    const payback = gross > 0 ? platform / (gross / 365) : Infinity;
    const roi = platform > 0 ? (net / platform) * 100 : 0;
    const out = q("#roi-out");
    out.replaceChildren();
    const stat = (label, value, tone = "") => {
      const box = el("div", "kpi " + tone);
      box.append(el("span", "eyebrow", label), el("strong", "", value));
      out.append(box);
    };
    stat("Calls / year", number(calls));
    stat("Escalations avoided", number(avoided));
    stat("Gross savings", money(gross));
    stat("Platform cost", money(platform));
    stat("Net annual savings", money(net), net >= 0 ? "good" : "bad");
    stat("Payback period", Number.isFinite(payback) ? `${Math.ceil(payback)} days` : "—", payback <= 90 ? "good" : "");
    stat("ROI", `${roi.toFixed(0)}%`, roi >= 0 ? "good" : "bad");
  }

  drawTrajectory();
  renderWebhook();
  renderRoi();
  const initial = location.hash.slice(1);
  if (tabs.some(tab => tab.dataset.tab === initial)) showTab(initial);
})();
