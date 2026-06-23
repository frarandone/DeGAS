import { useEffect, useMemo, useRef, useState } from "react";
import { fetchLossFunctions, fetchOptimizers, fetchHealth, type ServerHealth } from "./api";
import { buildExportBundle, downloadBundle } from "./export";
import { SogaEditor } from "./components/SogaEditor";
import { LossDSLEditor } from "./components/LossDSLEditor";
import { ResultsPanel } from "./components/ResultsPanel";
import { DistributionPanel } from "./components/DistributionPanel";
import { ErrorToast } from "./components/ErrorToast";
import { ThemeToggle } from "./components/ThemeToggle";
import { Toolbar } from "./components/Toolbar";
import { EXAMPLES, type Example } from "./examples";
import { useOptimization } from "./hooks/useOptimization";
import { useTheme } from "./hooks/useTheme";
import type { LossFunctionInfo, LossMode, LossParamInfo, OptimizationRequest, RunErrorKind } from "./types";

function formatRunError(kind: RunErrorKind | null, detail: string): string {
  switch (kind) {
    case "connection_lost":
      return detail; // already a friendly sentence
    case "compute_error":
      return `Optimization failed: ${detail}`;
    case "rate_limited":
      return `Rate limited: ${detail}`;
    case "at_capacity":
      return `Server busy: ${detail}`;
    default:
      return detail; // setup_error / unknown
  }
}

const DEFAULT_LOSS_SOURCE = `// Custom loss - parameter names cannot be grammar keywords.
// Use short aliases: d: dist, ts: traj_set, idx: index_list, etc.

loss signal_error(d: dist, target: scalar) =
    sum( (d.mean[range(0, 5)] - ones(range(0, 5)) * target) ^ 2 )
`;

const REPO_URL = "https://github.com/frarandone/DeGAS";

const PANEL: React.CSSProperties = {
  border: "1px solid var(--border)",
  borderRadius: 12,
  overflow: "hidden",
  display: "flex",
  flexDirection: "column",
};


// Takes the mousedown event and the ref directly (not a ref-capturing factory
// called during render) so callers pass the ref from inside the event handler,
// where reading ref.current is allowed (react-hooks/refs).
function startDrag(
  e: React.MouseEvent<HTMLDivElement>,
  ref: React.RefObject<HTMLDivElement | null>,
  axis: "x" | "y",
  setter: (f: number) => void,
  min = 0.15,
  max = 0.85,
): void {
  e.preventDefault();
  const el = ref.current;
  if (!el) return;
  const rect = el.getBoundingClientRect();
  document.body.style.userSelect = "none";
  document.body.style.cursor = axis === "x" ? "col-resize" : "row-resize";

  const onMove = (mv: MouseEvent) => {
    const raw =
      axis === "x"
        ? (mv.clientX - rect.left) / rect.width
        : (mv.clientY - rect.top) / rect.height;
    setter(Math.max(min, Math.min(max, raw)));
  };
  const onUp = () => {
    document.body.style.userSelect = "";
    document.body.style.cursor = "";
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
  };
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
}

function Resizer({
  direction,
  onMouseDown,
}: {
  direction: "x" | "y";
  onMouseDown: (e: React.MouseEvent<HTMLDivElement>) => void;
}) {
  const [hover, setHover] = useState(false);
  const isCol = direction === "x";

  return (
    <div
      onMouseDown={onMouseDown}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        flexShrink: 0,
        ...(isCol
          ? { width: 10, cursor: "col-resize", display: "flex", justifyContent: "center", alignItems: "stretch" }
          : { height: 8, cursor: "row-resize", display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "stretch" }),
      }}
    >
      <div
        style={{
          ...(isCol ? { width: 2, height: "100%" } : { height: 2, width: "100%" }),
          background: hover ? "#6699cc99" : "var(--border)",
          borderRadius: 1,
          transition: "background 0.15s",
        }}
      />
    </div>
  );
}


function GitHubLink() {
  return (
    <a
      href={REPO_URL}
      target="_blank"
      rel="noopener noreferrer"
      title="View on GitHub (program & loss syntax docs)"
      style={{
        position: "fixed",
        top: 16,
        right: 64,
        zIndex: 100,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "var(--btn-bg)",
        border: "1px solid var(--btn-border)",
        borderRadius: 8,
        padding: "6px 10px",
        cursor: "pointer",
        lineHeight: 1,
        color: "var(--text-primary)",
        transition: "background 0.15s",
      }}
      onMouseEnter={(e) => (e.currentTarget.style.background = "var(--btn-hover)")}
      onMouseLeave={(e) => (e.currentTarget.style.background = "var(--btn-bg)")}
    >
      { }
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
      </svg>
    </a>
  );
}

// ─────────────────────────────────────────────────────────────────────────────

export default function App() {
  const { theme, toggle } = useTheme();
  const { steps, status, error, errorKind, outcome, lastRequest, wallClockMs, run, abort } = useOptimization();
  const displayError = error == null ? null : formatRunError(errorKind, error);

  const [program, setProgram] = useState(EXAMPLES[0].program);
  const [lossFunctions, setLossFunctions] = useState<LossFunctionInfo[]>([]);
  const [optimizers, setOptimizers] = useState<string[]>([]);
  const [dismissedError, setDismissedError] = useState<string | null>(null);
  const [health, setHealth] = useState<ServerHealth | null>(null);
  const healthTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [lossMode, setLossMode] = useState<LossMode>("builtin");
  const [lossName, setLossName] = useState(EXAMPLES[0].loss_function);
  const [lossSource, setLossSource] = useState(DEFAULT_LOSS_SOURCE);
  const [dslParams, setDslParams] = useState<LossParamInfo[]>([]);
  const [dslErrors, setDslErrors] = useState<string[]>([]);

  // Built-in loss definitions, keyed by name, from the backend response.
  const lossDefs = useMemo<Record<string, string>>(() => {
    const m: Record<string, string> = {};
    for (const lf of lossFunctions) if (lf.definition) m[lf.name] = lf.definition;
    return m;
  }, [lossFunctions]);

  // Sources we may safely overwrite when the user switches loss/mode: the
  // boilerplate and any *unmodified* built-in definition. A hand-edited source
  // is never one of these, so user edits are preserved.
  const pristineSources = useMemo(
    () => new Set<string>([DEFAULT_LOSS_SOURCE, ...Object.values(lossDefs)]),
    [lossDefs],
  );

  function isPristineLossSource(src: string): boolean {
    return src.trim() === "" || pristineSources.has(src);
  }

  function lossPreview(name: string): { source: string; available: boolean } {
    const src = lossDefs[name];
    if (src) return { source: src, available: true };
    return {
      source:
        `// No DSL definition available for "${name}" yet.\n` +
        `// Switch to 'custom' to write your own loss.`,
      available: false,
    };
  }

  function handleLossModeChange(mode: LossMode) {
    if (mode === "custom") {
      const def = lossDefs[lossName];
      if (def && isPristineLossSource(lossSource)) {
        setLossSource(def);
      }
    }
    setLossMode(mode);
  }

  // panel size fractions (0–1)
  const [leftFrac, setLeftFrac] = useState(0.5);       // left col width
  const [rightYFrac, setRightYFrac] = useState(0.6);   // results / distribution split
  const [editorFrac, setEditorFrac] = useState(0.6);   // soga / dsl split

  const mainRef = useRef<HTMLDivElement>(null);
  const rightColRef = useRef<HTMLDivElement>(null);
  const leftEditorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchLossFunctions().then(setLossFunctions).catch(console.error);
    fetchOptimizers().then(setOptimizers).catch(console.error);

    function pollHealth() {
      fetchHealth().then(setHealth).catch(() => setHealth(null));
      healthTimerRef.current = setTimeout(pollHealth, 15_000);
    }
    pollHealth();
    return () => { if (healthTimerRef.current) clearTimeout(healthTimerRef.current); };
  }, []);

  function handleSelectExample(ex: Example) {
    setProgram(ex.program);
  }

  function handleDownload() {
    if (!lastRequest || steps.length === 0) return;
    const exportOutcome = status === "error" ? (errorKind ?? "error") : (outcome ?? "not_converged");
    downloadBundle(
      buildExportBundle({ steps, request: lastRequest, outcome: exportOutcome, error, wallClockMs }),
    );
  }

  const preview = lossPreview(lossName);
  const selectedDef = lossDefs[lossName];
  const isCustom = lossMode === "custom";

  return (
    <div
      style={{
        height: "100vh",
        width: "100vw",
        display: "flex",
        flexDirection: "column",
        padding: 20,
        gap: 16,
        background: "var(--bg-page)",
        transition: "background 0.2s",
        boxSizing: "border-box",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <ThemeToggle theme={theme} onToggle={toggle} />
        <GitHubLink />
        {health !== null && (
          <span
            title={`${health.slots_free} of ${health.max_concurrent} optimization slots free`}
            style={{ fontSize: 11, color: health.slots_free === 0 ? "#ec5f67" : "var(--text-muted)" }}
          >
            {health.slots_free === 0 ? "server busy" : `${health.slots_free}/${health.max_concurrent} slots free`}
          </span>
        )}
      </div>

      {/* main: left | resizer | right */}
      <div ref={mainRef} style={{ flex: 1, minHeight: 0, display: "flex" }}>

        {/* left panel */}
        <div style={{ flex: leftFrac, minWidth: 200, ...PANEL, background: "var(--bg-surface)" }}>
          {/* editor area: soga [resizer] loss-editor (always shown) */}
          <div ref={leftEditorRef} style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
            <div style={{ flex: editorFrac, minHeight: 60 }}>
              <SogaEditor
                theme={theme}
                value={program}
                onChange={(v) => { if (v !== undefined) setProgram(v); }}
              />
            </div>

            <Resizer direction="y" onMouseDown={(e) => startDrag(e, leftEditorRef, "y", setEditorFrac, 0.15, 0.85)} />
            <div style={{ flex: 1 - editorFrac, minHeight: 40, display: "flex", flexDirection: "column" }}>
              <div style={{
                borderBottom: "1px solid var(--border)",
                padding: "5px 14px",
                fontSize: 11,
                color: "var(--text-muted)",
                background: "var(--bg-page)",
                display: "flex",
                alignItems: "center",
                gap: 8,
                flexShrink: 0,
              }}>
                {isCustom ? (
                  <>
                    <span>loss function</span>
                    {dslErrors.length > 0 && (
                      <span style={{ color: "#ec5f67" }}>
                        {dslErrors.length} error{dslErrors.length > 1 ? "s" : ""}
                      </span>
                    )}
                    {selectedDef && (
                      <button
                        type="button"
                        onClick={() => setLossSource(selectedDef)}
                        title={`Replace the editor with the built-in ${lossName} definition`}
                        style={{
                          marginLeft: "auto",
                          background: "transparent",
                          border: "1px solid var(--border)",
                          borderRadius: 5,
                          color: "var(--text-muted)",
                          fontSize: 10,
                          padding: "2px 7px",
                          cursor: "pointer",
                          fontFamily: "inherit",
                        }}
                      >
                        load {lossName} definition
                      </button>
                    )}
                  </>
                ) : (
                  <span>
                    loss preview · <span style={{ color: "#fac863" }}>{lossName}</span>
                    {preview.available ? " · read-only" : " · no definition yet"}
                  </span>
                )}
              </div>
              <div style={{ flex: 1, minHeight: 0 }}>
                <LossDSLEditor
                  key={isCustom ? "loss-editor-custom" : `loss-preview-${lossName}`}
                  theme={theme}
                  value={isCustom ? lossSource : preview.source}
                  onChange={setLossSource}
                  onValidate={(params, errors) => {
                    setDslParams(params);
                    setDslErrors(errors);
                  }}
                  readOnly={!isCustom}
                />
              </div>
            </div>
          </div>

          <Toolbar
            program={program}
            lossMode={lossMode}
            lossSource={lossSource}
            lossName={lossName}
            onLossNameChange={setLossName}
            dslParams={dslParams}
            dslErrors={dslErrors}
            onLossModeChange={handleLossModeChange}
            lossFunctions={lossFunctions}
            optimizers={optimizers}
            status={status}
            onRun={(req: OptimizationRequest) => run(req)}
            onAbort={abort}
            onSelectExample={handleSelectExample}
          />
        </div>

        {/* left / right resizer */}
        <Resizer direction="x" onMouseDown={(e) => startDrag(e, mainRef, "x", setLeftFrac)} />

        {/* right column: results [resizer] distribution */}
        <div ref={rightColRef} style={{ flex: 1 - leftFrac, minWidth: 200, display: "flex", flexDirection: "column" }}>
          <div style={{ ...PANEL, flex: rightYFrac, background: "var(--bg-results)", minHeight: 60 }}>
            <ResultsPanel
              steps={steps}
              status={status}
              error={displayError}
              outcome={outcome}
              theme={theme}
              canDownload={steps.length > 0}
              onDownload={handleDownload}
            />
          </div>
          <Resizer direction="y" onMouseDown={(e) => startDrag(e, rightColRef, "y", setRightYFrac)} />
          <div style={{ ...PANEL, flex: 1 - rightYFrac, background: "var(--bg-results)", minHeight: 60 }}>
            <DistributionPanel steps={steps} theme={theme} />
          </div>
        </div>
      </div>

      <ErrorToast
        message={error !== dismissedError ? displayError : null}
        onDismiss={() => setDismissedError(error)}
      />
    </div>
  );
}
