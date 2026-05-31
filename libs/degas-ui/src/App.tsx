import { useEffect, useRef, useState } from "react";
import { fetchLossFunctions, fetchOptimizers, fetchHealth, type ServerHealth } from "./api";
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
import type { LossFunctionInfo, LossMode, LossParamInfo, OptimizationRequest } from "./types";

const DEFAULT_LOSS_SOURCE = `// Custom loss — parameter names cannot be grammar keywords.
// Use short aliases: d: dist, ts: traj_set, idx: index_list, etc.

loss signal_error(d: dist, target: scalar) =
    sum( (d.mean[range(0, 5)] - ones(range(0, 5)) * target) ^ 2 )
`;

const PANEL: React.CSSProperties = {
  border: "1px solid var(--border)",
  borderRadius: 12,
  overflow: "hidden",
  display: "flex",
  flexDirection: "column",
};

// ── drag-to-resize ────────────────────────────────────────────────────────────

function startDrag(
  ref: React.RefObject<HTMLDivElement | null>,
  axis: "x" | "y",
  setter: (f: number) => void,
  min = 0.15,
  max = 0.85,
): (e: React.MouseEvent<HTMLDivElement>) => void {
  return (e) => {
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
  };
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

// ─────────────────────────────────────────────────────────────────────────────

export default function App() {
  const { theme, toggle } = useTheme();
  const { steps, status, error, converged, run, abort } = useOptimization();

  const [program, setProgram] = useState(EXAMPLES[0].program);
  const [lossFunctions, setLossFunctions] = useState<LossFunctionInfo[]>([]);
  const [optimizers, setOptimizers] = useState<string[]>([]);
  const [dismissedError, setDismissedError] = useState<string | null>(null);
  const [health, setHealth] = useState<ServerHealth | null>(null);
  const healthTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [lossMode, setLossMode] = useState<LossMode>("builtin");
  const [lossSource, setLossSource] = useState(DEFAULT_LOSS_SOURCE);
  const [dslParams, setDslParams] = useState<LossParamInfo[]>([]);
  const [dslErrors, setDslErrors] = useState<string[]>([]);

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
          {/* editor area: soga [resizer dsl] */}
          <div ref={leftEditorRef} style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
            <div style={{ flex: lossMode === "custom" ? editorFrac : 1, minHeight: 60 }}>
              <SogaEditor
                theme={theme}
                value={program}
                onChange={(v) => { if (v !== undefined) setProgram(v); }}
              />
            </div>

            {lossMode === "custom" && (
              <>
                <Resizer direction="y" onMouseDown={startDrag(leftEditorRef, "y", setEditorFrac, 0.15, 0.85)} />
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
                    <span>loss function</span>
                    {dslErrors.length > 0 && (
                      <span style={{ color: "#ec5f67" }}>
                        {dslErrors.length} error{dslErrors.length > 1 ? "s" : ""}
                      </span>
                    )}
                  </div>
                  <div style={{ flex: 1, minHeight: 0 }}>
                    <LossDSLEditor
                      theme={theme}
                      value={lossSource}
                      onChange={setLossSource}
                      onValidate={(params, errors) => {
                        setDslParams(params);
                        setDslErrors(errors);
                      }}
                    />
                  </div>
                </div>
              </>
            )}
          </div>

          <Toolbar
            program={program}
            lossMode={lossMode}
            lossSource={lossSource}
            dslParams={dslParams}
            dslErrors={dslErrors}
            onLossModeChange={setLossMode}
            lossFunctions={lossFunctions}
            optimizers={optimizers}
            status={status}
            onRun={(req: OptimizationRequest) => run(req)}
            onAbort={abort}
            onSelectExample={handleSelectExample}
          />
        </div>

        {/* left / right resizer */}
        <Resizer direction="x" onMouseDown={startDrag(mainRef, "x", setLeftFrac)} />

        {/* right column: results [resizer] distribution */}
        <div ref={rightColRef} style={{ flex: 1 - leftFrac, minWidth: 200, display: "flex", flexDirection: "column" }}>
          <div style={{ ...PANEL, flex: rightYFrac, background: "var(--bg-results)", minHeight: 60 }}>
            <ResultsPanel
              steps={steps}
              status={status}
              error={error}
              converged={converged}
              theme={theme}
            />
          </div>
          <Resizer direction="y" onMouseDown={startDrag(rightColRef, "y", setRightYFrac)} />
          <div style={{ ...PANEL, flex: 1 - rightYFrac, background: "var(--bg-results)", minHeight: 60 }}>
            <DistributionPanel steps={steps} theme={theme} />
          </div>
        </div>
      </div>

      <ErrorToast
        message={error !== dismissedError ? error : null}
        onDismiss={() => setDismissedError(error)}
      />
    </div>
  );
}
