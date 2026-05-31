import { useEffect, useMemo, useRef, useState } from "react";
import type { RunStatus, StepOut } from "../types";

const COLORS = [
  "#6699cc",
  "#99c794",
  "#c594c5",
  "#fac863",
  "#ec5f67",
  "#5fb3b3",
];

function traceColor(index: number) {
  return COLORS[index % COLORS.length];
}

interface Props {
  steps: StepOut[];
  status: RunStatus;
  error: string | null;
  converged: boolean | null;
  theme: "light" | "dark";
}

export function ResultsPanel({ steps, status, error, converged, theme }: Props) {
  const dark = theme === "dark";
  const bg = dark ? "#0a0a0a" : "#fafaf9";
  const fg = dark ? "#e8e6e0" : "#1a1a1a";
  const grid = dark ? "#1e1e1e" : "#e2ddd8";

  const paramNames = useMemo(
    () => (steps.length > 0 ? Object.keys(steps[0].params) : []),
    [steps],
  );
  const paramKey = paramNames.join(",");

  // 'hidden' is the inverse of visible — everything is shown by default.
  // Reset on new run (new paramKey) using conditional setState during render.
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [lastParamKey, setLastParamKey] = useState("");
  if (paramKey !== lastParamKey) {
    setLastParamKey(paramKey);
    setHidden(new Set());
  }

  function toggle(key: string) {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  const isVisible = (key: string) => !hidden.has(key);

  const isEmpty = steps.length === 0;
  const finalParams = steps.length > 0 ? steps[steps.length - 1].params : null;
  const plotRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isEmpty || !plotRef.current) return;

    const xs = steps.map((s) => s.step);

    const lossTrace = {
      x: xs,
      y: steps.map((s) => s.loss),
      name: "Loss",
      type: "scatter",
      mode: "lines",
      yaxis: "y",
      line: { color: "#f99157", width: 2 },
      visible: isVisible("loss") ? true : "legendonly",
    };

    const paramTraces = paramNames.map((name, i) => ({
      x: xs,
      y: steps.map((s) => s.params[name] ?? null),
      name,
      type: "scatter",
      mode: "lines",
      yaxis: "y2",
      line: { color: traceColor(i), width: 2 },
      visible: isVisible(name) ? true : "legendonly",
    }));

    const layout = {
      paper_bgcolor: bg,
      plot_bgcolor: bg,
      font: {
        color: fg,
        family: "Menlo, Consolas, 'Courier New', monospace",
        size: 11,
      },
      margin: { t: 16, r: 60, b: 48, l: 60 },
      xaxis: {
        title: { text: "step", font: { size: 11 } },
        gridcolor: grid,
        zerolinecolor: grid,
        tickfont: { size: 10 },
      },
      yaxis: {
        title: { text: "loss", font: { size: 11 } },
        gridcolor: grid,
        zerolinecolor: grid,
        tickfont: { size: 10 },
        side: "left",
      },
      yaxis2: {
        title: { text: "params", font: { size: 11 } },
        gridcolor: "transparent",
        zerolinecolor: "transparent",
        tickfont: { size: 10 },
        overlaying: "y",
        side: "right",
      },
      legend: {
        bgcolor: "transparent",
        font: { size: 10 },
        x: 0.01,
        y: 0.99,
        xanchor: "left",
        yanchor: "top",
      },
      showlegend: false,
    };

    const config = { displayModeBar: false, responsive: true };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const Plotly = (window as any).Plotly;
    if (Plotly && plotRef.current) {
      Plotly.react(
        plotRef.current,
        [lossTrace, ...paramTraces],
        layout,
        config,
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [steps, hidden, theme]);

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          padding: "10px 14px 0",
          flexWrap: "wrap",
        }}
      >
        <TogglePill
          label="Loss"
          color="#f99157"
          active={isVisible("loss")}
          onClick={() => toggle("loss")}
        />
        {paramNames.map((name, i) => (
          <TogglePill
            key={name}
            label={`_${name}`}
            color={traceColor(i)}
            active={isVisible(name)}
            onClick={() => toggle(name)}
          />
        ))}

        {status === "running" && (
          <span
            style={{
              marginLeft: "auto",
              fontSize: 11,
              color: "var(--text-muted)",
            }}
          >
            step {steps.length > 0 ? steps[steps.length - 1].step + 1 : "…"}
          </span>
        )}
        {status === "done" && steps.length > 0 && (
          <span
            style={{
              marginLeft: "auto",
              fontSize: 11,
              color: "var(--text-muted)",
            }}
          >
            {steps.length} steps
          </span>
        )}
        {converged !== null && status === "done" && (
          <span
            style={{
              marginLeft: converged ? 0 : "auto",
              fontSize: 10,
              padding: "2px 7px",
              borderRadius: 20,
              background: converged ? "#99c79422" : "#ec5f6722",
              border: `1px solid ${converged ? "#99c794" : "#ec5f67"}`,
              color: converged ? "#99c794" : "#ec5f67",
            }}
          >
            {converged ? "converged" : "not converged"}
          </span>
        )}
      </div>

      {status === "done" && finalParams && Object.keys(finalParams).length > 0 && (
        <div
          style={{
            display: "flex",
            gap: 16,
            padding: "6px 14px",
            borderTop: "1px solid var(--border)",
            flexWrap: "wrap",
          }}
        >
          {Object.entries(finalParams).map(([name, val]) => (
            <span key={name} style={{ fontSize: 11, fontFamily: "inherit" }}>
              <span style={{ color: "var(--text-muted)" }}>_{name} = </span>
              <span style={{ color: "#f99157" }}>{val.toFixed(4)}</span>
            </span>
          ))}
        </div>
      )}

      {/* Plotly div stays mounted to keep the ref alive */}
      <div style={{ flex: 1, minHeight: 0, position: "relative" }}>
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: isEmpty ? "flex" : "none",
            alignItems: "center",
            justifyContent: "center",
            color: "var(--text-muted)",
            fontSize: 12,
            letterSpacing: "0.08em",
          }}
        >
          {error ? (
            <span
              style={{ color: "#ec5f67", maxWidth: 320, textAlign: "center" }}
            >
              {error}
            </span>
          ) : (
            "run optimization to see results"
          )}
        </div>
        <div
          ref={plotRef}
          style={{
            width: "100%",
            height: "100%",
            display: isEmpty ? "none" : "block",
          }}
        />
      </div>
    </div>
  );
}

function TogglePill({
  label,
  color,
  active,
  onClick,
}: {
  label: string;
  color: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 5,
        padding: "3px 8px",
        borderRadius: 20,
        border: `1px solid ${active ? color : "var(--border)"}`,
        background: active ? `${color}22` : "transparent",
        color: active ? color : "var(--text-muted)",
        fontSize: 11,
        cursor: "pointer",
        fontFamily: "inherit",
        transition: "all 0.15s",
      }}
    >
      <span
        style={{
          width: 7,
          height: 7,
          borderRadius: "50%",
          background: active ? color : "var(--text-muted)",
          flexShrink: 0,
        }}
      />
      {label}
    </button>
  );
}
