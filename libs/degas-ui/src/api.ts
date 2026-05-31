import type { LossFunctionInfo, LossValidateResponse, OptimizationRequest, StepOut } from "./types";

const BASE = "/api";

export interface ServerHealth {
  slots_free: number;
  max_concurrent: number;
}

export async function fetchHealth(): Promise<ServerHealth> {
  const res = await fetch("/health");
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function fetchLossFunctions(): Promise<LossFunctionInfo[]> {
  const res = await fetch(`${BASE}/optimization/loss-functions`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function fetchOptimizers(): Promise<string[]> {
  const res = await fetch(`${BASE}/optimization/optimizers`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function validateLossSource(source: string): Promise<LossValidateResponse> {
  const res = await fetch(`${BASE}/optimization/loss/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export function streamOptimization(
  request: OptimizationRequest,
  onStep: (step: StepOut) => void,
  onEnd: (converged: boolean, finalParams: Record<string, number>) => void,
  onError: (detail: string) => void,
  signal: AbortSignal,
): void {
  fetch(`${BASE}/optimization/run?stream=true`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(request),
    signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        const err = await res
          .json()
          .catch(() => ({ detail: `HTTP ${res.status}` }));
        onError(
          typeof err.detail === "string"
            ? err.detail
            : JSON.stringify(err.detail),
        );
        return;
      }

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop()!;

        for (const part of parts) {
          if (!part.trim()) continue;
          const lines = part.split("\n");
          const eventType = lines
            .find((l) => l.startsWith("event:"))
            ?.slice(6)
            .trim();
          const dataLine = lines
            .find((l) => l.startsWith("data:"))
            ?.slice(5)
            .trim();
          if (!dataLine) continue;
          try {
            const data = JSON.parse(dataLine);
            if (eventType === "step") onStep(data);
            else if (eventType === "end")
              onEnd(data.converged, data.final_params);
            else if (eventType === "error") onError(data.detail);
          } catch {
            // malformed event — ignore
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") onError(String(err));
    });
}
