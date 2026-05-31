import { useCallback, useEffect, useRef, useState } from 'react'
import { streamOptimization } from '../api'
import type { OptimizationRequest, RunStatus, StepOut } from '../types'

export function useOptimization() {
  const [steps, setSteps] = useState<StepOut[]>([])
  const [status, setStatus] = useState<RunStatus>('idle')
  const [error, setError] = useState<string | null>(null)
  const [converged, setConverged] = useState<boolean | null>(null)
  const ctrlRef = useRef<AbortController | null>(null)
  // Set to true at run() start; first incoming step clears old data before appending.
  const clearOnNextStepRef = useRef(false)

  const run = useCallback((request: OptimizationRequest) => {
    ctrlRef.current?.abort()
    ctrlRef.current = new AbortController()
    clearOnNextStepRef.current = true

    setError(null)
    setConverged(null)
    setStatus('running')

    streamOptimization(
      request,
      (step) => {
        if (clearOnNextStepRef.current) {
          clearOnNextStepRef.current = false
          setSteps([step])
        } else {
          setSteps((prev) => [...prev, step])
        }
      },
      (conv: boolean) => { setConverged(conv); setStatus('done') },
      (detail) => {
        setError(detail)
        setStatus('error')
      },
      ctrlRef.current.signal,
    )
  }, [])

  const abort = useCallback(() => {
    ctrlRef.current?.abort()
    setConverged(null)
    setStatus('idle')
  }, [])

  useEffect(() => () => ctrlRef.current?.abort(), [])

  return { steps, status, error, converged, run, abort }
}
