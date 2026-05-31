export type ParamType = 'float' | 'int' | 'float[][]' | 'int[]'

export interface LossParamDef {
  name: string
  type: ParamType
  required: boolean
  default?: number | null
}

export interface LossFunctionInfo {
  name: string
  params: LossParamDef[]
}

export interface StepOut {
  step: number
  loss: number
  params: Record<string, number>
  elapsed_ms: number
  dist: { var_list: string[]; mean: number[] } | null
}

export type DslParamType = 'traj_set' | 'index_list' | 'scalar' | 'int' | null

export interface LossParamInfo {
  name: string
  type: DslParamType
}

export interface LossValidateResponse {
  errors: string[]
  params: LossParamInfo[]
}

export type LossMode = 'builtin' | 'custom'

export interface OptimizationRequest {
  program: string
  program_language: 'soga' | 'soga_highlevel'
  compile_seed: number
  loss_function?: string
  loss_kwargs: Record<string, unknown>
  loss_source?: string
  loss_bindings?: Record<string, unknown>
  optimizer: string
  optimizer_kwargs: Record<string, unknown>
  initial_params: Record<string, number>
  n_steps: number
  return_dist_summary: boolean
  tolerance?: number | null
  patience?: number
}

export type RunStatus = 'idle' | 'running' | 'done' | 'error'
