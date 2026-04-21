import { AgentModel } from './types';

export const INITIAL_MODELS: AgentModel[] = [
  { id: 'm1', name: 'Gemini 3 Pro', provider: 'Gemini', version: 'Preview', avatarColor: 'bg-slate-600', capabilities: ['Reasoning', 'Vision'] },
  { id: 'm2', name: 'Gemini 2.5 Flash', provider: 'Gemini', version: 'Latest', avatarColor: 'bg-slate-500', capabilities: ['Speed', 'Context'] },
  { id: 'm3', name: 'GPT-4o', provider: 'OpenAI', version: '2024-05', avatarColor: 'bg-slate-700', capabilities: ['General', 'Code'] },
  { id: 'm4', name: 'Claude 3.5 Sonnet', provider: 'Anthropic', version: 'v1', avatarColor: 'bg-slate-600', capabilities: ['Writing', 'Nuance'] },
  { id: 'm5', name: 'Llama 3 70B', provider: 'Meta', version: 'Instruct', avatarColor: 'bg-slate-500', capabilities: ['Open', 'Instruct'] },
  { id: 'm6', name: 'Gemini 3 Flash', provider: 'Gemini', version: 'Preview', avatarColor: 'bg-slate-700', capabilities: ['Speed', 'Reasoning'] },
];

export const INITIAL_METRICS = {
  ldcl: 0,
  gne: 0,
  mlasProbability: 0,
  diversityCheck: false,
  softRoleBias: 0,
  tier0: {
    sd: 0,
    rc: 0,
    is: 0,
    eic: 0,
    stability: 0,
  }
};
