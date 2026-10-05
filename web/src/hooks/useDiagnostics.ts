import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';

export interface GpuCheckResult {
  component: string;
  ok: boolean | null;
  detail: string;
  hint: string;
  warning: boolean;
}

export interface ModelEntry {
  name: string;
  purpose: string;
  path?: string | null;
  // missing | verified | unverified | wrong_size | unpinned | present | api:<provider>
  state: string;
  size?: number | null;
  expected_size?: number | null;
}

export interface Diagnostics {
  system: { version: string; backend: string | null; docker: boolean };
  worker: {
    state: 'never_started' | 'idle' | 'busy' | 'not_responding';
    heartbeat_at: string | null;
    started_at?: string | null;
    version?: string | null;
    job: { id: number; stage: string | null } | null;
  };
  gpu_check: {
    backend?: string | null;
    ok?: boolean;
    error?: string;
    results?: GpuCheckResult[];
    checked_at: string | null;
    requested_at: string | null;
    pending: boolean;
  } | null;
  models: ModelEntry[];
  models_error: string | null;
  ffmpeg: { ok: boolean; version: string | null; error?: string };
  disk: { name: string; path: string; free: number | null; total: number | null }[];
}

export function useDiagnostics() {
  return useQuery({
    queryKey: ['diagnostics'],
    queryFn: async () => (await api.get<Diagnostics>('/diagnostics/')).data,
    // A requested GPU check finishes in the worker; poll until it is in.
    refetchInterval: (query) => (query.state.data?.gpu_check?.pending ? 3000 : 15000),
  });
}

export function useRequestGpuCheck() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async () => (await api.post('/diagnostics/gpu-check')).data,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['diagnostics'] }),
  });
}
