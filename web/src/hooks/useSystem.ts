import { useMutation, useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';

export type ShutdownMode = 'wait' | 'cancel' | 'requeue';

export interface SystemStatus {
  state: 'running' | 'shutting_down';
  shutdown: {
    requested_at: string;
    mode: ShutdownMode | null;
    reason: 'user' | 'idle' | null;
    worker_stopped: boolean;
  } | null;
  worker: { state: string; job: { id: number; stage: string | null } | null };
  jobs: { running: { id: number; stage: string | null } | null; queued: number };
  auto_shutdown_min: number;
  idle_min: number;
}

export function useSystemStatus(refetchInterval: number | false = false) {
  return useQuery({
    queryKey: ['system-status'],
    queryFn: async () => (await api.get<SystemStatus>('/system/status')).data,
    refetchInterval,
    // During a shutdown the API goes away: keep the last answer.
    retry: false,
  });
}

export function useShutdown() {
  return useMutation({
    mutationFn: async (mode: ShutdownMode) => (await api.post('/system/shutdown', { mode })).data,
  });
}
