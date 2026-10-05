import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';

export function useJobs() {
  return useQuery({
    queryKey: ['jobs'],
    queryFn: async () => {
      const res = await api.get('/jobs/');
      return res.data;
    },
  });
}

export function useJob(jobId: number) {
  return useQuery({
    queryKey: ['job', jobId],
    queryFn: async () => {
      const res = await api.get(`/jobs/${jobId}`);
      return res.data;
    },
    enabled: !!jobId,
    // Poll status so the badge updates once the pipeline finishes (SSE only
    // streams stage events; the final status lives in the job row).
    refetchInterval: 5000,
  });
}

export function useCreateJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data: { input_source: string; job_type?: 'clips' | 'chapters'; config_overrides?: Record<string, unknown> }) => {
      const res = await api.post('/jobs/', data);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
  });
}

export function useCancelJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (jobId: number) => {
      await api.delete(`/jobs/${jobId}`);
    },
    onSuccess: (_data, jobId) => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
      queryClient.invalidateQueries({ queryKey: ['job', jobId] });
    },
  });
}

// Deletes a finished job with its clips and files (the cancel is DELETE /jobs/{id}).
export function useDeleteJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (jobId: number) => {
      await api.post(`/jobs/${jobId}/delete`);
    },
    onSuccess: (_data, jobId) => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
      queryClient.removeQueries({ queryKey: ['job', jobId] });
    },
  });
}

// Puts a failed or cancelled job back in the queue; it continues from the
// last finished stage.
export function useRetryJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (jobId: number) => {
      await api.post(`/jobs/${jobId}/retry`);
    },
    onSuccess: (_data, jobId) => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
      queryClient.invalidateQueries({ queryKey: ['job', jobId] });
    },
  });
}

export function useChapters(jobId: number, enabled: boolean = true) {
  return useQuery({
    queryKey: ['chapters', jobId],
    queryFn: async () => {
      const res = await api.get(`/jobs/${jobId}/chapters`);
      return res.data;
    },
    enabled: enabled && !!jobId,
  });
}

export function useChaptersText(jobId: number, enabled: boolean = true) {
  return useQuery({
    queryKey: ['chapters-text', jobId],
    queryFn: async () => {
      const res = await api.get(`/jobs/${jobId}/chapters/text`);
      return res.data;
    },
    enabled: enabled && !!jobId,
  });
}

export function useBroll(jobId: number, enabled: boolean = true) {
  return useQuery({
    queryKey: ['broll', jobId],
    queryFn: async () => {
      const res = await api.get(`/jobs/${jobId}/broll`);
      return res.data;
    },
    enabled: enabled && !!jobId,
  });
}

export function useBrollText(jobId: number, enabled: boolean = true) {
  return useQuery({
    queryKey: ['broll-text', jobId],
    queryFn: async () => {
      const res = await api.get(`/jobs/${jobId}/broll/text`);
      return res.data;
    },
    enabled: enabled && !!jobId,
  });
}

export interface Notice {
  stage: string;
  code: string;
  message: string;
  error?: string;
}

// Fallbacks that changed the job's result (e.g. clips picked without the LLM).
export function useNotices(jobId: number, active: boolean) {
  return useQuery({
    queryKey: ['notices', jobId],
    queryFn: async (): Promise<Notice[]> => {
      const res = await api.get(`/jobs/${jobId}/notices`);
      return res.data.notices;
    },
    enabled: !!jobId,
    refetchInterval: active ? 5000 : false,
  });
}
