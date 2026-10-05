import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';

export function useConfig() {
  return useQuery({
    queryKey: ['config'],
    queryFn: async () => {
      const res = await api.get('/config/');
      return res.data;
    },
  });
}

export function useUpdateConfig() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data: Record<string, unknown>) => {
      const res = await api.put('/config/', data);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['config'] });
    },
  });
}

// Removes dotted keys (e.g. "scoring.max_clips_per_video") from the user
// settings; an empty list resets everything to the defaults.
export function useResetConfig() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (keys: string[]) => {
      const res = await api.post('/config/reset', { keys });
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['config'] });
    },
  });
}

export function useProfiles() {
  return useQuery({
    queryKey: ['profiles'],
    queryFn: async () => {
      const res = await api.get('/config/profiles');
      return res.data;
    },
  });
}
