import { useMutation } from '@tanstack/react-query';
import { api } from '@/lib/api';

interface UploadResponse {
  path: string;
  filename: string;
  size: number;
}

interface UploadArgs {
  file: File;
  onProgress?: (percent: number) => void;
}

// The file is the request body and the API streams it to disk. Asking first
// saves sending gigabytes only to hear that the file is too big or the disk
// is full.
export function useUpload() {
  return useMutation({
    mutationFn: async ({ file, onProgress }: UploadArgs): Promise<UploadResponse> => {
      await api.get('/upload/check', { params: { filename: file.name, size: file.size } });
      const res = await api.post('/upload/', file, {
        params: { filename: file.name },
        headers: { 'Content-Type': 'application/octet-stream' },
        timeout: 0,
        onUploadProgress: (e) => {
          if (e.total) onProgress?.(Math.round((e.loaded * 100) / e.total));
        },
      });
      return res.data;
    },
  });
}
