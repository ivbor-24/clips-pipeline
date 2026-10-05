import { useState } from 'react';
import { useCreateJob } from '@/hooks/useJobs';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function PipelineForm() {
  const [inputSource, setInputSource] = useState('');
  const createJob = useCreateJob();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await createJob.mutateAsync({ input_source: inputSource });
    setInputSource('');
  };

  return (
    <Card className="w-full max-w-lg">
      <CardHeader>
        <CardTitle>New Pipeline Job</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Input
              placeholder="Video file path or URL"
              value={inputSource}
              onChange={(e) => setInputSource(e.target.value)}
            />
          </div>
          <Button type="submit" className="w-full" disabled={createJob.isPending}>
            {createJob.isPending ? 'Submitting...' : 'Start Pipeline'}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
