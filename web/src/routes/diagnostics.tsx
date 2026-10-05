import { createFileRoute, Link } from '@tanstack/react-router';
import { useState } from 'react';
import { useDiagnostics, useRequestGpuCheck, type Diagnostics, type ModelEntry } from '@/hooks/useDiagnostics';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

export const Route = createFileRoute('/diagnostics')({
  component: DiagnosticsPage,
});

const GB = 1e9;
// Below this a job may not fit (a lecture video, its copy and the clips).
const LOW_DISK_GB = 20;

function formatTime(iso: string | null | undefined): string {
  return iso ? new Date(iso).toLocaleString() : '—';
}

function formatGb(bytes: number | null | undefined): string {
  if (bytes == null) return '—';
  return bytes < GB ? `${(bytes / 1e6).toFixed(1)} MB` : `${(bytes / GB).toFixed(1)} GB`;
}

function WorkerStatus({ worker }: { worker: Diagnostics['worker'] }) {
  switch (worker.state) {
    case 'idle':
      return <><Badge variant="secondary">running</Badge> <span className="text-sm">waiting for jobs</span></>;
    case 'busy':
      return (
        <>
          <Badge>running</Badge>{' '}
          <span className="text-sm">
            job <Link to="/jobs/$jobId" params={{ jobId: String(worker.job?.id) }} className="underline">#{worker.job?.id}</Link>
            {worker.job?.stage ? `, ${worker.job.stage}` : ''}
          </span>
        </>
      );
    case 'not_responding':
      return (
        <>
          <Badge variant="destructive">not responding</Badge>{' '}
          <span className="text-sm">
            last seen {formatTime(worker.heartbeat_at)}: it stopped or hangs. Docker: just status, docker compose logs;
            dev mode: just worker
          </span>
        </>
      );
    default:
      return (
        <>
          <Badge variant="destructive">not started</Badge>{' '}
          <span className="text-sm">the job worker has not run yet: jobs stay queued (Docker: just up; dev mode: just worker)</span>
        </>
      );
  }
}

const MODEL_STATES: Record<string, { label: string; bad?: boolean; good?: boolean }> = {
  verified: { label: 'checked (sha256)', good: true },
  unverified: { label: 'size OK, sha256 checked on next use' },
  present: { label: 'downloaded', good: true },
  unpinned: { label: 'present (not pinned)' },
  missing: { label: 'missing: run ./setup.sh (Docker) or just prefetch-models', bad: true },
  wrong_size: { label: 'wrong size: download again (./setup.sh or just prefetch-models)', bad: true },
};

function ModelState({ model }: { model: ModelEntry }) {
  if (model.state.startsWith('api:')) {
    return <Badge variant="secondary">API provider: {model.state.slice(4)}</Badge>;
  }
  const info = MODEL_STATES[model.state] ?? { label: model.state };
  return (
    <Badge variant={info.bad ? 'destructive' : info.good ? 'default' : 'secondary'}>{info.label}</Badge>
  );
}

function copyText(text: string): Promise<void> {
  // The clipboard API needs a secure context; a LAN address over http is not one.
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text);
  }
  const area = document.createElement('textarea');
  area.value = text;
  document.body.appendChild(area);
  area.select();
  document.execCommand('copy');
  document.body.removeChild(area);
  return Promise.resolve();
}

function DiagnosticsPage() {
  const { data, isLoading, error } = useDiagnostics();
  const requestCheck = useRequestGpuCheck();
  const [copied, setCopied] = useState(false);

  if (isLoading) return <div className="p-6">Loading...</div>;
  if (error || !data) return <div className="p-6 text-red-600">Could not load diagnostics: {String(error)}</div>;

  const gpu = data.gpu_check;
  const checkPending = Boolean(gpu?.pending) || requestCheck.isPending;

  const handleCopy = async () => {
    await copyText(JSON.stringify({ ...data, copied_at: new Date().toISOString() }, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="p-6 space-y-6 max-w-5xl mx-auto">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Diagnostics</h1>
        <div className="flex gap-2">
          <Button variant="outline" onClick={handleCopy}>{copied ? 'Copied' : 'Copy report'}</Button>
          <Link to="/dashboard"><Button variant="outline">Back to Dashboard</Button></Link>
        </div>
      </div>

      <Card>
        <CardHeader><CardTitle>System</CardTitle></CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div>Version: <b>{data.system.version}</b></div>
          <div>
            GPU backend: <b>{data.system.backend ?? gpu?.backend ?? 'unknown'}</b>
            {data.system.docker ? ' (Docker)' : ' (native install)'}
          </div>
          <div>Job worker: <WorkerStatus worker={data.worker} /></div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>GPU check</CardTitle>
            <Button onClick={() => requestCheck.mutate()} disabled={checkPending}>
              {checkPending ? 'Checking...' : 'Check again'}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <div className="text-gray-500">
            Run by the job worker, which has the GPU: last {formatTime(gpu?.checked_at)}.
            {checkPending && data.worker.state === 'busy' && ' It runs again when the current job ends.'}
          </div>
          {gpu?.error && <div className="text-red-600">The check did not run: {gpu.error}</div>}
          {gpu?.results && gpu.results.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-8"></TableHead>
                  <TableHead>Component</TableHead>
                  <TableHead>Result</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {gpu.results.map((r) => (
                  <TableRow key={r.component}>
                    <TableCell className={r.warning ? 'text-yellow-600' : r.ok === false ? 'text-red-600' : r.ok ? 'text-green-600' : 'text-gray-400'}>
                      {r.warning ? '!' : r.ok === false ? '✗' : r.ok ? '✓' : '–'}
                    </TableCell>
                    <TableCell>{r.component}</TableCell>
                    <TableCell>
                      <div>{r.detail}</div>
                      {r.hint && r.ok !== true && <div className="text-gray-500">{r.hint}</div>}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            !gpu?.error && <div className="text-gray-500">No result yet: the worker runs the check when it starts.</div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Models</CardTitle></CardHeader>
        <CardContent>
          {data.models_error && <div className="text-red-600 text-sm mb-2">Config could not be read: {data.models_error}</div>}
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Purpose</TableHead>
                <TableHead>Model</TableHead>
                <TableHead>State</TableHead>
                <TableHead>Size</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.models.map((m) => (
                <TableRow key={`${m.purpose}-${m.name}`}>
                  <TableCell>{m.purpose}</TableCell>
                  <TableCell className="break-all">{m.name}</TableCell>
                  <TableCell><ModelState model={m} /></TableCell>
                  <TableCell>{formatGb(m.size)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>ffmpeg and disk</CardTitle></CardHeader>
        <CardContent className="space-y-3 text-sm">
          <div className={data.ffmpeg.ok ? '' : 'text-red-600'}>
            {data.ffmpeg.version ?? `ffmpeg does not run: ${data.ffmpeg.error ?? 'unknown error'}`}
          </div>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Data</TableHead>
                <TableHead>Directory</TableHead>
                <TableHead>Free</TableHead>
                <TableHead>Total</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.disk.map((d) => (
                <TableRow key={d.name}>
                  <TableCell>{d.name}</TableCell>
                  <TableCell>{d.path}</TableCell>
                  <TableCell className={d.free != null && d.free < LOW_DISK_GB * GB ? 'text-red-600' : ''}>
                    {formatGb(d.free)}
                  </TableCell>
                  <TableCell>{formatGb(d.total)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
