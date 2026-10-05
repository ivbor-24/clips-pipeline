import { createFileRoute, Link, Outlet, useNavigate, useRouterState } from '@tanstack/react-router';
import { useEffect, useState, useRef } from 'react';
import {
  useJob,
  useCancelJob,
  useDeleteJob,
  useRetryJob,
  useNotices,
  useChapters,
  useChaptersText,
  useBroll,
  useBrollText,
} from '@/hooks/useJobs';
import { API_BASE_URL, api, withToken } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

export const Route = createFileRoute('/jobs/$jobId')({
  component: JobDetailPage,
});

interface StageStatus {
  name: string;
  status: 'pending' | 'started' | 'running' | 'completed' | 'failed' | 'skipped';
  progress?: number;
  startTime?: number;
  endTime?: number;
}

interface LogEntry {
  timestamp: string | null;
  message: string;
  level: string;
}

// GET /jobs/{id}/log: the JSON lines of the job's job.log.
interface JobLogLine {
  timestamp: string | null;
  level: string;
  event: string;
  fields: Record<string, unknown>;
}

function logMessage(line: JobLogLine): string {
  const fields = Object.entries(line.fields).map(([key, value]) =>
    `${key}=${typeof value === 'string' ? value : JSON.stringify(value)}`
  );
  return [line.event, ...fields].join(' ');
}

const LOG_POLL_MS = 3000;

const CLIPS_STAGES = ['ingestion', 'transcription', 'scoring', 'face_cropping', 'rendering', 'review'];
const CHAPTERS_STAGES = ['ingestion', 'transcription', 'chapters'];

function JobDetailPage() {
  const { jobId } = Route.useParams();
  const { data: job, isLoading } = useJob(Number(jobId));
  const isChaptersJob = job?.job_type === 'chapters';
  // The files appear when the job finishes: fetching earlier only gets 404s,
  // and a page opened mid-run would never load them.
  const chaptersReady = isChaptersJob && job?.status === 'completed';
  const { data: chaptersData } = useChapters(Number(jobId), chaptersReady);
  const { data: chaptersTextData } = useChaptersText(Number(jobId), chaptersReady);
  const { data: brollData } = useBroll(Number(jobId), chaptersReady);
  const { data: brollTextData } = useBrollText(Number(jobId), chaptersReady);
  const [stageProgress, setStageProgress] = useState<Record<string, StageStatus>>({});
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [copied, setCopied] = useState(false);
  const [copiedBroll, setCopiedBroll] = useState(false);
  const logBoxRef = useRef<HTMLDivElement>(null);
  const logOffset = useRef(0);
  // Bumped on Resume: the progress stream closed when the job ended.
  const [streamKey, setStreamKey] = useState(0);
  const cancelJob = useCancelJob();
  const jobActive = job?.status === 'queued' || job?.status === 'running';
  const { data: notices } = useNotices(Number(jobId), jobActive);
  const retryJob = useRetryJob();
  const deleteJob = useDeleteJob();
  const navigate = useNavigate();

  const baseStages = isChaptersJob ? CHAPTERS_STAGES : CLIPS_STAGES;
  // B-roll is an optional stage (draft jobs only): only show it once an
  // event for it has actually arrived, instead of hardcoding it into
  // CHAPTERS_STAGES and leaving a permanently "pending" row when unused.
  const currentStages = [
    ...baseStages,
    ...Object.keys(stageProgress).filter(name => !baseStages.includes(name)),
  ];
  // Derived from stageProgress so refetches never reset already-received
  // progress (and no setState-in-effect is needed).
  const stages: StageStatus[] = currentStages.map(name =>
    stageProgress[name] ?? { name, status: 'pending' as const }
  );

  useEffect(() => {
    const eventSource = new EventSource(withToken(`${API_BASE_URL}/jobs/${jobId}/progress`));

    eventSource.addEventListener('stage_started', (e) => {
      const data = JSON.parse(e.data);
      setStageProgress(prev => ({
        ...prev,
        [data.stage]: { name: data.stage, status: 'started', startTime: Date.now() },
      }));
    });

    eventSource.addEventListener('stage_completed', (e) => {
      const data = JSON.parse(e.data);
      setStageProgress(prev => ({
        ...prev,
        [data.stage]: { name: data.stage, status: 'completed', progress: 100, endTime: Date.now() },
      }));
    });

    eventSource.addEventListener('stage_failed', (e) => {
      const data = JSON.parse(e.data);
      setStageProgress(prev => ({
        ...prev,
        [data.stage]: { name: data.stage, status: 'failed', endTime: Date.now() },
      }));
    });

    eventSource.addEventListener('stage_skipped', (e) => {
      const data = JSON.parse(e.data);
      setStageProgress(prev => ({
        ...prev,
        [data.stage]: { name: data.stage, status: 'skipped' },
      }));
    });

    eventSource.addEventListener('job_completed', () => {
      eventSource.close();
    });

    eventSource.addEventListener('job_failed', () => {
      eventSource.close();
    });

    eventSource.addEventListener('job_cancelled', () => {
      eventSource.close();
    });

    return () => eventSource.close();
  }, [jobId, streamKey]);

  // The job's log: read again from the start when the page opens or the job
  // starts or ends, then only the new lines every few seconds while it runs.
  useEffect(() => {
    let cancelled = false;
    logOffset.current = 0;
    const load = async () => {
      const from = logOffset.current;
      try {
        const { data } = await api.get<{ lines: JobLogLine[]; offset: number }>(
          `/jobs/${jobId}/log`,
          { params: { offset: from } },
        );
        if (cancelled) return;
        logOffset.current = data.offset;
        const entries = data.lines.map(line => ({
          timestamp: line.timestamp,
          message: logMessage(line),
          level: line.level,
        }));
        // From the start: replace whatever an earlier read showed.
        if (from === 0) setLogs(entries);
        else if (entries.length > 0) setLogs(prev => [...prev, ...entries]);
      } catch {
        // No log yet, or the API restarts: the next poll tries again.
      }
    };
    load();
    if (!jobActive) {
      return () => { cancelled = true; };
    }
    const timer = setInterval(load, LOG_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [jobId, jobActive, streamKey]);

  // Keep the newest line in view inside the log box (not the whole page).
  useEffect(() => {
    const box = logBoxRef.current;
    if (box) box.scrollTop = box.scrollHeight;
  }, [logs]);

  const handleCancel = () => {
    if (window.confirm(`Cancel job #${jobId}? Finished stages are kept: Resume continues from them.`)) {
      cancelJob.mutate(Number(jobId));
    }
  };

  const handleResume = () => {
    retryJob.mutate(Number(jobId), {
      onSuccess: () => {
        setStageProgress({});
        setStreamKey((k) => k + 1);
      },
    });
  };

  const handleDelete = () => {
    if (window.confirm(`Delete job #${jobId} with its clips and files? This cannot be undone.`)) {
      deleteJob.mutate(Number(jobId), { onSuccess: () => navigate({ to: '/dashboard' }) });
    }
  };

  const handleCopyChapters = () => {
    if (chaptersTextData?.text) {
      navigator.clipboard.writeText(chaptersTextData.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleCopyBroll = () => {
    if (brollTextData?.text) {
      navigator.clipboard.writeText(brollTextData.text);
      setCopiedBroll(true);
      setTimeout(() => setCopiedBroll(false), 2000);
    }
  };

  const isReviewRouteActive = useRouterState({
    select: (s) => s.matches.some((m) => m.routeId === '/jobs/$jobId/review'),
  });

  if (isLoading) return <div className="p-6">Loading...</div>;
  if (!job) return <div className="p-6">Job not found</div>;
  if (isReviewRouteActive) return <Outlet />;

  return (
    <div className="container mx-auto p-6 space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              Job #{job.id}
              <Badge variant={
                job.status === 'completed' ? 'default' :
                job.status === 'failed' ? 'destructive' :
                job.status === 'running' ? 'outline' :
                'secondary'
              }>
                {job.status}
              </Badge>
              <Badge variant="outline">
                {job.job_type}
              </Badge>
            </CardTitle>
            <div className="flex items-center gap-2">
              {(job.status === 'queued' || job.status === 'running') && (
                <Button
                  variant="destructive"
                  onClick={handleCancel}
                  disabled={job.cancel_requested || cancelJob.isPending}
                >
                  {job.cancel_requested ? 'Stopping...' : 'Cancel job'}
                </Button>
              )}
              {(job.status === 'failed' || job.status === 'cancelled') && !job.sources_removed_at && (
                <Button variant="outline" onClick={handleResume} disabled={retryJob.isPending}>
                  Resume
                </Button>
              )}
              {job.status === 'completed' && job.job_type === 'clips' && (
                <Link to="/jobs/$jobId/review" params={{ jobId: String(job.id) }}>
                  <Button>Review Clips</Button>
                </Link>
              )}
              {(job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') && (
                <Button variant="outline" onClick={handleDelete} disabled={deleteJob.isPending}>
                  {deleteJob.isPending ? 'Deleting...' : 'Delete'}
                </Button>
              )}
              <Link to="/dashboard">
                <Button variant="outline">Back to Dashboard</Button>
              </Link>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-2 gap-4">
            <div>
              <dt className="font-medium">Input</dt>
              <dd className="text-muted-foreground">{job.input_source}</dd>
            </div>
            <div>
              <dt className="font-medium">Created</dt>
              <dd className="text-muted-foreground">{new Date(job.created_at).toLocaleString()}</dd>
            </div>
          </dl>
          {job.sources_removed_at && (
            <div role="status" className="mt-4 rounded-md border border-slate-300 bg-slate-50 p-3 text-sm text-slate-700">
              The source video was removed on {new Date(job.sources_removed_at).toLocaleDateString()} to
              free disk space (retention on the Settings page, Dashboard). The clips are kept; this job
              can no longer be resumed.
            </div>
          )}
          {deleteJob.isError && (
            <div role="alert" className="mt-4 rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-800">
              Could not delete the job: {String(deleteJob.error)}
            </div>
          )}
          {job.error_message && (
            <div role="alert" className="mt-4 rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-800">
              <p className="font-medium">Error</p>
              <p className="whitespace-pre-wrap break-words">{job.error_message}</p>
            </div>
          )}
          {notices && notices.length > 0 && (
            <div role="status" className="mt-4 rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 space-y-2">
              {notices.map((notice, i) => (
                <div key={`${notice.stage}-${notice.code}-${i}`}>
                  <p>
                    <span className="font-medium">{notice.stage}:</span> {notice.message}
                  </p>
                  {notice.error && (
                    <p className="text-xs text-amber-800 break-words">{notice.error}</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Pipeline Progress</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {stages.map((stage) => (
              <div key={stage.name} className="flex items-center gap-3">
                <div className="w-32 font-medium">{stage.name}</div>
                <Badge variant={
                  stage.status === 'completed' ? 'default' :
                  stage.status === 'failed' ? 'destructive' :
                  stage.status === 'started' || stage.status === 'running' ? 'outline' :
                  stage.status === 'skipped' ? 'secondary' :
                  'secondary'
                }>
                  {stage.status}
                </Badge>
                {stage.startTime && stage.status !== 'pending' && stage.status !== 'skipped' && (
                  <div className="text-sm text-muted-foreground">
                    {stage.endTime
                      ? `${((stage.endTime - stage.startTime) / 1000).toFixed(1)}s`
                      : 'running...'
                    }
                  </div>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {isChaptersJob && job.status === 'completed' && chaptersData?.chapters && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Chapters ({chaptersData.chapters.length})</CardTitle>
              <Button onClick={handleCopyChapters} variant="outline" size="sm">
                {copied ? 'Copied!' : 'Copy YouTube Format'}
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {chaptersData.chapters.map((ch: { timestamp: string; title: string; summary?: string }, i: number) => (
                <div key={i} className="flex gap-3 p-3 border rounded">
                  <div className="font-mono text-sm text-blue-600 w-16 shrink-0">{ch.timestamp}</div>
                  <div>
                    <div className="font-medium">{ch.title}</div>
                    {ch.summary && (
                      <div className="text-sm text-muted-foreground">{ch.summary}</div>
                    )}
                  </div>
                </div>
              ))}
            </div>
            {chaptersTextData?.text && (
              <div className="mt-4">
                <div className="text-sm font-medium mb-2">YouTube/VK Format:</div>
                <pre className="bg-gray-100 dark:bg-gray-800 p-3 rounded text-sm overflow-x-auto whitespace-pre-wrap">
                  {chaptersTextData.text}
                </pre>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {isChaptersJob && job.status === 'completed' && brollData?.suggestions && brollData.suggestions.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>B-roll Suggestions ({brollData.suggestions.length})</CardTitle>
              <Button onClick={handleCopyBroll} variant="outline" size="sm">
                {copiedBroll ? 'Copied!' : 'Copy as Text'}
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {brollData.suggestions.map((s: { timestamp: string; suggestion: string; keywords?: string[] }, i: number) => (
                <div key={i} className="flex gap-3 p-3 border rounded">
                  <div className="font-mono text-sm text-blue-600 w-16 shrink-0">{s.timestamp}</div>
                  <div>
                    <div className="font-medium">{s.suggestion}</div>
                    {s.keywords && s.keywords.length > 0 && (
                      <div className="text-sm text-muted-foreground">{s.keywords.join(', ')}</div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Logs</CardTitle>
        </CardHeader>
        <CardContent>
          <div ref={logBoxRef} className="bg-gray-900 text-gray-100 rounded p-4 h-64 overflow-y-auto font-mono text-sm">
            {logs.length === 0 ? (
              <div className="text-gray-500">No logs yet...</div>
            ) : (
              logs.map((log, i) => (
                <div key={i} className="mb-1 break-all">
                  <span className="text-gray-500">
                    {log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : ''}
                  </span>{' '}
                  <span className={
                    log.level === 'error' ? 'text-red-400' :
                    log.level === 'warning' ? 'text-yellow-400' :
                    'text-gray-100'
                  }>
                    {log.message}
                  </span>
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
