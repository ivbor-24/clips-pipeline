import { createFileRoute, Link } from '@tanstack/react-router';
import { useSystemStatus } from '@/hooks/useSystem';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

export const Route = createFileRoute('/shutdown')({
  component: ShutdownPage,
});

const MODE_TEXT: Record<string, string> = {
  wait: 'Waiting for the running job to finish',
  cancel: 'Cancelling the running job',
  requeue: 'Stopping the running job: it continues from its last stage on the next start',
};

function Step({ done, active, children }: { done: boolean; active: boolean; children: React.ReactNode }) {
  return (
    <li className={done ? 'text-green-700' : active ? 'font-medium' : 'text-gray-400'}>
      {done ? '✓' : active ? '…' : '○'} {children}
    </li>
  );
}

// What happens after "Shut down", until the API is gone.
function ShutdownPage() {
  const { data, error, isError } = useSystemStatus(2000);
  // The API answered "shutting down" before and now does not answer: it is off.
  const off = isError && data?.state === 'shutting_down';

  if (!data && !isError) return <div className="p-6">Loading...</div>;
  if (!data) {
    return <div className="p-6 text-red-600">The system does not answer: {String(error)}</div>;
  }
  if (data.state !== 'shutting_down') {
    return (
      <div className="p-6 space-y-4 max-w-xl mx-auto">
        <p>The system is running; no shutdown was requested.</p>
        <Link to="/dashboard"><Button variant="outline">Back to Dashboard</Button></Link>
      </div>
    );
  }

  const shutdown = data.shutdown!;
  const job = data.jobs.running;
  const workerStopped = off || shutdown.worker_stopped;

  return (
    <div className="p-6 max-w-xl mx-auto">
      <Card>
        <CardHeader>
          <CardTitle>{off ? 'Shut down' : 'Shutting down...'}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          {shutdown.reason === 'idle' && (
            <p className="text-gray-500">
              Nothing was running and no page was open for {data.auto_shutdown_min} minutes.
            </p>
          )}
          <ul className="space-y-1">
            {job && shutdown.mode && (
              <Step done={workerStopped} active={!workerStopped}>
                {MODE_TEXT[shutdown.mode]} (job #{job.id}{job.stage ? `, ${job.stage}` : ''})
              </Step>
            )}
            <Step done={workerStopped} active={!workerStopped}>
              Stopping the job worker: models leave memory, temporary files are removed
            </Step>
            <Step done={off} active={workerStopped && !off}>Stopping the API and the web interface</Step>
          </ul>
          {data.jobs.queued > 0 && (
            <p>{data.jobs.queued} queued job(s) stay in the queue and run after the next start.</p>
          )}
          {off ? (
            <div className="rounded bg-green-50 p-3 text-green-800">
              Everything is stopped and its memory is freed. Clips and settings are kept. To start again run{' '}
              <code>just up</code> in the program's folder, then reload this page.
            </div>
          ) : (
            <p className="text-gray-500">Keep this page open to see when it is done.</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
