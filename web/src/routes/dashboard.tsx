import { createFileRoute, Link, useNavigate } from '@tanstack/react-router';
import { useState } from 'react';
import { useJobs } from '@/hooks/useJobs';
import { useShutdown, useSystemStatus, type ShutdownMode } from '@/hooks/useSystem';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

interface Job {
  id: number;
  status: string;
  input_source: string;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  job_type: string;
}

export const Route = createFileRoute('/dashboard')({
  component: DashboardPage,
});

const SHUTDOWN_CHOICES: { mode: ShutdownMode; label: string }[] = [
  { mode: 'requeue', label: 'Stop it now and continue from its last stage on the next start' },
  { mode: 'wait', label: 'Wait until it finishes, then shut down' },
  { mode: 'cancel', label: 'Cancel it' },
];

// "Shut down": stops the job worker, the API and the web UI.
function ShutdownButton() {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<ShutdownMode>('requeue');
  const { data: status } = useSystemStatus(open ? 3000 : false);
  const shutdown = useShutdown();
  const navigate = useNavigate();
  const job = status?.jobs.running;

  const confirm = async () => {
    await shutdown.mutateAsync(job ? mode : 'requeue');
    navigate({ to: '/shutdown' });
  };

  return (
    <>
      <Button variant="outline" onClick={() => setOpen(true)}>Shut down</Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Shut down?</DialogTitle>
            <DialogDescription>
              Stops the job worker, the API and this web interface and frees their memory. Clips and
              settings are kept; start again with <code>just up</code>.
            </DialogDescription>
          </DialogHeader>
          {job && (
            <div className="space-y-2 text-sm">
              <p>Job #{job.id} is running{job.stage ? ` (${job.stage})` : ''}:</p>
              {SHUTDOWN_CHOICES.map((choice) => (
                <label key={choice.mode} className="flex items-start gap-2">
                  <input
                    type="radio"
                    name="shutdown-mode"
                    checked={mode === choice.mode}
                    onChange={() => setMode(choice.mode)}
                    className="mt-1"
                  />
                  <span>{choice.label}</span>
                </label>
              ))}
            </div>
          )}
          {status && status.jobs.queued > 0 && (
            <p className="text-sm">{status.jobs.queued} queued job(s) stay queued until the next start.</p>
          )}
          {shutdown.isError && <p className="text-sm text-red-600">Could not shut down: {String(shutdown.error)}</p>}
          <DialogFooter>
            <Button variant="outline" onClick={() => setOpen(false)}>Cancel</Button>
            <Button variant="destructive" onClick={confirm} disabled={shutdown.isPending}>Shut down</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

function DashboardPage() {
  const { data, isLoading } = useJobs();

  if (isLoading) return <div className="p-6">Loading...</div>;

  const jobs: Job[] = (data?.jobs as Job[]) || [];
  const totalJobs = jobs.length;
  const completedJobs = jobs.filter((j) => j.status === 'completed').length;
  const successRate = totalJobs > 0 ? Math.round((completedJobs / totalJobs) * 100) : 0;
  
  const completedWithTime = jobs.filter((j) => 
    j.status === 'completed' && j.started_at && j.completed_at
  );
  const avgProcessingTime = completedWithTime.length > 0
    ? completedWithTime.reduce((acc: number, j) => {
        const start = new Date(j.started_at!).getTime();
        const end = new Date(j.completed_at!).getTime();
        return acc + (end - start) / 1000 / 60;
      }, 0) / completedWithTime.length
    : 0;

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Total Jobs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{totalJobs}</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{successRate}%</div>
            <p className="text-xs text-muted-foreground">{completedJobs} of {totalJobs} completed</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Avg Processing Time</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{avgProcessingTime.toFixed(1)}m</div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Jobs</CardTitle>
            <div className="flex gap-2">
              <ShutdownButton />
              <Link to="/diagnostics">
                <Button variant="outline">Diagnostics</Button>
              </Link>
              <Link to="/config">
                <Button variant="outline">Settings</Button>
              </Link>
              <Link to="/jobs/new">
                <Button>New Job</Button>
              </Link>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ID</TableHead>
                <TableHead>Input</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Created</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobs.map((job) => (
                <TableRow key={job.id}>
                  <TableCell>{job.id}</TableCell>
                  <TableCell className="max-w-xs truncate">{job.input_source}</TableCell>
                  <TableCell>
                    <Badge variant={
                      job.status === 'completed' ? 'default' :
                      job.status === 'failed' ? 'destructive' :
                      job.status === 'running' ? 'outline' :
                      'secondary'
                    }>
                      {job.status}
                    </Badge>
                  </TableCell>
                  <TableCell>{new Date(job.created_at).toLocaleString()}</TableCell>
                  <TableCell>
                    <Link to="/jobs/$jobId" params={{ jobId: String(job.id) }}>
                      <Button variant="outline" size="sm">View</Button>
                    </Link>
                  </TableCell>
                </TableRow>
              ))}
              {jobs.length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} className="text-center text-muted-foreground">
                    No jobs yet. Create your first job to get started.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
