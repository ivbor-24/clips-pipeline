import { createFileRoute, Link } from '@tanstack/react-router';
import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api, API_BASE_URL, withToken } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

export const Route = createFileRoute('/jobs/$jobId/review')({
  component: ReviewPage,
});

interface Clip {
  id: string;
  video_path: string;
  srt_path: string | null;
  score: number;
  duration: number;
  tags: string[];
  review_status: 'pending' | 'keep' | 'reject' | 'edit';
  review_notes: string | null;
  metadata: Record<string, unknown>;
}

function ReviewPage() {
  const { jobId } = Route.useParams();
  const queryClient = useQueryClient();
  const [currentIndex, setCurrentIndex] = useState(0);
  // The notes editor: its own state per clip, saved on blur or with the button.
  const [notes, setNotes] = useState('');
  const [notesClipId, setNotesClipId] = useState<string | null>(null);
  const [notesStatus, setNotesStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');

  const { data, isLoading } = useQuery({
    queryKey: ['clips', jobId],
    queryFn: async () => {
      const res = await api.get(`/jobs/${jobId}/clips`);
      return res.data;
    },
  });

  const clips: Clip[] = data?.clips || [];
  const currentClip = clips[currentIndex];

  const updateClip = useMutation({
    mutationFn: async ({ clipId, status }: { clipId: string; status: string }) => {
      await api.put(`/jobs/${jobId}/clips/${clipId}`, { review_status: status });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['clips', jobId] });
      if (currentIndex < clips.length - 1) {
        setCurrentIndex(currentIndex + 1);
      }
    },
  });

  // Fill the notes editor when the clip changes (render-time adjustment, not
  // setState-in-effect). notesStatus is left alone: "saved" stays visible.
  if (currentClip && notesClipId !== currentClip.id) {
    setNotesClipId(currentClip.id);
    setNotes(currentClip.review_notes ?? '');
    setNotesStatus('idle');
  }

  const saveNotes = useMutation({
    mutationFn: async ({ clipId, value }: { clipId: string; value: string }) => {
      await api.put(`/jobs/${jobId}/clips/${clipId}`, { notes: value });
    },
    onSuccess: () => {
      setNotesStatus('saved');
      queryClient.invalidateQueries({ queryKey: ['clips', jobId] });
    },
    onError: () => setNotesStatus('error'),
  });

  const handleNotesSave = (reason: 'blur' | 'button') => {
    if (!currentClip || notesStatus === 'saving') return;
    // On blur, save only what changed (avoid double saves on clip switches).
    const stored = currentClip.review_notes ?? '';
    if (reason === 'blur' && notes === stored) return;
    if (reason === 'blur' && notesClipId !== currentClip.id) return;
    setNotesStatus('saving');
    saveNotes.mutate({ clipId: currentClip.id, value: notes });
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      if (e.key === 'k' || e.key === 'K') {
        if (currentClip) updateClip.mutate({ clipId: currentClip.id, status: 'keep' });
      } else if (e.key === 'r' || e.key === 'R') {
        if (currentClip) updateClip.mutate({ clipId: currentClip.id, status: 'reject' });
      } else if (e.key === 'e' || e.key === 'E') {
        if (currentClip) updateClip.mutate({ clipId: currentClip.id, status: 'edit' });
      } else if (e.key === 'ArrowLeft' && currentIndex > 0) {
        setCurrentIndex(currentIndex - 1);
      } else if (e.key === 'ArrowRight' && currentIndex < clips.length - 1) {
        setCurrentIndex(currentIndex + 1);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentIndex, currentClip, clips.length, updateClip]);

  if (isLoading) return <div className="p-6">Loading clips...</div>;
  if (clips.length === 0) return <div className="p-6">No clips to review</div>;

  const videoUrl = withToken(`${API_BASE_URL}/jobs/${jobId}/clips/${currentClip.id}/video`);
  const downloadUrl = withToken(
    `${API_BASE_URL}/jobs/${jobId}/clips/${currentClip.id}/video?download=1`
  );
  const subtitlesUrl = withToken(
    `${API_BASE_URL}/jobs/${jobId}/clips/${currentClip.id}/subtitles`
  );

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Review Clips</h1>
        <div className="flex items-center gap-2">
          <Badge>{currentIndex + 1} / {clips.length}</Badge>
          <Link to="/jobs/$jobId" params={{ jobId }}>
            <Button variant="outline">Back to Job</Button>
          </Link>
        </div>
      </div>

      <Card>
        <CardContent className="p-0 flex justify-center bg-black rounded-lg overflow-hidden">
          {/* A 9:16 box, height-limited: a vertical clip is not letterboxed into 16:9. */}
          <video
            key={currentClip.id}
            controls
            autoPlay
            className="aspect-[9/16] h-[70vh] max-w-full object-contain bg-black"
          >
            <source src={videoUrl} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {currentClip.id}
            <Badge variant={
              currentClip.review_status === 'keep' ? 'default' :
              currentClip.review_status === 'reject' ? 'destructive' :
              currentClip.review_status === 'edit' ? 'outline' :
              'secondary'
            }>
              {currentClip.review_status}
            </Badge>
            <div className="flex gap-2 ml-auto">
              <a href={downloadUrl}>
                <Button variant="outline" size="sm">Download MP4</Button>
              </a>
              {currentClip.srt_path && (
                <a href={subtitlesUrl}>
                  <Button variant="outline" size="sm">Subtitles (SRT)</Button>
                </a>
              )}
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-2 gap-4">
            <div>
              <dt className="font-medium">Score</dt>
              <dd className="text-muted-foreground">{currentClip.score.toFixed(2)}</dd>
            </div>
            <div>
              <dt className="font-medium">Duration</dt>
              <dd className="text-muted-foreground">{currentClip.duration.toFixed(1)}s</dd>
            </div>
            <div className="col-span-2">
              <dt className="font-medium">Tags</dt>
              <dd className="flex flex-wrap gap-1 mt-1">
                {currentClip.tags.map((tag) => (
                  <Badge key={tag} variant="outline">{tag}</Badge>
                ))}
              </dd>
            </div>
          </dl>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Notes</CardTitle>
        </CardHeader>
        <CardContent>
          <textarea
            className="w-full min-h-20 rounded-lg border border-input bg-transparent p-2 text-sm outline-none focus-visible:border-ring"
            placeholder="What to change in this clip, or why you keep it..."
            value={notes}
            onChange={(e) => {
              setNotes(e.target.value);
              setNotesStatus('idle');
            }}
            onBlur={() => handleNotesSave('blur')}
          />
          <div className="flex items-center gap-2 mt-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleNotesSave('button')}
              disabled={notesStatus === 'saving'}
            >
              {notesStatus === 'saving' ? 'Saving...' : 'Save note'}
            </Button>
            {notesStatus === 'saved' && (
              <span className="text-sm text-green-600">Saved</span>
            )}
            {notesStatus === 'error' && (
              <span className="text-sm text-red-600">Could not save the note; try again.</span>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Review Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <Button
              onClick={() => updateClip.mutate({ clipId: currentClip.id, status: 'keep' })}
              disabled={updateClip.isPending}
            >
              Keep (K)
            </Button>
            <Button
              variant="destructive"
              onClick={() => updateClip.mutate({ clipId: currentClip.id, status: 'reject' })}
              disabled={updateClip.isPending}
            >
              Reject (R)
            </Button>
            <Button
              variant="outline"
              onClick={() => updateClip.mutate({ clipId: currentClip.id, status: 'edit' })}
              disabled={updateClip.isPending}
            >
              Edit (E)
            </Button>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            Keyboard shortcuts: K = Keep, R = Reject, E = Edit, ← = Previous, → = Next
          </p>
        </CardContent>
      </Card>

      <div className="flex justify-between">
        <Button
          variant="outline"
          onClick={() => setCurrentIndex(currentIndex - 1)}
          disabled={currentIndex === 0}
        >
          Previous
        </Button>
        <Button
          variant="outline"
          onClick={() => setCurrentIndex(currentIndex + 1)}
          disabled={currentIndex === clips.length - 1}
        >
          Next
        </Button>
      </div>
    </div>
  );
}
