import { createFileRoute, useNavigate } from '@tanstack/react-router';
import { useState } from 'react';
import { isAxiosError } from 'axios';
import { useCreateJob } from '@/hooks/useJobs';
import { useUpload } from '@/hooks/useUpload';
import { useProfiles } from '@/hooks/useConfig';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';

export const Route = createFileRoute('/jobs/new')({
  component: NewJobPage,
});

function NewJobPage() {
  const [inputSource, setInputSource] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [profile, setProfile] = useState('');
  const [jobType, setJobType] = useState<'clips' | 'chapters'>('clips');
  const [minMinutes, setMinMinutes] = useState(5);
  const [maxMinutes, setMaxMinutes] = useState(7);
  const [chapterCount, setChapterCount] = useState('');
  const [brollEnabled, setBrollEnabled] = useState(false);
  const [brollCount, setBrollCount] = useState('');
  const [configOverrides, setConfigOverrides] = useState('{}');
  const [uploadPercent, setUploadPercent] = useState(0);
  const createJob = useCreateJob();
  const uploadFile = useUpload();
  const navigate = useNavigate();

  const { data: profilesData } = useProfiles();
  const profiles = profilesData?.profiles || [];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      let overrides: Record<string, unknown> = {};
      if (configOverrides.trim()) {
        overrides = JSON.parse(configOverrides);
      }
      if (profile) {
        overrides.profile = profile;
      }
      if (jobType === 'chapters') {
        overrides.chapters = {
          min_minutes: minMinutes,
          max_minutes: maxMinutes,
          enabled: true,
          ...(chapterCount.trim() ? { target_count: Number(chapterCount) } : {}),
        };
        if (brollEnabled) {
          overrides.broll = {
            enabled: true,
            ...(brollCount.trim() ? { max_suggestions: Number(brollCount) } : {}),
          };
        }
      }

      let finalInputSource = inputSource;

      if (selectedFile) {
        setUploadPercent(0);
        const uploadResult = await uploadFile.mutateAsync({
          file: selectedFile,
          onProgress: setUploadPercent,
        });
        finalInputSource = uploadResult.path;
      }

      const job = await createJob.mutateAsync({
        input_source: finalInputSource,
        job_type: jobType,
        config_overrides: Object.keys(overrides).length > 0 ? overrides : undefined,
      });
      navigate({ to: '/jobs/$jobId', params: { jobId: String(job.id) } });
    } catch (err) {
      console.error('Failed to create job:', err);
      // The API explains refusals (file too big, disk full, wrong type).
      const detail = isAxiosError(err) ? err.response?.data?.detail : undefined;
      alert(typeof detail === 'string' ? detail : 'Failed to create job. Check console for details.');
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setInputSource('');
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputSource(e.target.value);
    if (e.target.value) {
      setSelectedFile(null);
    }
  };

  const isUploading = uploadFile.isPending;
  const isSubmitting = createJob.isPending || isUploading;
  const canSubmit = (inputSource || selectedFile) && !isSubmitting;

  return (
    <div className="container mx-auto p-6">
      <Card className="max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle>Create New Job</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="text-sm font-medium">Job Type</label>
              <div className="flex gap-2 mt-1">
                <Button
                  type="button"
                  variant={jobType === 'clips' ? 'default' : 'outline'}
                  onClick={() => setJobType('clips')}
                >
                  Video Clips
                </Button>
                <Button
                  type="button"
                  variant={jobType === 'chapters' ? 'default' : 'outline'}
                  onClick={() => setJobType('chapters')}
                >
                  Draft (Chapters &amp; B-roll)
                </Button>
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {jobType === 'clips'
                  ? 'Generate short vertical video clips from long content (final output)'
                  : 'Plan before producing clips: chapter outline with timestamps and optional B-roll ideas — no clips are rendered'}
              </p>
            </div>

            <div>
              <label className="text-sm font-medium">Input Source</label>
              <Input
                placeholder="/path/to/video.mp4 or https://youtube.com/watch?v=..."
                value={inputSource}
                onChange={handleInputChange}
                disabled={isSubmitting}
              />
              <p className="text-xs text-muted-foreground mt-1">
                Enter a local file path or YouTube URL
              </p>
            </div>

            <div>
              <label className="text-sm font-medium">Or Upload File</label>
              <Input
                type="file"
                accept="video/*"
                onChange={handleFileChange}
                disabled={isSubmitting}
              />
              {selectedFile && (
                <p className="text-xs text-green-600 mt-1">
                  Selected: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(1)} MB)
                </p>
              )}
              <p className="text-xs text-muted-foreground mt-1">
                Upload a video file (mp4, mkv, avi, mov, webm)
              </p>
            </div>

            {jobType === 'chapters' && (
              <>
                <div className="flex gap-4">
                  <div className="flex-1">
                    <label className="text-sm font-medium">Min Minutes</label>
                    <Input
                      type="number"
                      min={1}
                      max={30}
                      value={minMinutes}
                      onChange={(e) => setMinMinutes(Number(e.target.value))}
                      disabled={isSubmitting}
                    />
                  </div>
                  <div className="flex-1">
                    <label className="text-sm font-medium">Max Minutes</label>
                    <Input
                      type="number"
                      min={1}
                      max={30}
                      value={maxMinutes}
                      onChange={(e) => setMaxMinutes(Number(e.target.value))}
                      disabled={isSubmitting}
                    />
                  </div>
                  <div className="flex-1">
                    <label
                      className="text-sm font-medium"
                      title="Exact number of chapters to produce. Overrides Min/Max Minutes above. Leave empty to let the model decide."
                    >
                      Chapter count (optional)
                    </label>
                    <Input
                      type="number"
                      min={1}
                      max={50}
                      placeholder="model decides"
                      value={chapterCount}
                      onChange={(e) => setChapterCount(e.target.value)}
                      disabled={isSubmitting}
                    />
                  </div>
                </div>

                <div className="border rounded p-3 space-y-2">
                  <label
                    className="flex items-center gap-2 text-sm font-medium cursor-pointer"
                    title="Adds timestamped B-roll footage ideas as a text artifact alongside the chapter outline. This only suggests ideas for an editor — no footage is searched, downloaded, or inserted automatically."
                  >
                    <input
                      type="checkbox"
                      checked={brollEnabled}
                      onChange={(e) => setBrollEnabled(e.target.checked)}
                      disabled={isSubmitting}
                    />
                    Generate B-roll suggestions
                  </label>
                  {brollEnabled && (
                    <div>
                      <label
                        className="text-sm font-medium"
                        title="Maximum number of B-roll suggestions. Leave empty to let the model decide based on the video's content."
                      >
                        Suggestion count (optional)
                      </label>
                      <Input
                        type="number"
                        min={1}
                        max={100}
                        placeholder="model decides"
                        value={brollCount}
                        onChange={(e) => setBrollCount(e.target.value)}
                        disabled={isSubmitting}
                      />
                    </div>
                  )}
                </div>
              </>
            )}

            <div>
              <label className="text-sm font-medium">Profile (optional)</label>
              <select
                className="w-full border rounded px-3 py-2"
                value={profile}
                onChange={(e) => setProfile(e.target.value)}
                disabled={isSubmitting}
              >
                <option value="">Default</option>
                {profiles.map((p: string) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
              <p className="text-xs text-muted-foreground mt-1">
                Select a configuration profile (e.g., low_vram for 8GB GPUs)
              </p>
            </div>

            <div>
              <label className="text-sm font-medium">Config Overrides (optional)</label>
              <textarea
                className="w-full border rounded px-3 py-2 font-mono text-sm"
                rows={6}
                placeholder='{"transcription": {"model": "small"}}'
                value={configOverrides}
                onChange={(e) => setConfigOverrides(e.target.value)}
                disabled={isSubmitting}
              />
              <p className="text-xs text-muted-foreground mt-1">
                JSON object with config overrides (advanced)
              </p>
            </div>

            <Button type="submit" disabled={!canSubmit}>
              {isUploading ? `Uploading ${uploadPercent}%` : createJob.isPending ? 'Creating...' : 'Create Job'}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
