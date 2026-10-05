import { useState } from 'react';
import { useConfig, useProfiles, useResetConfig, useUpdateConfig } from '@/hooks/useConfig';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';

// The user settings (data/settings.yaml): only the keys changed on this page.
// "config" is the effective config, "defaults" the same without the user
// settings, "error" why the settings file was ignored (the banner below).
type Json = string | number | boolean | null | Json[] | { [key: string]: Json };
type Settings = { [key: string]: Json };

// "***" means a secret is configured but withheld by the backend (see
// src/api/routes/config.py::_redact_secrets) — never a real secret value.
const SECRET_PLACEHOLDER = '***';

interface FieldDef {
  label: string;
  path: string[];
  hint: string;
  kind: 'number' | 'text' | 'password' | 'select' | 'language';
  options?: { value: string; label: string }[];
  min?: number;
  max?: number;
  step?: number;
}

interface GroupDef {
  title: string;
  fields: FieldDef[];
}

const SUBTITLE_STYLES = ['typewriter', 'default', 'modern', 'minimal'].map((value) => ({
  value,
  label: value,
}));

const CLIPS_GROUP: GroupDef = {
  title: 'Clips',
  fields: [
    {
      label: 'Clips per video',
      path: ['scoring', 'max_clips_per_video'],
      hint: 'How many clip candidates to offer per video. You usually keep 5–7 in review.',
      kind: 'number',
      min: 1,
      max: 30,
    },
    {
      label: 'Min duration (s)',
      path: ['scoring', 'min_duration'],
      hint: 'Shortest allowed clip, in seconds.',
      kind: 'number',
      min: 1,
    },
    {
      label: 'Max duration (s)',
      path: ['scoring', 'max_duration'],
      hint: 'Longest allowed clip, in seconds (must be above the minimum).',
      kind: 'number',
      min: 1,
    },
    {
      label: 'Min score',
      path: ['scoring', 'min_score_threshold'],
      hint: 'Minimum clip score (0–1). Raise it to keep only the strongest moments.',
      kind: 'number',
      min: 0,
      max: 1,
      step: 0.05,
    },
  ],
};

const SPEECH_GROUP: GroupDef = {
  title: 'Speech',
  fields: [
    {
      label: 'Language',
      path: ['transcription', 'language'],
      hint: 'Language of the audio: auto (detect), ru, en, or another Whisper code.',
      kind: 'language',
    },
  ],
};

const SUBTITLES_GROUP: GroupDef = {
  title: 'Subtitles',
  fields: [
    {
      label: 'Style',
      path: ['rendering', 'subtitle_style'],
      hint: 'Look of the burned-in subtitles (typewriter is the recommended one).',
      kind: 'select',
      options: SUBTITLE_STYLES,
    },
  ],
};

const LLM_GROUP: GroupDef = {
  title: 'Analysis (LLM)',
  fields: [
    {
      label: 'Use a language model',
      path: ['scoring', 'llm', 'enabled'],
      hint: 'The LLM picks the clips. Off: simple heuristics, clearly worse clips.',
      kind: 'select',
      options: [
        { value: 'true', label: 'Yes' },
        { value: 'false', label: 'No' },
      ],
    },
    {
      label: 'Provider',
      path: ['scoring', 'llm', 'provider'],
      hint: 'llama_cpp runs the local model on your GPU; openai / anthropic need a key below.',
      kind: 'select',
      options: [
        { value: 'llama_cpp', label: 'llama_cpp (local model)' },
        { value: 'openai', label: 'openai (API)' },
        { value: 'anthropic', label: 'anthropic (API)' },
      ],
    },
  ],
};

// Shown only when the provider is an API one.
const LLM_API_FIELDS: FieldDef[] = [
  {
    label: 'Model',
    path: ['scoring', 'llm', 'model'],
    hint: 'The API model name, e.g. gpt-6-luna or claude-opus-5.',
    kind: 'text',
  },
  {
    label: 'API base (optional)',
    path: ['scoring', 'llm', 'api_base'],
    hint: 'openai only: the address of an OpenAI-compatible service (DeepSeek, OpenRouter, your own server); empty for OpenAI itself.',
    kind: 'text',
  },
  {
    label: 'API key',
    path: ['scoring', 'llm', 'api_key'],
    hint: 'Usually kept in .env (OPENAI_API_KEY / ANTHROPIC_API_KEY; ./setup.sh writes it there). A key entered here takes precedence.',
    kind: 'password',
  },
];

const STORAGE_GROUP: GroupDef = {
  title: 'Storage',
  fields: [
    {
      label: 'Keep source videos (days)',
      path: ['cleanup', 'retention_days'],
      hint: 'Days after a job finishes before its source video is deleted (clips and reviews stay); 0 — keep forever.',
      kind: 'number',
      min: 0,
    },
    {
      label: 'Warn above (GB)',
      path: ['cleanup', 'max_upload_size_gb'],
      hint: 'Only a warning in the log when uploads/ grows beyond this.',
      kind: 'number',
      min: 0,
    },
  ],
};

const HARDWARE_GROUP: GroupDef = {
  title: 'Hardware (advanced)',
  fields: [
    {
      label: 'Video encoder',
      path: ['rendering', 'video_encoder'],
      hint: 'How clips are encoded: auto — the GPU encoder if it works, else software; qsv / vaapi / nvenc — only that one (a job fails with the reason if it does not).',
      kind: 'select',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'software', label: 'software (CPU)' },
        { value: 'qsv', label: 'qsv (Intel Quick Sync)' },
        { value: 'vaapi', label: 'vaapi' },
        { value: 'nvenc', label: 'nvenc (NVIDIA)' },
      ],
    },
    {
      label: 'Frame decoder',
      path: ['cropping', 'frame_decoder'],
      hint: 'How video is decoded while faces are detected: auto — GPU if it works, else the CPU.',
      kind: 'select',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'software', label: 'software (CPU)' },
        { value: 'vaapi', label: 'vaapi' },
      ],
    },
  ],
};

function getAtPath(obj: Settings | undefined, path: string[]): Json | undefined {
  let node: Json | undefined = obj;
  for (const key of path) {
    if (node == null || typeof node !== 'object' || Array.isArray(node)) return undefined;
    node = (node as Settings)[key];
  }
  return node;
}

// An immutable copy of ``obj`` with ``value`` at ``path`` (empty objects on the way).
function setAtPath(obj: Settings, path: string[], value: Json): Settings {
  const [head, ...rest] = path;
  if (rest.length === 0) return { ...obj, [head]: value };
  const child = (typeof obj[head] === 'object' && obj[head] !== null && !Array.isArray(obj[head])
    ? { ...(obj[head] as Settings) }
    : {}) as Settings;
  return { ...obj, [head]: setAtPath(child, rest, value) };
}

// An immutable copy of ``obj`` without ``path`` (sections left empty are dropped).
function removeAtPath(obj: Settings, path: string[]): Settings {
  if (path.length === 0) return {};
  const [head, ...rest] = path;
  if (!(head in obj)) return obj;
  const value = obj[head];
  if (rest.length && (value == null || typeof value !== 'object' || Array.isArray(value))) {
    return obj;
  }
  const child = rest.length ? removeAtPath(value as Settings, rest) : undefined;
  const next = { ...obj };
  if (child && Object.keys(child).length > 0) next[head] = child;
  else delete next[head];
  return next;
}

function deepEqual(a: Json | undefined, b: Json | undefined): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b || a == null || b == null) return false;
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
    return a.every((item, i) => deepEqual(item, b[i]));
  }
  if (typeof a === 'object' && typeof b === 'object') {
    const ka = Object.keys(a as Settings);
    const kb = Object.keys(b as Settings);
    if (ka.length !== kb.length) return false;
    return ka.every((k) =>
      deepEqual((a as Settings)[k], (b as Settings)[k as keyof typeof b])
    );
  }
  return false;
}

function leafPaths(obj: Settings, base: string[] = []): string[][] {
  const paths: string[][] = [];
  for (const [key, value] of Object.entries(obj)) {
    const path = [...base, key];
    if (value != null && typeof value === 'object' && !Array.isArray(value)) {
      paths.push(...leafPaths(value as Settings, path));
    } else {
      paths.push(path);
    }
  }
  return paths;
}

// Only what differs from the saved user settings: new and changed values, and
// the default value for a key this page reset (the server drops it again).
function buildUpdates(draft: Settings, overrides: Settings, defaults: Settings): Settings | null {
  const seen = new Map<string, string[]>();
  for (const path of [...leafPaths(draft), ...leafPaths(overrides)]) {
    seen.set(path.join('\u0000'), path);
  }
  let updates: Settings = {};
  for (const path of seen.values()) {
    const draftValue = getAtPath(draft, path);
    const savedValue = getAtPath(overrides, path);
    if (deepEqual(draftValue, savedValue)) continue;
    const value = draftValue !== undefined ? draftValue : getAtPath(defaults, path) ?? null;
    updates = setAtPath(updates, path, value as Json);
  }
  return Object.keys(updates).length > 0 ? updates : null;
}

// The select element works with strings; booleans need converting back.
function selectValue(value: Json | undefined, options: { value: string }[]): string | undefined {
  const asString =
    value === true ? 'true' : value === false ? 'false' : value == null ? undefined : String(value);
  if (asString !== undefined && !options.some((o) => o.value === asString)) {
    return asString; // a valid value the list does not know: keep it selectable
  }
  return asString;
}

function numberError(def: FieldDef, value: Json | undefined): string | null {
  if (value == null || value === '') return 'Enter a number.';
  const n = Number(value);
  if (Number.isNaN(n)) return 'Enter a number.';
  if (def.min != null && n < def.min) return `Must be ${def.min} or more.`;
  if (def.max != null && n > def.max) return `Must be ${def.max} or less.`;
  return null;
}

const selectClass =
  'w-full border rounded-lg px-3 py-2 bg-transparent text-sm h-8';

function FieldRow({
  def,
  draft,
  defaults,
  onChange,
  onReset,
  error,
}: {
  def: FieldDef;
  draft: Settings;
  defaults: Settings;
  onChange: (path: string[], value: Json) => void;
  onReset: (path: string[]) => void;
  error: string | null;
}) {
  const draftValue = getAtPath(draft, def.path);
  const value: Json | undefined =
    draftValue !== undefined ? draftValue : getAtPath(defaults, def.path);
  const changed = draftValue !== undefined && !deepEqual(draftValue, getAtPath(defaults, def.path));
  const selectOptions = def.kind === 'select' ? (def.options ?? []) : undefined;

  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <label className="text-sm font-medium">
          {def.label}
          {changed && (
            <Badge variant="secondary" className="ml-2 align-middle">changed</Badge>
          )}
        </label>
        {changed && (
          <Button variant="ghost" size="xs" onClick={() => onReset(def.path)}>
            Reset
          </Button>
        )}
      </div>
      {def.kind === 'select' && selectOptions && (
        <select
          className={selectClass}
          value={selectValue(value, selectOptions) ?? ''}
          onChange={(e) => {
            const raw = e.target.value;
            const parsed =
              raw === 'true' ? true : raw === 'false' ? false : raw;
            onChange(def.path, parsed as Json);
          }}
        >
          {selectValue(value, selectOptions) !== undefined &&
            !selectOptions.some((o) => o.value === selectValue(value, selectOptions)) && (
              <option value={String(value)}>{String(value)}</option>
            )}
          {selectOptions.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
      )}
      {def.kind === 'language' && (
        <>
          <Input
            list="whisper-languages"
            value={value == null ? '' : String(value)}
            onChange={(e) => onChange(def.path, e.target.value === '' ? 'auto' : e.target.value)}
          />
          <datalist id="whisper-languages">
            <option value="auto" />
            <option value="ru" />
            <option value="en" />
            <option value="de" />
            <option value="fr" />
            <option value="es" />
            <option value="it" />
            <option value="pl" />
            <option value="uk" />
          </datalist>
        </>
      )}
      {(def.kind === 'number' || def.kind === 'text') && (
        <Input
          type={def.kind === 'number' ? 'number' : 'text'}
          step={def.step}
          value={value == null ? '' : String(value)}
          onChange={(e) => {
            const raw = e.target.value;
            if (def.kind === 'number') {
              onChange(def.path, raw === '' ? null : Number(raw));
            } else {
              onChange(def.path, raw);
            }
          }}
        />
      )}
      {def.kind === 'password' && (
        <Input
          type="password"
          placeholder={value === SECRET_PLACEHOLDER ? '•••• (set — leave blank to keep)' : ''}
          value={value === SECRET_PLACEHOLDER ? '' : value == null ? '' : String(value)}
          onChange={(e) => onChange(def.path, e.target.value === '' ? null : e.target.value)}
        />
      )}
      <p className="text-xs text-muted-foreground mt-1">{def.hint}</p>
      {error && <p className="text-xs text-red-600 mt-1">{error}</p>}
    </div>
  );
}

export function ConfigEditor() {
  const { data: configData, isLoading: configLoading, error: loadError } = useConfig();
  const { data: profilesData } = useProfiles();
  const updateConfig = useUpdateConfig();
  const resetConfig = useResetConfig();
  const [draft, setDraft] = useState<Settings | null>(null);
  const [loadedOverrides, setLoadedOverrides] = useState<unknown>(null);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [serverError, setServerError] = useState<string | null>(null);
  const [jsonText, setJsonText] = useState('');
  const [jsonDirty, setJsonDirty] = useState(false);
  const [jsonError, setJsonError] = useState<string | null>(null);

  // Initialize (and reset after save/reset, when the server data changes) the
  // local draft from the saved user settings. Render-time adjustment, not
  // setState-in-effect; TanStack Query's structural sharing keeps the object
  // identity (and so the draft) while the user edits.
  if (configData?.overrides && loadedOverrides !== configData.overrides) {
    setLoadedOverrides(configData.overrides);
    setDraft(JSON.parse(JSON.stringify(configData.overrides)) as Settings);
    setJsonDirty(false);
    setJsonError(null);
    setServerError(null);
  }
  if (!jsonDirty && configData?.overrides) {
    const text = JSON.stringify(draft ?? {}, null, 2);
    if (text !== jsonText) setJsonText(text);
  }

  const defaults: Settings = (configData?.defaults as Settings) ?? {};
  const overrides: Settings = (configData?.overrides as Settings) ?? {};

  const handleFieldChange = (path: string[], value: Json) => {
    setDraft((prev) => setAtPath(prev ?? {}, path, value));
    setSaveStatus('idle');
    setServerError(null);
  };

  const handleFieldReset = (path: string[]) => {
    setDraft((prev) => removeAtPath(prev ?? {}, path));
    setSaveStatus('idle');
    setServerError(null);
  };

  const applyJson = (text: string) => {
    try {
      const parsed = JSON.parse(text);
      if (parsed == null || typeof parsed !== 'object' || Array.isArray(parsed)) {
        setJsonError('The settings must be a JSON object, e.g. {"scoring": {"max_clips_per_video": 8}}');
        return;
      }
      setJsonError(null);
      setDraft(parsed as Settings);
      setSaveStatus('idle');
      setServerError(null);
    } catch (e) {
      setJsonError(`Not valid JSON: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  // Cross-field rules the server also checks, shown next to the fields.
  const fieldError = (def: FieldDef): string | null => {
    if (def.kind !== 'number') return null;
    const draftValue = getAtPath(draft ?? {}, def.path);
    return numberError(def, draftValue !== undefined ? draftValue : getAtPath(defaults, def.path));
  };
  const minDuration = getAtPath(draft ?? {}, ['scoring', 'min_duration']) ?? getAtPath(defaults, ['scoring', 'min_duration']);
  const maxDuration = getAtPath(draft ?? {}, ['scoring', 'max_duration']) ?? getAtPath(defaults, ['scoring', 'max_duration']);
  const durationError =
    typeof minDuration === 'number' && typeof maxDuration === 'number' && minDuration >= maxDuration
      ? 'The minimum duration must be below the maximum.'
      : null;

  const allFieldErrors = [durationError].filter(Boolean) as string[];
  const disableSave =
    jsonError !== null || allFieldErrors.length > 0 ||
    [CLIPS_GROUP, SPEECH_GROUP, SUBTITLES_GROUP, LLM_GROUP, STORAGE_GROUP, HARDWARE_GROUP]
      .flatMap((g) => g.fields)
      .some((f) => fieldError(f) !== null);

  const handleSave = () => {
    if (!draft) return;
    const updates = buildUpdates(draft, overrides, defaults);
    if (!updates) {
      setSaveStatus('saved');
      return;
    }
    setSaveStatus('saving');
    setServerError(null);
    updateConfig.mutate(
      { updates },
      {
        onSuccess: () => {
          setSaveStatus('saved');
          setJsonDirty(false);
        },
        onError: (error) => {
          setSaveStatus('error');
          const detail = (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
          setServerError(detail ?? 'Saving failed; the settings were not changed.');
        },
      }
    );
  };

  const handleResetAll = () => {
    if (!window.confirm('Reset every setting to its default? This cannot be undone.')) return;
    setSaveStatus('saving');
    setServerError(null);
    resetConfig.mutate([], {
      onSuccess: () => setSaveStatus('idle'),
      onError: (error) => {
        setSaveStatus('error');
        const detail = (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
        setServerError(detail ?? 'Resetting failed.');
      },
    });
  };

  if (configLoading || !draft) return <div className="p-6">Loading settings...</div>;

  // A field reset or edit only changes the draft until Save.
  const unsaved = buildUpdates(draft, overrides, defaults) !== null;

  const profiles: string[] = profilesData?.profiles || [];
  const llmProvider = String(
    getAtPath(draft, ['scoring', 'llm', 'provider']) ?? getAtPath(defaults, ['scoring', 'llm', 'provider']) ?? 'llama_cpp'
  );
  const isApiProvider = llmProvider === 'openai' || llmProvider === 'anthropic';

  const groups: { def: GroupDef; collapsed?: boolean }[] = [
    { def: CLIPS_GROUP },
    { def: SPEECH_GROUP },
    { def: SUBTITLES_GROUP },
    { def: LLM_GROUP },
    { def: STORAGE_GROUP },
    { def: HARDWARE_GROUP, collapsed: true },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Settings</h1>
          <p className="text-sm text-muted-foreground">
            Saved to data/settings.yaml; config/config.yaml stays as shipped. Values equal to
            the defaults are not stored.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {unsaved && saveStatus !== 'saving' && (
            <span className="text-sm text-amber-600">Unsaved changes</span>
          )}
          {saveStatus === 'saved' && !unsaved && <span className="text-sm text-green-600">Saved</span>}
          {saveStatus === 'error' && <span className="text-sm text-red-600">Save failed</span>}
          <Button variant="outline" onClick={handleResetAll} disabled={resetConfig.isPending}>
            {resetConfig.isPending ? 'Resetting...' : 'Reset all'}
          </Button>
          <Button onClick={handleSave} disabled={saveStatus === 'saving' || disableSave}>
            {saveStatus === 'saving' ? 'Saving...' : 'Save Changes'}
          </Button>
        </div>
      </div>

      {(loadError as Error | null) && (
        <div className="rounded-lg border border-red-500/50 bg-red-500/10 p-4 text-sm">
          Could not load the settings: {(loadError as Error).message}
        </div>
      )}
      {configData?.error && (
        <div className="rounded-lg border border-yellow-500/50 bg-yellow-500/10 p-4 text-sm">
          The settings file could not be used and was ignored, so these are the defaults.
          Saving will replace the broken file. Reason: {String(configData.error)}
        </div>
      )}
      {serverError && (
        <div className="rounded-lg border border-red-500/50 bg-red-500/10 p-4 text-sm">
          {serverError}
        </div>
      )}

      {groups.map(({ def, collapsed }) => (
        <Card key={def.title}>
          {collapsed ? (
            <details>
              <summary className="cursor-pointer select-none px-4 py-4 text-base font-medium">
                {def.title}
              </summary>
              <CardContent className="space-y-4">
                <p className="text-xs text-muted-foreground">
                  Advanced: change only if something does not work on your machine.
                </p>
                {def.fields.map((field) => (
                  <div key={field.path.join('.')}>
                    <FieldRow
                      def={field}
                      draft={draft}
                      defaults={defaults}
                      onChange={handleFieldChange}
                      onReset={handleFieldReset}
                      error={fieldError(field)}
                    />
                  </div>
                ))}
              </CardContent>
            </details>
          ) : (
            <>
              <CardHeader>
                <CardTitle>{def.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {def.fields.map((field) => (
                    <div key={field.path.join('.')}>
                      <FieldRow
                        def={field}
                        draft={draft}
                        defaults={defaults}
                        onChange={handleFieldChange}
                        onReset={handleFieldReset}
                        error={fieldError(field)}
                      />
                      {def === CLIPS_GROUP && field.path[1] === 'max_duration' && durationError && (
                        <p className="text-xs text-red-600 mt-1">{durationError}</p>
                      )}
                    </div>
                  ))}
                  {def === LLM_GROUP && isApiProvider && (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:col-span-2">
                      {LLM_API_FIELDS.map((field) => (
                        <FieldRow
                          key={field.path.join('.')}
                          def={field}
                          draft={draft}
                          defaults={defaults}
                          onChange={handleFieldChange}
                          onReset={handleFieldReset}
                          error={null}
                        />
                      ))}
                    </div>
                  )}
                </div>
              </CardContent>
            </>
          )}
        </Card>
      ))}

      <Card>
        <details>
          <summary className="cursor-pointer select-none px-4 py-4 text-base font-medium">
            Advanced (all settings as JSON)
          </summary>
          <CardContent className="space-y-2">
            <p className="text-xs text-muted-foreground">
              Every changed setting as JSON, for keys the form above does not show (see
              docs/CONFIG.md for what exists). Validated on save.
            </p>
            <textarea
              className="w-full rounded-lg border bg-gray-900 text-gray-100 p-4 font-mono text-sm min-h-48"
              spellCheck={false}
              value={jsonText}
              onChange={(e) => {
                setJsonText(e.target.value);
                setJsonDirty(true);
                setSaveStatus('idle');
              }}
              onBlur={() => {
                applyJson(jsonText);
                setJsonDirty(false);
              }}
            />
            {jsonError && <p className="text-xs text-red-600">{jsonError}</p>}
          </CardContent>
        </details>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Profiles</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground">
            Available profiles (applied when creating a job):
            {profiles.length > 0 ? ` ${profiles.join(', ')}` : ' none'}
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
