import { createFileRoute } from '@tanstack/react-router';
import { ConfigEditor } from '@/features/config/ConfigEditor';

export const Route = createFileRoute('/config')({
  component: ConfigPage,
});

function ConfigPage() {
  return (
    <div className="container mx-auto p-6">
      <ConfigEditor />
    </div>
  );
}
