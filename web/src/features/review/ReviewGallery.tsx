import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function ReviewGallery() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Review Gallery</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-muted-foreground">No clips to review yet.</p>
      </CardContent>
    </Card>
  );
}
