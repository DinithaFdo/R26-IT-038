"use client";

import { PageHeader } from "@/components/shared/page-header";
import { ErrorState } from "@/components/shared/error-state";
import { Skeleton } from "@/components/ui/skeleton";
import { SystemReadinessCard } from "@/components/status/system-readiness-card";
import { ModelHealthCard } from "@/components/status/model-health-card";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useModelHealth, useSystemReadiness } from "@/features/system-status/hooks";

export function SystemStatusPage() {
  const readiness = useSystemReadiness();
  const models = useModelHealth();

  return (
    <div className="space-y-6">
      <PageHeader
        title="System status"
        description="Backend readiness, model branch health, storage, MongoDB, and prediction runner state."
      />
      {readiness.isLoading ? <Skeleton className="h-48" /> : readiness.isError ? <ErrorState error={readiness.error} title="Readiness unavailable" /> : readiness.data ? <SystemReadinessCard readiness={readiness.data} /> : null}
      {models.isLoading ? <Skeleton className="h-72" /> : models.isError ? <ErrorState error={models.error} title="Model health unavailable" /> : models.data ? <ModelHealthCard models={models.data} /> : null}
      {readiness.data?.components ? (
        <Card>
          <CardHeader><CardTitle>Safe component details</CardTitle></CardHeader>
          <CardContent>
            <pre className="max-h-96 overflow-auto rounded-md bg-muted p-4 text-xs text-muted-foreground">
              {JSON.stringify(readiness.data.components, null, 2)}
            </pre>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
