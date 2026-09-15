"use client";

import { useMemo, useState } from "react";
import { FileAudio, Search } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { PageHeader } from "@/components/shared/page-header";
import { EmptyState } from "@/components/shared/empty-state";
import { ErrorState } from "@/components/shared/error-state";
import { Skeleton } from "@/components/ui/skeleton";
import { PredictionHistoryCards, PredictionHistoryTable } from "@/components/history/prediction-history-table";
import { useDeletePrediction, usePredictionHistory, useRerunPrediction } from "@/features/predictions/hooks";
import { getUserFriendlyErrorMessage } from "@/lib/api/errors";
import type { PredictionFilters } from "@/types/api";

export function HistoryPage() {
  const [filters, setFilters] = useState<PredictionFilters>({ page: 1, limit: 20 });
  const [search, setSearch] = useState("");
  const history = usePredictionHistory(filters);
  const deleteMutation = useDeletePrediction();
  const rerunMutation = useRerunPrediction();

  const visibleItems = useMemo(() => {
    const items = history.data?.items || [];
    const term = search.trim().toLowerCase();
    if (!term) return items;
    return items.filter((item) =>
      [item.filename, item.prediction_id, item.source_type, item.status, item.final_prediction]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(term)),
    );
  }, [history.data?.items, search]);

  const onDelete = async (id: string) => {
    try {
      await deleteMutation.mutateAsync(id);
      toast.success("Prediction deleted");
    } catch (error) {
      toast.error(getUserFriendlyErrorMessage(error));
    }
  };
  const onRerun = async (id: string) => {
    try {
      const response = await rerunMutation.mutateAsync({ predictionId: id });
      toast.success("Rerun started", { description: response.prediction_id });
    } catch (error) {
      toast.error(getUserFriendlyErrorMessage(error));
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader title="Prediction history" description="Owner-scoped prediction records returned by the backend. Filters use verified pagination parameters." />
      <Card>
        <CardContent className="grid gap-3 p-4 md:grid-cols-5">
          <div className="relative md:col-span-2">
            <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
            <Input className="pl-9" placeholder="Search visible page" value={search} onChange={(event) => setSearch(event.target.value)} />
          </div>
          <Select value={filters.status || "all"} onValueChange={(value) => setFilters((current) => ({ ...current, page: 1, status: value === "all" ? undefined : value }))}>
            <SelectTrigger><SelectValue placeholder="Status" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All statuses</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="failed">Failed</SelectItem>
              <SelectItem value="processing">Processing</SelectItem>
              <SelectItem value="queued">Queued</SelectItem>
            </SelectContent>
          </Select>
          <Select value={filters.sourceType || "all"} onValueChange={(value) => setFilters((current) => ({ ...current, page: 1, sourceType: value === "all" ? undefined : value }))}>
            <SelectTrigger><SelectValue placeholder="Source" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All sources</SelectItem>
              <SelectItem value="dashboard_upload">File upload</SelectItem>
              <SelectItem value="live_recording">Recording</SelectItem>
              <SelectItem value="public_api">Public API</SelectItem>
              <SelectItem value="mcp">MCP</SelectItem>
            </SelectContent>
          </Select>
          <Select value={filters.predictionLabel || "all"} onValueChange={(value) => setFilters((current) => ({ ...current, page: 1, predictionLabel: value === "all" ? undefined : value }))}>
            <SelectTrigger><SelectValue placeholder="Label" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All labels</SelectItem>
              <SelectItem value="spoof">Spoof</SelectItem>
              <SelectItem value="bonafide">Bonafide</SelectItem>
            </SelectContent>
          </Select>
        </CardContent>
      </Card>
      {history.isLoading ? <Skeleton className="h-96" /> : null}
      {history.isError ? <ErrorState error={history.error} /> : null}
      {!history.isLoading && !history.isError && !visibleItems.length ? (
        <EmptyState icon={FileAudio} title="No predictions found" description="Submit a new audio sample or change the current filters." action={{ href: "/dashboard/analyze", label: "Analyze audio" }} />
      ) : null}
      {visibleItems.length ? (
        <>
          <PredictionHistoryTable items={visibleItems} onDelete={onDelete} onRerun={onRerun} />
          <PredictionHistoryCards items={visibleItems} onDelete={onDelete} onRerun={onRerun} />
          <div className="flex items-center justify-between">
            <Button variant="outline" disabled={filters.page <= 1} onClick={() => setFilters((current) => ({ ...current, page: current.page - 1 }))}>Previous</Button>
            <p className="text-sm text-muted-foreground">Page {history.data?.page || filters.page}</p>
            <Button variant="outline" disabled={!history.data?.has_next} onClick={() => setFilters((current) => ({ ...current, page: current.page + 1 }))}>Next</Button>
          </div>
        </>
      ) : null}
    </div>
  );
}
