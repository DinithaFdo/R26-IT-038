"use client";

import Link from "next/link";
import { RotateCcw, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { LabelBadge, StatusBadge } from "@/components/prediction/status-badge";
import { ResearchEligibilityBadge } from "@/components/prediction/research-eligibility-badge";
import { ConfirmActionDialog } from "@/components/shared/confirm-action-dialog";
import { formatDateTime, formatDuration, formatProbability, sourceText } from "@/lib/formatters";
import type { PredictionHistoryItem } from "@/types/api";

export function PredictionHistoryTable({
  items,
  onDelete,
  onRerun,
}: {
  items: PredictionHistoryItem[];
  onDelete: (id: string) => void;
  onRerun: (id: string) => void;
}) {
  return (
    <div className="hidden rounded-lg border bg-card md:block">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Date</TableHead>
            <TableHead>Audio</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Result</TableHead>
            <TableHead>Research</TableHead>
            <TableHead>Confidence</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {items.map((item) => (
            <TableRow key={item.prediction_id}>
              <TableCell>{formatDateTime(item.created_at)}</TableCell>
              <TableCell>
                <Link className="font-medium hover:underline" href={`/dashboard/predictions/${item.prediction_id}`}>{item.filename || "Untitled audio"}</Link>
                <p className="text-xs text-muted-foreground">{sourceText(item.source_type)} · {formatDuration(item.duration_seconds)}</p>
              </TableCell>
              <TableCell><StatusBadge status={item.status} /></TableCell>
              <TableCell><LabelBadge label={item.final_prediction} /></TableCell>
              <TableCell><ResearchEligibilityBadge eligible={!item.mode_summary.contains_dummy && item.status === "completed"} /></TableCell>
              <TableCell>{formatProbability(item.confidence)}</TableCell>
              <TableCell>
                <div className="flex justify-end gap-2">
                  <Button asChild size="sm" variant="outline"><Link href={`/dashboard/predictions/${item.prediction_id}`}>Open</Link></Button>
                  <ConfirmActionDialog
                    trigger={<Button size="icon" variant="outline" aria-label="Rerun prediction"><RotateCcw className="h-4 w-4" /></Button>}
                    title="Rerun prediction?"
                    description="This creates a new prediction if source audio is still available."
                    actionLabel="Rerun"
                    onConfirm={() => onRerun(item.prediction_id)}
                  />
                  <ConfirmActionDialog
                    trigger={<Button size="icon" variant="outline" aria-label="Delete prediction"><Trash2 className="h-4 w-4" /></Button>}
                    title="Delete prediction?"
                    description="This removes the owner-scoped prediction record and retained audio when available."
                    actionLabel="Delete"
                    destructive
                    onConfirm={() => onDelete(item.prediction_id)}
                  />
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

export function PredictionHistoryCards({
  items,
  onDelete,
  onRerun,
}: {
  items: PredictionHistoryItem[];
  onDelete: (id: string) => void;
  onRerun: (id: string) => void;
}) {
  return (
    <div className="space-y-3 md:hidden">
      {items.map((item) => (
        <div key={item.prediction_id} className="rounded-lg border bg-card p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <Link className="font-medium hover:underline" href={`/dashboard/predictions/${item.prediction_id}`}>{item.filename || "Untitled audio"}</Link>
              <p className="mt-1 text-xs text-muted-foreground">{formatDateTime(item.created_at)}</p>
            </div>
            <LabelBadge label={item.final_prediction} />
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <StatusBadge status={item.status} />
            <ResearchEligibilityBadge eligible={!item.mode_summary.contains_dummy && item.status === "completed"} />
          </div>
          <p className="mt-3 text-sm text-muted-foreground">{sourceText(item.source_type)} · {formatDuration(item.duration_seconds)} · {formatProbability(item.confidence)}</p>
          <div className="mt-4 flex flex-wrap gap-2">
            <Button asChild size="sm"><Link href={`/dashboard/predictions/${item.prediction_id}`}>Open</Link></Button>
            <Button size="sm" variant="outline" onClick={() => onRerun(item.prediction_id)}><RotateCcw className="h-4 w-4" /> Rerun</Button>
            <Button size="sm" variant="outline" onClick={() => onDelete(item.prediction_id)}><Trash2 className="h-4 w-4" /> Delete</Button>
          </div>
        </div>
      ))}
    </div>
  );
}
