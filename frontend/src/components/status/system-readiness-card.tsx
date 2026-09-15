import { Activity, Database, HardDrive, RadioTower, Wrench } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { ReadinessResponse } from "@/types/api";

function ReadyBadge({ ready }: { ready: boolean }) {
  return <Badge variant={ready ? "success" : "destructive"}>{ready ? "Ready" : "Not ready"}</Badge>;
}

export function SystemReadinessCard({ readiness }: { readiness: ReadinessResponse }) {
  const items = [
    { label: "Prediction readiness", ready: readiness.prediction_ready, icon: Activity },
    { label: "Research readiness", ready: readiness.research_ready, icon: RadioTower },
    { label: "Audio tools", ready: readiness.ffmpeg_available && readiness.ffprobe_available, icon: Wrench },
    { label: "MongoDB", ready: readiness.mongodb_configured && readiness.mongodb_available, icon: Database },
    { label: "Storage", ready: readiness.storage_enabled ? readiness.storage_available : true, icon: HardDrive },
  ];
  return (
    <Card>
      <CardHeader>
        <CardTitle>Backend readiness</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
        {items.map((item) => (
          <div key={item.label} className="rounded-lg border p-4">
            <item.icon className="h-5 w-5 text-primary" aria-hidden="true" />
            <p className="mt-3 text-sm font-medium">{item.label}</p>
            <div className="mt-2"><ReadyBadge ready={item.ready} /></div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
