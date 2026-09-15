import { Progress } from "@/components/ui/progress";

export function UploadProgress({ value, stage }: { value: number; stage: string }) {
  return (
    <div className="rounded-lg border bg-card p-4" aria-live="polite">
      <div className="mb-2 flex justify-between text-sm">
        <span>{stage}</span>
        <span>{value}%</span>
      </div>
      <Progress value={value} />
    </div>
  );
}
