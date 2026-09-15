import { LucideIcon } from "lucide-react";
import { Button } from "@/components/ui/button";

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
}: {
  icon: LucideIcon;
  title: string;
  description: string;
  action?: { href: string; label: string };
}) {
  return (
    <div className="rounded-lg border border-dashed bg-card p-8 text-center">
      <Icon className="mx-auto h-8 w-8 text-muted-foreground" aria-hidden="true" />
      <h2 className="mt-4 text-base font-semibold">{title}</h2>
      <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">{description}</p>
      {action ? (
        <Button asChild className="mt-5">
          <a href={action.href}>{action.label}</a>
        </Button>
      ) : null}
    </div>
  );
}
