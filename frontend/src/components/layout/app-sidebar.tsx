import Link from "next/link";
import { ShieldCheck } from "lucide-react";
import { dashboardNavItems } from "@/components/layout/navigation";
import { cn } from "@/lib/utils";

export function AppSidebar({ pathname }: { pathname?: string }) {
  return (
    <aside className="hidden min-h-screen w-72 shrink-0 border-r bg-card/70 print:hidden lg:block">
      <div className="sticky top-0 flex h-screen flex-col p-5">
        <Link href="/" className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <ShieldCheck className="h-5 w-5" />
          </div>
          <div>
            <p className="font-semibold">MULTI-SCOPE</p>
            <p className="text-xs text-muted-foreground">Voice research console</p>
          </div>
        </Link>
        <nav className="mt-8 space-y-1">
          {dashboardNavItems.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex min-h-10 items-center gap-3 rounded-md px-3 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground",
                  active && "bg-accent text-accent-foreground",
                )}
              >
                <item.icon className="h-4 w-4" aria-hidden="true" />
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="mt-auto rounded-lg border bg-background/70 p-4 text-xs text-muted-foreground">
          Backend validation remains authoritative. The dashboard does not run or simulate model inference.
        </div>
      </div>
    </aside>
  );
}
