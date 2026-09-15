"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, ShieldAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { dashboardNavItems } from "@/components/layout/navigation";
import { useAuthBoundary } from "@/providers/auth-provider";

export function DashboardHeader() {
  const pathname = usePathname();
  const auth = useAuthBoundary();
  return (
    <header className="sticky top-0 z-30 border-b bg-background/90 print:hidden backdrop-blur">
      <div className="flex min-h-16 items-center justify-between gap-4 px-4 lg:px-6">
        <div className="lg:hidden">
          <MobileNavigation pathname={pathname} />
        </div>
        <div>
          <p className="text-sm font-medium">MULTI-SCOPE</p>
          <p className="text-xs text-muted-foreground">Deepfake voice classification dashboard</p>
        </div>
        <div className="flex items-center gap-2">
          {!auth.configured ? (
            <Badge variant="warning" className="hidden sm:inline-flex">
              <ShieldAlert className="mr-1 h-3 w-3" />
              Auth adapter pending
            </Badge>
          ) : null}
          <Button asChild variant="outline" size="sm">
            <Link href="/dashboard/analyze">New analysis</Link>
          </Button>
        </div>
      </div>
    </header>
  );
}

function MobileNavigation({ pathname }: { pathname: string }) {
  return (
    <details className="relative">
      <summary className="flex h-10 w-10 cursor-pointer list-none items-center justify-center rounded-md border">
        <Menu className="h-5 w-5" />
        <span className="sr-only">Open navigation</span>
      </summary>
      <nav className="absolute left-0 top-12 w-64 rounded-lg border bg-popover p-2">
        {dashboardNavItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={`flex min-h-10 items-center gap-3 rounded-md px-3 text-sm ${pathname === item.href ? "bg-accent text-accent-foreground" : "text-muted-foreground"}`}
          >
            <item.icon className="h-4 w-4" />
            {item.label}
          </Link>
        ))}
      </nav>
    </details>
  );
}
