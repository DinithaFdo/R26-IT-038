"use client";

import { usePathname } from "next/navigation";
import { AppSidebar } from "@/components/layout/app-sidebar";
import { DashboardHeader } from "@/components/layout/dashboard-header";

export function DashboardShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  return (
    <div className="flex min-h-screen bg-background">
      <AppSidebar pathname={pathname} />
      <div className="min-w-0 flex-1">
        <DashboardHeader />
        <main className="mx-auto w-full max-w-7xl p-4 print:max-w-none print:p-0 lg:p-6">{children}</main>
      </div>
    </div>
  );
}
