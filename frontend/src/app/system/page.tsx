import Link from "next/link";
import { Button } from "@/components/ui/button";
import { SystemStatusPage } from "@/features/system-status/system-status-page";

export const metadata = {
  title: "System Status",
};

export default function PublicSystemPage() {
  return (
    <main id="main-content" className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container flex min-h-16 items-center justify-between">
          <Link href="/" className="font-mono text-sm font-semibold">MULTI-SCOPE</Link>
          <Button asChild variant="outline" size="sm"><Link href="/dashboard">Dashboard</Link></Button>
        </div>
      </header>
      <div className="container py-10">
        <SystemStatusPage />
      </div>
    </main>
  );
}
