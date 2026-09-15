import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function NotFound() {
  return (
    <main className="flex min-h-screen items-center justify-center p-6">
      <div className="max-w-md rounded-lg border bg-card p-6 text-center">
        <h1 className="text-2xl font-semibold">Page not found</h1>
        <p className="mt-2 text-sm text-muted-foreground">This MULTI-SCOPE page does not exist.</p>
        <Button asChild className="mt-5">
          <Link href="/dashboard">Return to dashboard</Link>
        </Button>
      </div>
    </main>
  );
}
