import Link from "next/link";
import { BookOpen, Home } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";

const docsNav = [
  { href: "/developers", label: "Overview" },
  { href: "/developers/api", label: "REST API" },
  { href: "/developers/mcp", label: "MCP Server" },
];

export function DocsLayout({ children }: { children: React.ReactNode }) {
  return (
    <main id="main-content" className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container flex min-h-16 items-center justify-between gap-4">
          <Link href="/" className="flex items-center gap-2 font-mono text-sm font-semibold">
            <BookOpen className="h-4 w-4" /> MULTI-SCOPE Docs
          </Link>
          <Button asChild variant="outline" size="sm"><Link href="/"><Home className="h-4 w-4" /> Home</Link></Button>
        </div>
      </header>
      <div className="container grid gap-8 py-10 lg:grid-cols-[240px_1fr]">
        <aside className="lg:sticky lg:top-20 lg:h-[calc(100vh-6rem)]">
          <nav className="space-y-1" aria-label="Developer documentation">
            {docsNav.map((item) => (
              <Link key={item.href} href={item.href} className="block rounded-md px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-foreground">
                {item.label}
              </Link>
            ))}
          </nav>
          <Separator className="my-5" />
          <p className="text-xs text-muted-foreground">All examples use placeholders. Do not paste real keys into documentation or source code.</p>
        </aside>
        <div className="min-w-0">{children}</div>
      </div>
    </main>
  );
}
