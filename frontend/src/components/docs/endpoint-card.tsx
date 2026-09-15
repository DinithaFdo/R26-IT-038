import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CopyCodeButton } from "@/components/docs/copy-code-button";
import type { EndpointDoc } from "@/components/docs/docs-data";

export function EndpointCard({ endpoint }: { endpoint: EndpointDoc }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={endpoint.method === "GET" ? "outline" : endpoint.method === "DELETE" ? "destructive" : "default"}>{endpoint.method}</Badge>
          <code className="rounded-md bg-muted px-2 py-1 font-mono text-sm">{endpoint.path}</code>
        </div>
        <CardTitle className="pt-3">{endpoint.purpose}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <dl className="grid gap-3 md:grid-cols-2">
          <div><dt className="text-muted-foreground">Authentication</dt><dd>{endpoint.auth}</dd></div>
          <div><dt className="text-muted-foreground">Required scopes</dt><dd>{endpoint.scopes || "None"}</dd></div>
          <div><dt className="text-muted-foreground">Response</dt><dd className="font-mono">{endpoint.response}</dd></div>
          <div><dt className="text-muted-foreground">Errors</dt><dd>{endpoint.errors.join(", ")}</dd></div>
        </dl>
        {endpoint.params?.length ? <p><span className="text-muted-foreground">Path params:</span> {endpoint.params.join(", ")}</p> : null}
        {endpoint.query?.length ? <p><span className="text-muted-foreground">Query params:</span> {endpoint.query.join(", ")}</p> : null}
        {endpoint.body?.length ? <p><span className="text-muted-foreground">Request fields:</span> {endpoint.body.join(", ")}</p> : null}
        <div className="rounded-lg border bg-[hsl(var(--code-background))] p-4">
          <div className="mb-3 flex items-center justify-between gap-3">
            <span className="font-mono text-xs uppercase text-muted-foreground">cURL</span>
            <CopyCodeButton value={endpoint.curl} />
          </div>
          <pre className="overflow-auto font-mono text-xs leading-6"><code>{endpoint.curl}</code></pre>
        </div>
      </CardContent>
    </Card>
  );
}
