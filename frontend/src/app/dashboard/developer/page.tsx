"use client";

import { PageHeader } from "@/components/shared/page-header";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ApiKeysManager } from "@/components/developer/api-keys-manager";
import { Braces, Bot, Network, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function DeveloperHubPage() {
  const codeSnippet = `curl -X POST "http://localhost:8000/api/v1/external/predictions" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -F "file=@sample.wav" \\
  -F "idempotency_key=ext-001"`;

  const mcpConfig = `{
  "mcpServers": {
    "multi-scope": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "env": {
        "API_BASE_URL": "http://localhost:8000",
        "API_KEY": "YOUR_API_KEY"
      }
    }
  }
}`;

  const copyCode = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <div className="space-y-6">
      <PageHeader 
        title="Developer" 
        description="Integrate MULTI-SCOPE into your applications and AI workflows." 
      />

      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="flex flex-wrap gap-2 h-auto p-1 bg-transparent border-b rounded-none w-full justify-start">
          <TabsTrigger value="overview" className="data-[state=active]:bg-muted">Overview</TabsTrigger>
          <TabsTrigger value="keys" className="data-[state=active]:bg-muted">API Keys</TabsTrigger>
          <TabsTrigger value="api" className="data-[state=active]:bg-muted">API Integration</TabsTrigger>
          <TabsTrigger value="mcp" className="data-[state=active]:bg-muted">MCP & AI Integration</TabsTrigger>
          <TabsTrigger value="platform" className="data-[state=active]:bg-muted">Platform Integration</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader>
                <Braces className="h-5 w-5 text-brand mb-2" />
                <CardTitle>API Integration</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground space-y-4">
                <p>Direct REST access for automated analysis workflows, batch processing, and internal tools.</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <Bot className="h-5 w-5 text-brand mb-2" />
                <CardTitle>MCP & AI Integration</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground space-y-4">
                <p>Equip LLMs and AI agents with deepfake voice analysis capabilities via the Model Context Protocol.</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <Network className="h-5 w-5 text-brand mb-2" />
                <CardTitle>Platform Integration</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                <p>Embed authenticity checks into SaaS platforms, moderation pipelines, or communication apps.</p>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Quick Start</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
                <li>Navigate to the <strong>API Keys</strong> tab and create a new key.</li>
                <li>Copy the generated secret safely.</li>
                <li>Send an audio file via the REST API using <code className="bg-muted px-1 rounded">Authorization: Bearer YOUR_API_KEY</code>.</li>
                <li>Receive a prediction ID and poll for status.</li>
                <li>Retrieve the branch-level scores and fusion result.</li>
              </ol>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="keys">
          <ApiKeysManager />
        </TabsContent>

        <TabsContent value="api" id="api-integration" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Authentication</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm text-muted-foreground">
              <p>All external API requests require an API key passed in the <code className="bg-muted px-1 rounded">Authorization</code> header as a Bearer token.</p>
              <div className="bg-muted p-4 rounded-md font-mono flex justify-between items-center text-foreground">
                <span>Authorization: Bearer YOUR_API_KEY</span>
              </div>
            </CardContent>
          </Card>

          <Card id="openapi-spec">
            <CardHeader>
              <CardTitle>OpenAPI / Submit Audio</CardTitle>
              <CardDescription>Upload a voice recording for analysis (16kHz WAV recommended).</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="bg-code-background p-4 rounded-md relative text-sm font-mono overflow-x-auto text-foreground">
                <pre>{codeSnippet}</pre>
                <Button 
                  size="icon" 
                  variant="ghost" 
                  className="absolute top-2 right-2 h-8 w-8 text-muted-foreground hover:text-foreground"
                  onClick={() => copyCode(codeSnippet)}
                >
                  <Copy className="h-4 w-4" />
                </Button>
              </div>
              <div className="text-sm text-muted-foreground">
                <h4 className="font-semibold text-foreground mt-4 mb-2">Errors</h4>
                <p>The API uses structured error responses with HTTP status codes and detailed message payloads.</p>
                <h4 className="font-semibold text-foreground mt-4 mb-2">Limits</h4>
                <p>Max file size is typically 50MB, max duration 10 minutes (configurable on the backend).</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="mcp" id="mcp-integration" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Model Context Protocol (MCP)</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm text-muted-foreground">
              <p>The MULTI-SCOPE MCP server allows supported AI clients (like Claude Desktop) to invoke voice classification tools securely.</p>
              
              <h4 className="font-semibold text-foreground mt-4 mb-2">Configuration Steps</h4>
              <ol className="list-decimal list-inside space-y-2">
                <li>Create an API Key in the API Keys tab.</li>
                <li>Copy the configuration block below.</li>
                <li>Add it to your AI client&apos;s MCP configuration file (e.g., <code className="bg-muted px-1 rounded">claude_desktop_config.json</code>).</li>
                <li>Restart the client.</li>
                <li>Ask the AI to &quot;Analyze this voice recording for deepfakes.&quot;</li>
              </ol>

              <div className="bg-code-background p-4 rounded-md relative text-sm font-mono overflow-x-auto mt-4 text-foreground">
                <pre>{mcpConfig}</pre>
                <Button 
                  size="icon" 
                  variant="ghost" 
                  className="absolute top-2 right-2 h-8 w-8 text-muted-foreground hover:text-foreground"
                  onClick={() => copyCode(mcpConfig)}
                >
                  <Copy className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Troubleshooting</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm text-muted-foreground">
              <ul className="list-disc list-inside space-y-2">
                <li><strong>Connection failed:</strong> Ensure the backend server is running and accessible at the specified URL.</li>
                <li><strong>Authentication failed:</strong> Verify your API key is active and correctly formatted in the env config.</li>
                <li><strong>Tool not visible:</strong> Restart the AI client after modifying its config file.</li>
              </ul>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="platform" id="platform-integration" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Architecture Flow</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm text-muted-foreground">
              <p>For integrating MULTI-SCOPE into a larger SaaS or moderation pipeline:</p>
              <div className="bg-muted p-4 rounded-md font-mono text-center space-y-2 text-foreground">
                <p>Your Platform</p>
                <p>↓</p>
                <p>MULTI-SCOPE API (Submit Job)</p>
                <p>↓</p>
                <p>Asynchronous Analysis</p>
                <p>↓</p>
                <p>Poll or Webhook (if supported)</p>
                <p>↓</p>
                <p>Action on Result</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
