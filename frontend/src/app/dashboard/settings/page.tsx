import { ShieldCheck } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader } from "@/components/shared/page-header";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AppearanceSettings } from "@/components/settings/appearance-settings";

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <PageHeader title="Settings" description="Minimal account, appearance, privacy, and integration information." />
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>Account</CardTitle></CardHeader>
          <CardContent>
            <Alert>
              <ShieldCheck className="mb-2 h-4 w-4" />
              <AlertTitle>Clerk authentication</AlertTitle>
              <AlertDescription>
                Sign-in is handled by Clerk. Every backend request carries a fresh Clerk session token as a bearer header, obtained per-request through the auth provider boundary. The token is never persisted to localStorage, sessionStorage, or the query cache.
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>
        <AppearanceSettings />
        <Card>
          <CardHeader><CardTitle>Privacy and audio retention</CardTitle></CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            Audio retention and playback availability are controlled by the backend storage policy. Signed playback URLs are requested only for eligible prediction detail pages and are not persisted.
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>System version</CardTitle></CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            No frontend-readable backend version endpoint was verified beyond root metadata and health/readiness.
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
