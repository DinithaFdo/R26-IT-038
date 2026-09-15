"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Copy, Plus, Trash2, Key, Check } from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { format } from "date-fns";
import { apiClient } from "@/lib/api/client";
import { Badge } from "@/components/ui/badge";

export function ApiKeysManager() {
  const queryClient = useQueryClient();
  const [newKeyName, setNewKeyName] = useState("");
  const [createdSecret, setCreatedSecret] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const fetchKeys = async () => {
    const { data } = await apiClient.get("/api/v1/me/api-keys");
    return data.items || [];
  };

  const { data: apiKeys, isLoading } = useQuery({
    queryKey: ["api-keys"],
    queryFn: fetchKeys
  });

  const createMutation = useMutation({
    mutationFn: async (name: string) => {
      const { data } = await apiClient.post("/api/v1/me/api-keys", { name, scopes: ["predict"] });
      return data;
    },
    onSuccess: (data) => {
      setCreatedSecret(data.secret_key);
      setNewKeyName("");
      queryClient.invalidateQueries({ queryKey: ["api-keys"] });
    }
  });

  const revokeMutation = useMutation({
    mutationFn: async (id: string) => {
      const { data } = await apiClient.delete(`/api/v1/me/api-keys/${id}`);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["api-keys"] });
    }
  });

  const copyToClipboard = () => {
    if (createdSecret) {
      navigator.clipboard.writeText(createdSecret);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-xl">Create API Key</CardTitle>
          <CardDescription>Generate a secure key for programmatic access.</CardDescription>
        </CardHeader>
        <CardContent>
          {!createdSecret ? (
            <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center">
              <Input
                placeholder="Key Name (e.g., Production Backend)"
                value={newKeyName}
                onChange={(e) => setNewKeyName(e.target.value)}
                className="max-w-md"
              />
              <Button
                disabled={!newKeyName.trim() || createMutation.isPending}
                onClick={() => createMutation.mutate(newKeyName)}
                className="bg-brand text-brand-foreground hover:bg-brand/90"
              >
                <Plus className="mr-2 h-4 w-4" /> Generate Key
              </Button>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="p-4 rounded-md bg-amber-500/10 border border-amber-500/20 text-amber-500 text-sm">
                <strong>Important:</strong> Copy your API key now. You will not be able to see it again!
              </div>
              <div className="flex gap-2 items-center">
                <code className="flex-1 p-3 bg-muted rounded-md font-mono text-sm break-all">
                  {createdSecret}
                </code>
                <Button onClick={copyToClipboard} variant="outline" size="icon">
                  {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <Button onClick={() => setCreatedSecret(null)} variant="secondary">
                I have saved this key safely
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-xl">Existing Keys</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="text-sm text-muted-foreground">Loading keys...</div>
          ) : apiKeys && apiKeys.length > 0 ? (
            <div className="space-y-4">
              {apiKeys.map((key: { id: string; name: string; status: string; key_prefix: string; created_at: string; last_used_at?: string }) => (
                <div key={key.id} className="flex flex-col sm:flex-row sm:items-center justify-between p-4 border rounded-md gap-4">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <Key className="h-4 w-4 text-muted-foreground" />
                      <span className="font-medium">{key.name}</span>
                      {key.status === "active" ? (
                        <Badge variant="outline" className="text-green-500 border-green-500/30">Active</Badge>
                      ) : (
                        <Badge variant="outline" className="text-red-500 border-red-500/30">Revoked</Badge>
                      )}
                    </div>
                    <div className="text-xs font-mono text-muted-foreground">Prefix: {key.key_prefix}****</div>
                    <div className="text-xs text-muted-foreground flex gap-4">
                      <span>Created: {format(new Date(key.created_at), "MMM d, yyyy")}</span>
                      {key.last_used_at && <span>Last used: {format(new Date(key.last_used_at), "MMM d, yyyy")}</span>}
                    </div>
                  </div>
                  {key.status === "active" && (
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={() => revokeMutation.mutate(key.id)}
                      disabled={revokeMutation.isPending}
                    >
                      <Trash2 className="h-4 w-4 mr-2" /> Revoke
                    </Button>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-sm text-muted-foreground text-center py-8">
              No API keys created yet.
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
