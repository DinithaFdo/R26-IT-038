"use client";

import { useAuth, UserButton } from "@clerk/nextjs";
import { createContext, useContext, useEffect, useMemo } from "react";
import { setApiAuthTokenGetter } from "@/lib/api/client";
import { isClerkConfigured } from "@/lib/config/env";

type AuthContextValue = {
  configured: boolean;
  getToken: () => Promise<string | null>;
};

const AuthContext = createContext<AuthContextValue>({
  configured: false,
  getToken: async () => null,
});

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const value = useMemo<AuthContextValue>(
    () => ({
      configured: isClerkConfigured,
      getToken: async () => null,
    }),
    [],
  );

  useEffect(() => {
    setApiAuthTokenGetter(value.getToken);
    return () => setApiAuthTokenGetter(null);
  }, [value]);

  return (
    <AuthContext.Provider value={value}>
      {isClerkConfigured ? <ClerkTokenBridge /> : null}
      {children}
    </AuthContext.Provider>
  );
}

export function useAuthBoundary() {
  return useContext(AuthContext);
}

export function AuthUserButton() {
  if (!isClerkConfigured) return null;
  return <UserButton />;
}

function ClerkTokenBridge() {
  const { getToken, isLoaded, isSignedIn } = useAuth();

  useEffect(() => {
    setApiAuthTokenGetter(async () => {
      if (!isLoaded || !isSignedIn) return null;
      return getToken();
    });
    return () => setApiAuthTokenGetter(null);
  }, [getToken, isLoaded, isSignedIn]);

  return null;
}
