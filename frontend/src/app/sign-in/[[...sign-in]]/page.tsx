import Link from "next/link";
import { SignIn } from "@clerk/nextjs";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { isClerkConfigured } from "@/lib/config/env";

export const metadata = {
  title: "Sign in",
};

export default function SignInPage() {
  if (isClerkConfigured) {
    return (
      <main id="main-content" className="flex min-h-screen items-center justify-center bg-background p-6">
        <SignIn
          routing="path"
          path="/sign-in"
          signUpUrl="/sign-up"
          fallbackRedirectUrl="/dashboard"
          appearance={{
            elements: {
              rootBox: "mx-auto",
              cardBox: "shadow-none border border-neutral-200 rounded-lg",
            },
          }}
        />
      </main>
    );
  }

  return (
    <main id="main-content" className="flex min-h-screen items-center justify-center p-6">
      <Card className="max-w-md">
        <CardHeader>
          <CardTitle>Clerk configuration required</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm text-muted-foreground">
          <p>Set `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` and `CLERK_SECRET_KEY` to enable protected dashboard routes and backend session-token injection.</p>
          <Button asChild>
            <Link href="/">Return home</Link>
          </Button>
        </CardContent>
      </Card>
    </main>
  );
}
