import { SignUp } from "@clerk/nextjs";
import { isClerkConfigured } from "@/lib/config/env";
import SignInFallback from "@/app/sign-in/[[...sign-in]]/page";

export const metadata = {
  title: "Sign up",
};

export default function SignUpPage() {
  if (!isClerkConfigured) return <SignInFallback />;

  return (
    <main id="main-content" className="flex min-h-screen items-center justify-center bg-background p-6">
      <SignUp
        routing="path"
        path="/sign-up"
        signInUrl="/sign-in"
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
