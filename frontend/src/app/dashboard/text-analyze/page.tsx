import type { Metadata } from "next";
import { TextAnalyzePage } from "@/features/text-classification/analyze-page";

export const metadata: Metadata = {
  title: "Text Analysis",
  description: "Detect AI-generated text with explainability.",
};

export default function Page() {
  return <TextAnalyzePage />;
}
