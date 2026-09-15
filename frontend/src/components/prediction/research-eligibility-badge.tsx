import { Badge } from "@/components/ui/badge";

export function ResearchEligibilityBadge({ eligible }: { eligible: boolean }) {
  return eligible ? (
    <Badge variant="success">Research eligible</Badge>
  ) : (
    <Badge variant="warning">Research validation in progress</Badge>
  );
}
