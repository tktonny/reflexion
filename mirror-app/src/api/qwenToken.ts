import { getActiveQwenTicket } from './sessionSync'

// Every Qwen call uses a short-lived ticket issued for the active backend session.
// The long-lived provider key never enters the application bundle.

export function clearTokenCache(): void {
  // Session ticket cache is owned by sessionSync and cleared when the session completes.
}

/** Bearer credential for a Qwen HTTP/WS call. */
export async function getBearer(): Promise<string> {
  return getActiveQwenTicket()
}
