/**
 * The v1 API error type, deliberately in its own module.
 *
 * It lives here rather than in v1Client because v1Client reaches SecureStore for the session, which pulls in
 * expo-modules-core and cannot load outside Metro. Anything that only needs to CLASSIFY a failure — the
 * caregiver-facing message mapping, and its tests — should not have to boot a native module to do it.
 */
export class V1ApiError extends Error {
  status: number;
  code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = 'V1ApiError';
    this.status = status;
    this.code = code;
  }
}
