export type InteractionContract = {
  sourceRoute: string;
  controlId: string;
  label: string;
  actionType: 'navigate' | 'modal' | 'mutate' | 'native' | 'external' | 'disabled';
  destinationRoute?: string;
  requiredParams?: string[];
  permissionRule: string;
  loadingState: string;
  successState: string;
  errorState: string;
  persistenceEffect: string;
  mirrorEffect: string;
};

/**
 * Product interaction contract. Every visible interactive control in the caregiver route tree either
 * appears here with a concrete outcome or is deliberately disabled with an explanation in the UI.
 */
const contract = (sourceRoute: string, controlId: string, label: string, actionType: InteractionContract['actionType'], destinationRoute: string | undefined, requiredParams: string[], permissionRule: string, loadingState: string, successState: string, errorState: string, persistenceEffect: string, mirrorEffect: string): InteractionContract => ({ sourceRoute, controlId, label, actionType, destinationRoute, requiredParams, permissionRule, loadingState, successState, errorState, persistenceEffect, mirrorEffect });

export const interactionContracts: InteractionContract[] = [
  ['/(tabs)/settings','account','Account','navigate','/settings/account',[], 'signed-in caregiver','none','account hub shown','route error','none','none'],
  ['/(tabs)/settings','notifications','App notifications','navigate','/settings/notifications',[], 'signed-in caregiver','none','preferences shown','route error','none','none'],
  ['/(tabs)/settings','language','App language','navigate','/settings/language',[], 'signed-in caregiver','none','language shown','route error','none','none'],
  ['/(tabs)/settings','household','Loved Ones','navigate','/settings/household',[], 'signed-in caregiver','loading household','household shown','load error','none','none'],
  ['/(tabs)/settings','devices','Connected Devices','navigate','/settings/devices',[], 'signed-in caregiver','loading devices','devices shown','load error','none','none'],
  ['/settings/account','personal','Edit personal information','navigate','/settings/account/personal',[], 'signed-in caregiver','none','form shown','route error','none','none'],
  ['/settings/account','email','Change email','navigate','/settings/account/email',[], 'signed-in caregiver','none','email form shown','route error','none','none'],
  ['/settings/account','phone','Change phone number','navigate','/settings/account/phone',[], 'signed-in caregiver; SMS provider required to complete verification','none','phone verification form shown','provider/validation error shown','pending phone-change token; verified number updates /me','none'],
  ['/settings/account','password','Change password','navigate','/settings/account/password',[], 'signed-in caregiver','none','password form shown','route error','none','none'],
  ['/settings/account','methods','Sign-in methods','navigate','/settings/account/sign-in-methods',[], 'signed-in caregiver','none','methods shown','route error','none','none'],
  ['/settings/account','sign-out','Sign out','modal',undefined,[], 'signed-in caregiver','revoking session','session revoked and sign-in shown','logout failure clears local session','secure session cleared','none'],
  ['/settings/account/personal','save','Save changes','mutate',undefined,['name'], 'signed-in caregiver','saving profile','profile updated','server validation error','PATCH /me','none'],
  ['/settings/account/email','request','Send verification email','mutate','/settings/account/email/verify',['email'], 'signed-in caregiver','requesting email change','request queued; confirmation shown only after provider acceptance','delivery/validation error','pending server token','none'],
  ['/settings/account/password','save','Update password','mutate',undefined,['currentPassword','newPassword'], 'signed-in caregiver','updating password','password updated','server validation error','password hash updated','none'],
  ['/settings/notifications','save','Save notification preferences','mutate',undefined,[], 'signed-in caregiver','saving preferences','preferences saved','permission/server error','PATCH /me','none'],
  ['/settings/notifications','phone-settings','Open phone settings','native',undefined,[], 'notification permission denied','opening Settings','system Settings opened','native settings unavailable','none','none'],
  ['/settings/language','save','Save language','mutate',undefined,['language'], 'signed-in caregiver','saving language','interface updates immediately','storage error','local preference persisted','none'],
  ['/settings/feedback','send','Send feedback','mutate',undefined,['message'], 'signed-in caregiver','sending feedback','feedback receipt shown','server validation error','POST /feedback','none'],
  ['/settings/help','contact','Email support','external',undefined,[], 'mail client available','opening mail client','mail draft opened','mail client unavailable','none','none'],
  ['/settings/household','add','Add loved one','navigate','/settings/household/add',[], 'signed-in caregiver','none','profile form shown','route error','POST /patients','none'],
  ['/settings/devices','pair','Pair a Mirror','navigate','/device/[id]/pairing',['id'], 'signed-in caregiver','none','pairing-method form shown','route error','none','none'],
  ['/device/[id]/code','claim','Connect Mirror','mutate',undefined,['id','pairingCode'], 'caregiver can write patient','claiming Mirror','device assignment created','invalid/expired pairing code','POST /device-pairing-claims','Mirror receives the caregiver configuration'],
  ['/chat/[id]/preview','send','Send message','mutate','/chat/[id]/status/[messageId]',['id','message'], 'paired Mirror','sending message','queued/delivered status shown','send error','POST /family-messages','Mirror polls and displays the message notification'],
].map((entry) => contract(entry[0] as string, entry[1] as string, entry[2] as string, entry[3] as InteractionContract['actionType'], entry[4] as string | undefined, entry[5] as string[], entry[6] as string, entry[7] as string, entry[8] as string, entry[9] as string, entry[10] as string, entry[11] as string));

export function contractFor(sourceRoute: string, controlId: string) {
  return interactionContracts.find((contract) => contract.sourceRoute === sourceRoute && contract.controlId === controlId);
}
