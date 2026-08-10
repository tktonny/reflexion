import {
  chineseConversationPrompt,
  englishConversationPrompt,
} from '../prompts'

export const conversationPrompt: Record<string, string> = {
  en: englishConversationPrompt,
  zh: chineseConversationPrompt,
}
