import React, { useState, useRef } from 'react';
import { useMutation } from '@tanstack/react-query';
import {
  View, Text, StyleSheet, FlatList, TextInput, TouchableOpacity,
  KeyboardAvoidingView, Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  listSupportThreadsV1,
  openSupportThreadV1,
  postSupportMessageV1,
} from '../src/lib/v1Client';
import { hasV1Session } from '../src/lib/v1AuthSession';
import { STATUS_META } from '../src/lib/v1Status';
import { colors, spacing, radius, fontSize, MIN_TOUCH_TARGET } from '../src/theme';

interface Message {
  id: string;
  from: 'user' | 'support';
  text: string;
  time: string;
}

const INITIAL_MESSAGES: Message[] = [
  {
    id: '0',
    from: 'support',
    text: "Hi there! 👋 I'm here to help with any questions about Reflexion. What can I help you with today?",
    time: now(),
  },
];

// Wizard-of-Oz: in MVP, messages go to a human support agent (a v1 support thread the operator console
// reads). The acknowledgement below is only shown once the message has actually been delivered — it used
// to appear unconditionally while the request 404'd into a console warning.
const DELIVERED_ACKNOWLEDGEMENT = 'Thank you, we will bring you to a chat with our Reflexion team.';
const UNDELIVERED_NOTICE =
  "We could not reach our support team just now, so this message was not sent. Please check your connection and try again.";

function now(): string {
  return new Date().toLocaleTimeString('en-SG', { hour: '2-digit', minute: '2-digit' });
}

export default function ChatbotScreen() {
  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES);
  const [input, setInput] = useState('');
  const listRef = useRef<FlatList>(null);
  // One thread per caregiver conversation: opened on the first message, reused for the rest.
  const threadIdRef = useRef<string | null>(null);
  const sendMessageMutation = useMutation({ mutationFn: deliverSupportMessage });

  function appendSupportReply(text: string) {
    setMessages(prev => [...prev, { id: `${Date.now()}-support`, from: 'support', text, time: now() }]);
    setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 100);
  }

  async function send() {
    const text = input.trim();
    if (!text || sendMessageMutation.isPending) return;
    const userMsg: Message = { id: Date.now().toString(), from: 'user', text, time: now() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');

    try {
      threadIdRef.current = await sendMessageMutation.mutateAsync({ text, threadId: threadIdRef.current });
      appendSupportReply(DELIVERED_ACKNOWLEDGEMENT);
    } catch {
      appendSupportReply(UNDELIVERED_NOTICE);
    }
  }

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.agentBanner}>
        <View accessibilityElementsHidden importantForAccessibility="no" style={styles.agentDot} />
        <Text style={styles.agentText}>Reflexion Support · Usually replies in &lt;2 hours</Text>
      </View>

      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined} keyboardVerticalOffset={90}>
        <FlatList
          ref={listRef}
          data={messages}
          keyExtractor={m => m.id}
          contentContainerStyle={styles.list}
          onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: false })}
          renderItem={({ item, index }) => (
            <View
              accessible
              // One label per bubble: who said it, when, and what. Read as separate nodes, a screen reader
              // announces the timestamp as an orphaned number after every message.
              accessibilityLabel={`${item.from === 'user' ? 'You' : 'Reflexion Support'} at ${item.time}. ${item.text}`}
              // Only the newest support bubble announces itself, so the caregiver hears whether the message
              // reached us — the delivery outcome is carried in this bubble and nowhere else on screen.
              accessibilityLiveRegion={
                item.from === 'support' && index === messages.length - 1 ? 'polite' : 'none'
              }
              style={[styles.bubble, item.from === 'user' ? styles.bubbleUser : styles.bubbleSupport]}
            >
              <Text style={[styles.bubbleText, item.from === 'user' && styles.bubbleTextUser]}>{item.text}</Text>
              <Text style={[styles.bubbleTime, item.from === 'user' && styles.bubbleTimeUser]}>{item.time}</Text>
            </View>
          )}
        />

        <View style={styles.inputBar}>
          <TextInput
            // The placeholder is the only visible label, and a placeholder disappears the moment typing
            // starts — the label has to be spelled out for the screen reader.
            accessibilityLabel="Type a message to Reflexion Support"
            style={styles.input}
            placeholder="Type a message..."
            value={input}
            onChangeText={setInput}
            multiline
            returnKeyType="send"
            onSubmitEditing={() => void send()}
            blurOnSubmit={false}
          />
          <TouchableOpacity
            accessibilityLabel="Send message"
            accessibilityRole="button"
            accessibilityState={{ disabled: !input.trim() }}
            style={[styles.sendBtn, !input.trim() && styles.sendBtnDisabled]}
            onPress={() => void send()}
            disabled={!input.trim()}
          >
            <Text style={styles.sendBtnText}>Send</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

/**
 * Delivers one caregiver message into a v1 support thread (the same threads the operator console reads)
 * and returns the thread it landed in. Throws when the message did not reach support, so the screen can
 * say so instead of showing a reassuring auto-reply over a dropped message.
 */
async function deliverSupportMessage({ text, threadId }: { text: string; threadId: string | null }): Promise<string> {
  if (!hasV1Session()) {
    throw new Error('Support needs you to be signed in.');
  }

  if (threadId) {
    await postSupportMessageV1(threadId, text);
    return threadId;
  }

  // Continue the caregiver's most recent conversation if they already have one, so a returning user does
  // not spawn a new thread per visit.
  const existing = await listSupportThreadsV1().catch(() => [] as Awaited<ReturnType<typeof listSupportThreadsV1>>);
  const openThread = existing.find((thread) => thread.status === 'open') || existing[0];
  if (openThread?.threadId) {
    await postSupportMessageV1(openThread.threadId, text);
    return openThread.threadId;
  }

  const created = await openSupportThreadV1(subjectFor(text), text);
  return created.threadId;
}

function subjectFor(message: string) {
  const firstLine = message.replace(/\s+/g, ' ').trim();
  return firstLine.length > 60 ? `${firstLine.slice(0, 57)}...` : firstLine || 'Support request';
}

// Brought onto src/theme.ts alongside app/faq.tsx — these two were the last screens on an older blue
// palette that predated the theme, and the chat's timestamp grey (#AAA on white) was 2.32:1, well under AA.
const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.surface.page },
  agentBanner: {
    flexDirection: 'row', alignItems: 'center', gap: spacing.sm,
    paddingHorizontal: spacing.lg, paddingVertical: 10, backgroundColor: colors.surface.card,
    borderBottomWidth: 1, borderBottomColor: colors.border.default,
  },
  agentDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: STATUS_META.doing_well.dot },
  agentText: { fontSize: fontSize.body, color: colors.text.secondary },
  list: { padding: spacing.lg, gap: 10, paddingBottom: spacing.sm },
  bubble: {
    maxWidth: '80%', borderRadius: radius.xl, padding: spacing.md,
    shadowColor: colors.shadow, shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.04, shadowRadius: 2, elevation: 1,
  },
  bubbleUser: { alignSelf: 'flex-end', backgroundColor: colors.accent, borderBottomRightRadius: 4 },
  bubbleSupport: { alignSelf: 'flex-start', backgroundColor: colors.surface.card, borderBottomLeftRadius: 4 },
  bubbleText: { fontSize: fontSize.subheading, color: colors.text.primary, lineHeight: 21 },
  bubbleTextUser: { color: colors.text.onAccent },
  bubbleTime: { fontSize: fontSize.caption, color: colors.text.tertiary, marginTop: spacing.xs, textAlign: 'right' },
  bubbleTimeUser: { color: 'rgba(255,255,255,0.65)' },
  inputBar: {
    flexDirection: 'row', alignItems: 'flex-end', gap: spacing.sm,
    padding: spacing.md, backgroundColor: colors.surface.card, borderTopWidth: 1, borderTopColor: colors.border.default,
  },
  input: {
    // 44pt floor so the field is easy to hit one-handed, and a taller cap so two lines still fit when the
    // system text size is turned up — at 100 the caregiver could only see part of one line.
    flex: 1, backgroundColor: colors.surface.input, borderRadius: 22, paddingHorizontal: spacing.lg, paddingVertical: 10,
    fontSize: fontSize.subheading, minHeight: MIN_TOUCH_TARGET, maxHeight: 132, borderWidth: 1, borderColor: colors.border.default,
  },
  sendBtn: {
    alignItems: 'center', backgroundColor: colors.accent, borderRadius: 22, justifyContent: 'center',
    // Padding alone left this at roughly 41pt tall, under the 44pt minimum.
    minHeight: MIN_TOUCH_TARGET, minWidth: MIN_TOUCH_TARGET, paddingHorizontal: spacing.xl, paddingVertical: spacing.md,
  },
  sendBtnDisabled: { backgroundColor: colors.border.strong },
  sendBtnText: { color: colors.text.onAccent, fontWeight: '700', fontSize: fontSize.bodyLarge },
});
