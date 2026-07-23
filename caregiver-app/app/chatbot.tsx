import React, { useState, useRef } from 'react';
import { useMutation } from '@tanstack/react-query';
import {
  View, Text, StyleSheet, FlatList, TextInput, TouchableOpacity,
  KeyboardAvoidingView, Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { apiSend } from '../src/lib/apiClient';
import { getStoredAuthSession } from '../src/lib/authSession';

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

// Wizard-of-Oz: in MVP, messages go to a human support agent.
// Boilerplate auto-responses simulate the flow for demo purposes.
const AUTO_RESPONSES: Record<string, string> = {
  default: 'Thank you, we will bring you to a chat with our Reflexion team.',
};

function getAutoResponse(): string {
  return AUTO_RESPONSES.default;
}

function now(): string {
  return new Date().toLocaleTimeString('en-SG', { hour: '2-digit', minute: '2-digit' });
}

export default function ChatbotScreen() {
  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES);
  const [input, setInput] = useState('');
  const listRef = useRef<FlatList>(null);
  const feedbackMutation = useMutation({
    mutationFn: saveFeedback,
    onError: (error) => {
      console.warn('[ChatbotScreen] feedback save failed', error);
    },
  });

  async function send() {
    const text = input.trim();
    if (!text) return;
    const userMsg: Message = { id: Date.now().toString(), from: 'user', text, time: now() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');

    feedbackMutation.mutate(text);

    // Simulate a support response after 1.2s
    setTimeout(() => {
      const reply: Message = {
        id: (Date.now() + 1).toString(),
        from: 'support',
        text: getAutoResponse(),
        time: now(),
      };
      setMessages(prev => [...prev, reply]);
      setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 100);
    }, 1200);
  }

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.agentBanner}>
        <View style={styles.agentDot} />
        <Text style={styles.agentText}>Reflexion Support · Usually replies in &lt;2 hours</Text>
      </View>

      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined} keyboardVerticalOffset={90}>
        <FlatList
          ref={listRef}
          data={messages}
          keyExtractor={m => m.id}
          contentContainerStyle={styles.list}
          onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: false })}
          renderItem={({ item }) => (
            <View style={[styles.bubble, item.from === 'user' ? styles.bubbleUser : styles.bubbleSupport]}>
              <Text style={[styles.bubbleText, item.from === 'user' && styles.bubbleTextUser]}>{item.text}</Text>
              <Text style={[styles.bubbleTime, item.from === 'user' && styles.bubbleTimeUser]}>{item.time}</Text>
            </View>
          )}
        />

        <View style={styles.inputBar}>
          <TextInput
            style={styles.input}
            placeholder="Type a message..."
            value={input}
            onChangeText={setInput}
            multiline
            returnKeyType="send"
            onSubmitEditing={() => void send()}
            blurOnSubmit={false}
          />
          <TouchableOpacity style={[styles.sendBtn, !input.trim() && styles.sendBtnDisabled]} onPress={() => void send()} disabled={!input.trim()}>
            <Text style={styles.sendBtnText}>Send</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

async function saveFeedback(message: string) {
  const session = getStoredAuthSession();
  if (!session?.nurseId) {
    console.warn('[ChatbotScreen] feedback skipped: missing nurse session');
    return;
  }

  await apiSend('/api/feedback', {
    method: 'POST',
    body: JSON.stringify({
      nurseId: session.nurseId,
      message,
    }),
  });
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#F7F9FC' },
  agentBanner: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    paddingHorizontal: 16, paddingVertical: 10, backgroundColor: '#fff',
    borderBottomWidth: 1, borderBottomColor: '#F0F0F0',
  },
  agentDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: '#2ECC71' },
  agentText: { fontSize: 13, color: '#555' },
  list: { padding: 16, gap: 10, paddingBottom: 8 },
  bubble: {
    maxWidth: '80%', borderRadius: 16, padding: 12,
    shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.04, shadowRadius: 2, elevation: 1,
  },
  bubbleUser: { alignSelf: 'flex-end', backgroundColor: '#1A6FA8', borderBottomRightRadius: 4 },
  bubbleSupport: { alignSelf: 'flex-start', backgroundColor: '#fff', borderBottomLeftRadius: 4 },
  bubbleText: { fontSize: 15, color: '#333', lineHeight: 21 },
  bubbleTextUser: { color: '#fff' },
  bubbleTime: { fontSize: 11, color: '#AAA', marginTop: 4, textAlign: 'right' },
  bubbleTimeUser: { color: 'rgba(255,255,255,0.65)' },
  inputBar: {
    flexDirection: 'row', alignItems: 'flex-end', gap: 8,
    padding: 12, backgroundColor: '#fff', borderTopWidth: 1, borderTopColor: '#F0F0F0',
  },
  input: {
    flex: 1, backgroundColor: '#F7F9FC', borderRadius: 22, paddingHorizontal: 16, paddingVertical: 10,
    fontSize: 15, maxHeight: 100, borderWidth: 1, borderColor: '#E0E0E0',
  },
  sendBtn: { backgroundColor: '#1A6FA8', borderRadius: 22, paddingHorizontal: 20, paddingVertical: 12 },
  sendBtnDisabled: { backgroundColor: '#B0C4D8' },
  sendBtnText: { color: '#fff', fontWeight: '700', fontSize: 14 },
});
