import React, { useState, useRef } from 'react';
import {
  View, Text, StyleSheet, FlatList, TextInput, TouchableOpacity,
  KeyboardAvoidingView, Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

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
  default: "Thanks for your message! Our team will get back to you shortly. Typical response time is under 2 hours.",
  mirror: "For mirror setup issues, please make sure the device is plugged in and the LED is blue before scanning the QR code. If problems persist, reply here and we'll help you remotely.",
  notification: "You can adjust notification preferences in Settings → Notifications. If you're not receiving pushes, check that notifications are allowed for Reflexion in your phone's Settings app.",
  billing: "For billing and subscription questions, please email billing@reflexion.sg and we'll sort it out within one business day.",
};

function getAutoResponse(text: string): string {
  const lower = text.toLowerCase();
  if (lower.includes('mirror') || lower.includes('device') || lower.includes('qr')) return AUTO_RESPONSES.mirror;
  if (lower.includes('notif') || lower.includes('push') || lower.includes('alert')) return AUTO_RESPONSES.notification;
  if (lower.includes('bill') || lower.includes('payment') || lower.includes('subscri')) return AUTO_RESPONSES.billing;
  return AUTO_RESPONSES.default;
}

function now(): string {
  return new Date().toLocaleTimeString('en-SG', { hour: '2-digit', minute: '2-digit' });
}

export default function ChatbotScreen() {
  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES);
  const [input, setInput] = useState('');
  const listRef = useRef<FlatList>(null);

  function send() {
    const text = input.trim();
    if (!text) return;
    const userMsg: Message = { id: Date.now().toString(), from: 'user', text, time: now() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');

    // Simulate a support response after 1.2s
    setTimeout(() => {
      const reply: Message = {
        id: (Date.now() + 1).toString(),
        from: 'support',
        text: getAutoResponse(text),
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
            onSubmitEditing={send}
            blurOnSubmit={false}
          />
          <TouchableOpacity style={[styles.sendBtn, !input.trim() && styles.sendBtnDisabled]} onPress={send} disabled={!input.trim()}>
            <Text style={styles.sendBtnText}>Send</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
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
