import React, { useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { FAQ_ITEMS } from '../src/data/mockData';

export default function FAQScreen() {
  const router = useRouter();
  const [open, setOpen] = useState<number | null>(0);

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.intro}>
          Everything you need to know about Reflexion and Aria.
        </Text>

        {FAQ_ITEMS.map((item, i) => (
          <TouchableOpacity
            key={i}
            style={[styles.card, open === i && styles.cardOpen]}
            onPress={() => setOpen(open === i ? null : i)}
            activeOpacity={0.8}
          >
            <View style={styles.question}>
              <Text style={styles.questionText}>{item.q}</Text>
              <Text style={styles.chevron}>{open === i ? '▲' : '▼'}</Text>
            </View>
            {open === i && (
              <Text style={styles.answerText}>{item.a}</Text>
            )}
          </TouchableOpacity>
        ))}

        <TouchableOpacity
          activeOpacity={0.82}
          onPress={() => router.push('/chatbot')}
          style={styles.bottomCard}
        >
          <Text style={styles.bottomTitle}>Still have questions?</Text>
          <Text style={styles.bottomText}>Click here to chat with the Reflexion team.</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#F7F9FC' },
  content: { padding: 20, paddingBottom: 48 },
  intro: { fontSize: 15, color: '#666', marginBottom: 20, lineHeight: 22 },
  card: {
    backgroundColor: '#fff', borderRadius: 16, padding: 16, marginBottom: 10,
    borderWidth: 1.5, borderColor: '#F0F0F0',
    shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.04, shadowRadius: 3, elevation: 1,
  },
  cardOpen: { borderColor: '#1A6FA8' },
  question: { flexDirection: 'row', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8 },
  questionText: { fontSize: 15, fontWeight: '700', color: '#1A1A2E', flex: 1, lineHeight: 22 },
  chevron: { fontSize: 12, color: '#1A6FA8', marginTop: 4 },
  answerText: { fontSize: 14, color: '#555', marginTop: 12, lineHeight: 22 },
  bottomCard: {
    backgroundColor: '#EEF6FC', borderRadius: 16, padding: 20, marginTop: 8, alignItems: 'center',
  },
  bottomTitle: { fontSize: 16, fontWeight: '700', color: '#1A6FA8', marginBottom: 6 },
  bottomText: { fontSize: 14, color: '#555', textAlign: 'center', lineHeight: 20 },
});
