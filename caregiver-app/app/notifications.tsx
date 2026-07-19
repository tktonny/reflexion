import { Feather } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React from 'react';
import {
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

export default function NotificationTestScreen() {
  const router = useRouter();

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.header}>
        <TouchableOpacity style={styles.backButton} onPress={() => router.back()}>
          <Feather name="chevron-left" size={24} color="#87566A" />
        </TouchableOpacity>
        <View style={styles.headerTextBlock}>
          <Text style={styles.eyebrow}>Notifications</Text>
          <Text style={styles.title}>Coming soon</Text>
        </View>
      </View>

      <View style={styles.content}>
        <View style={styles.iconWrap}>
          <Feather name="bell" size={32} color="#87566A" />
        </View>
        <Text style={styles.messageTitle}>Bear with us</Text>
        <Text style={styles.messageText}>
          Notification testing requires a development build on Android. This page is disabled in Expo Go for now.
        </Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#F8F3EC' },
  header: { alignItems: 'center', flexDirection: 'row', gap: 12, padding: 20, paddingBottom: 8 },
  backButton: {
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 22,
    borderWidth: 1,
    height: 44,
    justifyContent: 'center',
    width: 44,
  },
  headerTextBlock: { flex: 1 },
  eyebrow: {
    color: '#A69C92',
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 0.8,
    textTransform: 'uppercase',
  },
  title: { color: '#2B2522', fontFamily: 'Georgia', fontSize: 28, fontWeight: '500', marginTop: 4 },
  content: {
    alignItems: 'center',
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: 28,
  },
  iconWrap: {
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderColor: '#E7DED2',
    borderRadius: 36,
    borderWidth: 1,
    height: 72,
    justifyContent: 'center',
    marginBottom: 18,
    width: 72,
  },
  messageTitle: { color: '#2B2522', fontFamily: 'Georgia', fontSize: 24, fontWeight: '500', marginBottom: 8 },
  messageText: { color: '#756C64', fontSize: 15, lineHeight: 22, maxWidth: 320, textAlign: 'center' },
});
