import { Ionicons } from '@expo/vector-icons'
import { router, useLocalSearchParams } from 'expo-router'
import { useEffect } from 'react'
import { Pressable, StyleSheet, Text, View } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'

export default function ConversationClosingScreen() {
  const params = useLocalSearchParams<{ nurseName?: string }>()
  const nurseName = typeof params.nurseName === 'string' && params.nurseName.trim()
    ? params.nurseName.trim()
    : 'your nurse'

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      router.replace('/conversation')
    }, 5000)

    return () => clearTimeout(timeoutId)
  }, [])

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.stage}>
        <Pressable style={styles.card} onPress={() => router.replace('/conversation')}>
          <Text style={styles.message}>Thank you for chatting with me.</Text>
          <Text style={styles.message}>I’ll let {nurseName} know you checked in today.</Text>
          <Text style={styles.message}>Have a good day!</Text>
          <View style={styles.smileCircle}>
            <Ionicons name="happy-outline" size={58} color="#8E7F6D" />
          </View>
        </Pressable>
      </View>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  safeArea: {
    backgroundColor: '#FFF9F1',
    flex: 1,
  },
  stage: {
    alignItems: 'center',
    backgroundColor: '#FFF9F1',
    flex: 1,
    justifyContent: 'center',
    padding: 18,
  },
  card: {
    alignItems: 'center',
    backgroundColor: '#FFFBF4',
    borderColor: '#F1E5D2',
    borderRadius: 8,
    borderWidth: 1,
    gap: 28,
    height: '100%',
    justifyContent: 'center',
    maxHeight: 760,
    maxWidth: 430,
    minHeight: 620,
    padding: 34,
    shadowColor: '#D8C6A8',
    shadowOffset: { height: 10, width: 0 },
    shadowOpacity: 0.22,
    shadowRadius: 22,
    width: '100%',
  },
  message: {
    color: '#282828',
    fontSize: 22,
    fontWeight: '800',
    lineHeight: 32,
    textAlign: 'center',
  },
  smileCircle: {
    alignItems: 'center',
    borderColor: '#8E7F6D',
    borderRadius: 48,
    borderWidth: 2,
    height: 96,
    justifyContent: 'center',
    marginTop: 12,
    width: 96,
  },
})
