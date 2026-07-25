import React from 'react';
import { Alert, Image, Platform, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { colors, fontSize, MIN_TOUCH_TARGET, radius, spacing } from '../../theme';

export function PhotoInput({
  photoUrl,
  onChange,
}: {
  photoUrl: string;
  onChange: (value: string) => void;
}) {
  async function pickImage() {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Photo access needed', 'Allow photo library access to choose a profile photo.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      aspect: [1, 1],
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.7,
    });

    if (!result.canceled && result.assets[0]?.uri) {
      onChange(result.assets[0].uri);
    }
  }

  if (Platform.OS === 'web') {
    return (
      <View style={styles.uploadBox}>
        {photoUrl ? (
          <Image
            accessibilityLabel="Selected profile photo"
            accessibilityRole="image"
            source={{ uri: photoUrl }}
            style={styles.photoPreview}
          />
        ) : null}
        {React.createElement('input', {
          accept: 'image/*',
          'aria-label': 'Choose a profile photo',
          type: 'file',
          onChange: (event: { target?: { files?: FileList | null } }) => {
            const file = event.target?.files?.[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = () => {
              if (typeof reader.result === 'string') {
                onChange(reader.result);
              }
            };
            reader.readAsDataURL(file);
          },
        })}
        <Text style={styles.uploadText}>
          {photoUrl ? 'Photo selected' : 'Choose a photo for the dashboard card'}
        </Text>
      </View>
    );
  }

  return (
    <View style={styles.uploadBox}>
      {photoUrl ? (
        <Image
          accessibilityLabel="Selected profile photo"
          accessibilityRole="image"
          source={{ uri: photoUrl }}
          style={styles.photoPreview}
        />
      ) : (
        <View style={styles.photoPlaceholder}>
          <Text style={styles.photoPlaceholderText}>No photo selected</Text>
        </View>
      )}
      <TouchableOpacity
        accessibilityRole="button"
        activeOpacity={0.8}
        onPress={() => void pickImage()}
        style={styles.photoButton}
      >
        <Text style={styles.photoButtonText}>{photoUrl ? 'Change photo' : 'Choose photo'}</Text>
      </TouchableOpacity>
      {photoUrl ? (
        <TouchableOpacity
          accessibilityRole="button"
          activeOpacity={0.8}
          onPress={() => onChange('')}
          style={styles.clearPhotoButton}
        >
          <Text style={styles.clearPhotoText}>Remove photo</Text>
        </TouchableOpacity>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  uploadBox: {
    backgroundColor: colors.surface.card,
    borderColor: colors.border.strong,
    borderRadius: radius.sm,
    borderWidth: 1,
    gap: 10,
    padding: 14,
  },
  photoPreview: {
    alignSelf: 'center',
    backgroundColor: colors.surface.muted,
    borderRadius: 12,
    height: 132,
    width: 132,
  },
  photoPlaceholder: {
    alignItems: 'center',
    alignSelf: 'center',
    backgroundColor: colors.surface.muted,
    borderColor: colors.border.default,
    borderRadius: 12,
    borderWidth: 1,
    justifyContent: 'center',
    // minHeight: this box wraps text, so a fixed height clips "No photo selected" at large font sizes.
    minHeight: 132,
    paddingHorizontal: 10,
    width: 132,
  },
  photoPlaceholderText: {
    color: colors.text.tertiary,
    fontSize: fontSize.body,
    fontWeight: '700',
    textAlign: 'center',
  },
  photoButton: {
    alignItems: 'center',
    backgroundColor: colors.accent,
    borderRadius: radius.sm,
    justifyContent: 'center',
    minHeight: MIN_TOUCH_TARGET,
    paddingVertical: 11,
  },
  photoButtonText: {
    color: colors.text.onAccent,
    fontSize: fontSize.body,
    fontWeight: '800',
  },
  clearPhotoButton: {
    alignItems: 'center',
    justifyContent: 'center',
    // Was a ~22pt tap target sitting right under the 44pt primary button — easy to miss, easy to mis-hit.
    minHeight: MIN_TOUCH_TARGET,
    paddingVertical: spacing.xs,
  },
  clearPhotoText: {
    color: colors.accent,
    fontSize: fontSize.caption,
    fontWeight: '800',
  },
  uploadText: {
    color: colors.text.secondary,
    fontSize: fontSize.body,
    fontWeight: '700',
  },
});
