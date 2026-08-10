import * as Haptics from 'expo-haptics';
import { Feather } from '@expo/vector-icons';
import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import {
  AccessibilityInfo,
  Animated,
  Modal,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  type GestureResponderEvent,
  type PressableProps,
  type StyleProp,
  type ViewStyle,
} from 'react-native';

import { colors, fontSize, radius, spacing } from '../theme';

export type HapticKind = 'selection' | 'success' | 'warning' | 'error';
export type MotionFeedback = 'button' | 'card' | 'none';

export const motionTokens = {
  press: {
    buttonScale: 0.975,
    cardScale: 0.988,
    buttonOpacity: 0.86,
    cardOpacity: 0.9,
  },
  spring: {
    stiffness: 420,
    damping: 38,
    mass: 1,
  },
  reducedDuration: 90,
  contentDuration: 180,
} as const;

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);
const ReducedMotionContext = createContext<boolean | undefined>(undefined);
const animatedUseNativeDriver = Platform.OS !== 'web';

function useSystemReducedMotion() {
  const [reducedMotion, setReducedMotion] = useState(false);

  useEffect(() => {
    let active = true;
    void AccessibilityInfo.isReduceMotionEnabled()
      .then((enabled) => {
        if (active) setReducedMotion(enabled);
      })
      .catch(() => undefined);

    const subscription = AccessibilityInfo.addEventListener('reduceMotionChanged', setReducedMotion);
    return () => {
      active = false;
      subscription.remove();
    };
  }, []);

  return reducedMotion;
}

export function MotionProvider({ children }: { children: React.ReactNode }) {
  const reducedMotion = useSystemReducedMotion();
  return <ReducedMotionContext.Provider value={reducedMotion}>{children}</ReducedMotionContext.Provider>;
}

export function useReducedMotion() {
  const provided = useContext(ReducedMotionContext);
  const [local, setLocal] = useState(false);

  useEffect(() => {
    if (provided !== undefined) return;
    let active = true;
    void AccessibilityInfo.isReduceMotionEnabled()
      .then((enabled) => {
        if (active) setLocal(enabled);
      })
      .catch(() => undefined);
    const subscription = AccessibilityInfo.addEventListener('reduceMotionChanged', setLocal);
    return () => {
      active = false;
      subscription.remove();
    };
  }, [provided]);

  return provided ?? local;
}

export function triggerHaptic(kind: HapticKind) {
  if (Platform.OS === 'web') return;
  const task = kind === 'selection'
    ? Haptics.selectionAsync()
    : kind === 'success'
      ? Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success)
      : kind === 'warning'
        ? Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning)
        : Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
  void task.catch(() => undefined);
}

export type MotionPressableProps = Omit<PressableProps, 'style' | 'onPressIn' | 'onPressOut'> & {
  feedback?: MotionFeedback;
  haptic?: HapticKind;
  onPressIn?: PressableProps['onPressIn'];
  onPressOut?: PressableProps['onPressOut'];
  style?: StyleProp<ViewStyle>;
};

/**
 * A native-driver press surface. Feedback begins on touch-down and the action still
 * commits on the normal Pressable release path, so motion never delays navigation.
 */
export function MotionPressable({
  feedback = 'button',
  haptic,
  onPress,
  onPressIn,
  onPressOut,
  disabled,
  style,
  ...props
}: MotionPressableProps) {
  const reducedMotion = useReducedMotion();
  const scale = useRef(new Animated.Value(1)).current;
  const opacity = useRef(new Animated.Value(1)).current;

  const animateValue = useCallback((value: Animated.Value, toValue: number) => {
    value.stopAnimation();
    if (reducedMotion || feedback === 'none') {
      Animated.timing(value, {
        duration: motionTokens.reducedDuration,
        easing: undefined,
        toValue,
        useNativeDriver: animatedUseNativeDriver,
      }).start();
      return;
    }
    Animated.spring(value, {
      damping: motionTokens.spring.damping,
      mass: motionTokens.spring.mass,
      overshootClamping: true,
      stiffness: motionTokens.spring.stiffness,
      toValue,
      useNativeDriver: animatedUseNativeDriver,
    }).start();
  }, [feedback, reducedMotion]);

  const handlePressIn = (event: GestureResponderEvent) => {
    if (!disabled && feedback !== 'none') {
      const targetScale = feedback === 'card' ? motionTokens.press.cardScale : motionTokens.press.buttonScale;
      const targetOpacity = feedback === 'card' ? motionTokens.press.cardOpacity : motionTokens.press.buttonOpacity;
      animateValue(scale, targetScale);
      animateValue(opacity, targetOpacity);
    }
    onPressIn?.(event);
  };

  const handlePressOut = (event: GestureResponderEvent) => {
    if (feedback !== 'none') {
      animateValue(scale, 1);
      animateValue(opacity, 1);
    }
    onPressOut?.(event);
  };

  return (
    <AnimatedPressable
      {...props}
      accessibilityState={disabled ? { ...(props.accessibilityState || {}), disabled: true } : props.accessibilityState}
      disabled={disabled}
      onPress={(event) => {
        if (haptic && !disabled) triggerHaptic(haptic);
        onPress?.(event);
      }}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      style={[style, { opacity, transform: [{ scale }] }]}
    />
  );
}

export function MotionFadeIn({ children, delay = 0, playKey = 0, style }: { children: React.ReactNode; delay?: number; playKey?: string | number; style?: StyleProp<ViewStyle> }) {
  const reducedMotion = useReducedMotion();
  const opacity = useRef(new Animated.Value(reducedMotion ? 1 : 0)).current;
  const translateY = useRef(new Animated.Value(reducedMotion ? 0 : 8)).current;

  useEffect(() => {
    opacity.stopAnimation();
    translateY.stopAnimation();
    opacity.setValue(reducedMotion ? 1 : 0);
    translateY.setValue(reducedMotion ? 0 : 8);
    Animated.parallel([
      Animated.timing(opacity, { delay, duration: reducedMotion ? motionTokens.reducedDuration : motionTokens.contentDuration, toValue: 1, useNativeDriver: animatedUseNativeDriver }),
      Animated.timing(translateY, { delay, duration: reducedMotion ? motionTokens.reducedDuration : motionTokens.contentDuration, toValue: 0, useNativeDriver: animatedUseNativeDriver }),
    ]).start();
  }, [delay, opacity, playKey, reducedMotion, translateY]);

  return <Animated.View style={[style, { opacity, transform: [{ translateY }] }]}>{children}</Animated.View>;
}

export function MotionDisclosure({ open, children, style }: { open: boolean; children: React.ReactNode; style?: StyleProp<ViewStyle> }) {
  const reducedMotion = useReducedMotion();
  const progress = useRef(new Animated.Value(open ? 1 : 0)).current;
  const [contentHeight, setContentHeight] = useState(0);

  useEffect(() => {
    progress.stopAnimation();
    if (reducedMotion) {
      Animated.timing(progress, { duration: motionTokens.reducedDuration, toValue: open ? 1 : 0, useNativeDriver: false }).start();
      return;
    }
    Animated.spring(progress, {
      damping: motionTokens.spring.damping,
      mass: motionTokens.spring.mass,
      overshootClamping: true,
      stiffness: 320,
      toValue: open ? 1 : 0,
      useNativeDriver: false,
    }).start();
  }, [open, progress, reducedMotion]);

  const height = progress.interpolate({ inputRange: [0, 1], outputRange: [0, contentHeight] });
  const opacity = progress.interpolate({ inputRange: [0, 0.2, 1], outputRange: [0, 0.3, 1] });

  return (
    <Animated.View pointerEvents={open ? 'auto' : 'none'} style={[styles.disclosure, style, { height, opacity }]}>
      <Animated.View onLayout={(event) => setContentHeight(event.nativeEvent.layout.height)}>{children}</Animated.View>
    </Animated.View>
  );
}

export function MotionToggle({ label, value, onValueChange, disabled = false }: { label: string; value: boolean; onValueChange: (value: boolean) => void; disabled?: boolean }) {
  const reducedMotion = useReducedMotion();
  const progress = useRef(new Animated.Value(value ? 1 : 0)).current;

  useEffect(() => {
    progress.stopAnimation();
    if (reducedMotion) {
      Animated.timing(progress, { duration: motionTokens.reducedDuration, toValue: value ? 1 : 0, useNativeDriver: animatedUseNativeDriver }).start();
      return;
    }
    Animated.spring(progress, {
      damping: motionTokens.spring.damping,
      mass: motionTokens.spring.mass,
      overshootClamping: true,
      stiffness: motionTokens.spring.stiffness,
      toValue: value ? 1 : 0,
      useNativeDriver: animatedUseNativeDriver,
    }).start();
  }, [progress, reducedMotion, value]);

  const knobX = progress.interpolate({ inputRange: [0, 1], outputRange: [2, 22] });
  return (
    <MotionPressable
      accessibilityLabel={label}
      accessibilityRole="switch"
      accessibilityState={{ checked: value, disabled }}
      disabled={disabled}
      feedback="none"
      haptic="selection"
      onPress={() => onValueChange(!value)}
      style={[styles.toggleTrack, value && styles.toggleTrackOn]}
    >
      <Animated.View style={[styles.toggleKnob, { transform: [{ translateX: knobX }] }]} />
    </MotionPressable>
  );
}

export function MotionCheckmark({ visible, color = colors.accent, size = 21 }: { visible: boolean; color?: string; size?: number }) {
  const reducedMotion = useReducedMotion();
  const progress = useRef(new Animated.Value(visible ? 1 : 0)).current;

  useEffect(() => {
    progress.stopAnimation();
    if (reducedMotion) {
      Animated.timing(progress, { duration: motionTokens.reducedDuration, toValue: visible ? 1 : 0, useNativeDriver: animatedUseNativeDriver }).start();
      return;
    }
    Animated.spring(progress, {
      damping: motionTokens.spring.damping,
      mass: motionTokens.spring.mass,
      overshootClamping: true,
      stiffness: motionTokens.spring.stiffness,
      toValue: visible ? 1 : 0,
      useNativeDriver: animatedUseNativeDriver,
    }).start();
  }, [progress, reducedMotion, visible]);

  const scale = progress.interpolate({ inputRange: [0, 1], outputRange: [0.72, 1] });
  return <Animated.View style={{ opacity: progress, transform: [{ scale }], width: size }}><Feather color={color} name="check" size={size} /></Animated.View>;
}

export function MotionSheet({ visible, title, children, onClose }: { visible: boolean; title: string; children: React.ReactNode; onClose: () => void }) {
  const reducedMotion = useReducedMotion();
  const [mounted, setMounted] = useState(visible);
  const progress = useRef(new Animated.Value(visible ? 1 : 0)).current;

  const animate = useCallback((toValue: number, finished?: () => void) => {
    progress.stopAnimation();
    const animation = reducedMotion
      ? Animated.timing(progress, { duration: motionTokens.reducedDuration, toValue, useNativeDriver: animatedUseNativeDriver })
      : Animated.spring(progress, { damping: 34, mass: 1, overshootClamping: true, stiffness: 360, toValue, useNativeDriver: animatedUseNativeDriver });
    animation.start(({ finished: didFinish }) => {
      if (didFinish) finished?.();
    });
  }, [progress, reducedMotion]);

  useEffect(() => {
    if (visible) {
      setMounted(true);
      requestAnimationFrame(() => animate(1));
    } else if (mounted) {
      animate(0, () => setMounted(false));
    }
  }, [animate, mounted, visible]);

  if (!mounted) return null;
  const translateY = progress.interpolate({ inputRange: [0, 1], outputRange: [460, 0] });
  const backdropOpacity = progress.interpolate({ inputRange: [0, 1], outputRange: [0, 0.28] });

  return (
    <Modal accessibilityViewIsModal onRequestClose={onClose} transparent visible>
      <Pressable accessibilityLabel="Close sheet" onPress={onClose} style={styles.sheetBackdrop}>
        <Animated.View pointerEvents="none" style={[styles.sheetScrim, { opacity: backdropOpacity }]} />
        <Animated.View onStartShouldSetResponder={() => true} style={[styles.sheet, { transform: [{ translateY }] }]}>
          <Text accessibilityRole="header" style={styles.sheetTitle}>{title}</Text>
          {children}
        </Animated.View>
      </Pressable>
    </Modal>
  );
}

const styles = StyleSheet.create({
  disclosure: { overflow: 'hidden' },
  toggleTrack: { alignItems: 'flex-start', backgroundColor: '#D8DEDB', borderRadius: radius.pill, height: 32, justifyContent: 'center', width: 54 },
  toggleTrackOn: { backgroundColor: colors.accent },
  toggleKnob: { backgroundColor: colors.surface.card, borderRadius: radius.pill, elevation: 1, height: 28, shadowColor: '#16324A', shadowOpacity: 0.12, shadowRadius: 3, width: 28 },
  sheetBackdrop: { backgroundColor: 'transparent', flex: 1, justifyContent: 'flex-end' },
  sheetScrim: { backgroundColor: '#16324A', bottom: 0, left: 0, position: 'absolute', right: 0, top: 0 },
  sheet: { backgroundColor: colors.surface.card, borderTopLeftRadius: radius.xl, borderTopRightRadius: radius.xl, gap: spacing.md, maxHeight: '88%', minWidth: 0, padding: spacing.xl, paddingBottom: spacing.xxl },
  sheetTitle: { color: colors.text.primary, fontSize: fontSize.heading, fontWeight: '700', lineHeight: 26 },
});
