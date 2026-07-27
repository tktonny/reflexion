import { Tabs } from 'expo-router';
import { Feather } from '@expo/vector-icons';
import React from 'react';
import { Text, type ColorValue } from 'react-native';
import { colors, fontSize, spacing, tabBarContentHeight, tabIconSize } from '../../src/theme';

/**
 * The tab label, rendered here rather than through `tabBarLabelStyle`, and deliberately without a single
 * fixed dimension.
 *
 * The navigator's own label puts the text in an 11px box with `overflow: hidden` while a 12px font needs
 * about 14, so the descender was cut — the "g" in "Settings" lost its tail. Styling that box does not help:
 * a `lineHeight` grows the CONTENT inside the same fixed box and clips more, not less.
 *
 * So: no `numberOfLines` (that is what installs the fixed height and the clip), no `lineHeight`, and no
 * `maxFontSizeMultiplier`. The line box is then whatever the font needs at the reader's own text size, and
 * the bar grows with it. Capping the scale here would be the same mistake in a different place — an older
 * caregiver raising the system font is the case this app exists to serve, not an edge to clamp.
 */
function TabLabel({ children, color }: { children: string; color: ColorValue }) {
  return (
    <Text style={{ color, fontSize: fontSize.caption, fontWeight: '500', textAlign: 'center' }}>
      {children}
    </Text>
  );
}

const iconSize = tabIconSize();

export default function TabLayout() {
  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: colors.accent,
        tabBarInactiveTintColor: colors.tabInactive,
        tabBarStyle: {
          backgroundColor: colors.surface.card,
          borderTopColor: colors.border.default,
          borderTopWidth: 1,
          // Derived from the bar's own content, not typed and not left to the navigator.
          //
          // Leaving it unset does not make the bar adaptive: react-navigation falls back to a fixed 49px
          // default, which fits a 12px label on a 360dp phone and clips the same label on a 744dp tablet —
          // where scaleSize takes the caption to 14 and the icon with it. So the height follows the tokens
          // its content is built from and the reader's font scale, which is what actually tracks.
          minHeight: tabBarContentHeight(),
          paddingTop: spacing.sm,
          paddingBottom: spacing.md,
        },
        headerShown: false,
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: 'Home',
          tabBarLabel: ({ color }) => <TabLabel color={color}>Home</TabLabel>,
          tabBarIcon: ({ color }) => <Feather name="home" size={iconSize} color={color} />,
        }}
      />
      <Tabs.Screen
        name="alerts"
        options={{
          title: 'Notifications',
          tabBarLabel: ({ color }) => <TabLabel color={color}>Notifications</TabLabel>,
          tabBarIcon: ({ color }) => <Feather name="bell" size={iconSize} color={color} />,
        }}
      />
      <Tabs.Screen
        name="settings"
        options={{
          title: 'Settings',
          tabBarLabel: ({ color }) => <TabLabel color={color}>Settings</TabLabel>,
          tabBarIcon: ({ color }) => <Feather name="settings" size={iconSize} color={color} />,
        }}
      />
    </Tabs>
  );
}
