import { Tabs } from 'expo-router';
import { Feather } from '@expo/vector-icons';
import { colors, fontSize, tabBar } from '../../src/theme';

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
          // Geometry lives in the theme so the scrolling screens can reserve exactly this much room.
          // minHeight, not height: a fixed bar clips its own labels once the system font size is raised,
          // which is the first setting an older caregiver changes.
          minHeight: tabBar.minHeight,
          paddingBottom: tabBar.paddingBottom,
          paddingTop: tabBar.paddingTop,
        },
        tabBarLabelStyle: {
          fontSize: fontSize.caption,
          fontWeight: '500',
          // Explicit, because the default line box is tight enough to clip a descender — the "g" in
          // "Settings" was losing its tail. 1.4x leaves room for it at any font scale.
          lineHeight: Math.round(fontSize.caption * 1.4),
        },
        headerShown: false,
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: 'Home',
          tabBarIcon: ({ color, size }) => <Feather name="home" size={21} color={color} />,
        }}
      />
      <Tabs.Screen
        name="alerts"
        options={{
          title: 'Notifications',
          tabBarIcon: ({ color }) => <Feather name="bell" size={21} color={color} />,
        }}
      />
      <Tabs.Screen
        name="settings"
        options={{
          title: 'Settings',
          tabBarIcon: ({ color }) => <Feather name="settings" size={21} color={color} />,
        }}
      />
    </Tabs>
  );
}
