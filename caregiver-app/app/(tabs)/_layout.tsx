import { Tabs } from 'expo-router';
import { Feather } from '@expo/vector-icons';
import React from 'react';
import { Text, type ColorValue } from 'react-native';
import { colors, fontSize, spacing, tabBarContentHeight, tabIconSize } from '../../src/theme';

function TabLabel({ children, color }: { children: string; color: ColorValue }) {
  return (
    <Text style={{ color, fontSize: fontSize.caption, fontWeight: '600', textAlign: 'center' }}>
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
          minHeight: Math.max(64, tabBarContentHeight()),
          paddingTop: spacing.xs,
          paddingBottom: spacing.xs,
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
        name="activity"
        options={{
          title: 'Activity',
          tabBarLabel: ({ color }) => <TabLabel color={color}>Activity</TabLabel>,
          tabBarIcon: ({ color }) => <Feather name="clock" size={iconSize} color={color} />,
        }}
      />
      <Tabs.Screen
        name="chat"
        options={{
          title: 'Chat',
          tabBarLabel: ({ color }) => <TabLabel color={color}>Chat</TabLabel>,
          tabBarIcon: ({ color }) => <Feather name="message-circle" size={iconSize} color={color} />,
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
