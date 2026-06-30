import React from 'react';
import { PRIORITY_COLORS, CATEGORY_COLORS, STATUS_COLORS } from '../../utils/constants';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'priority' | 'category' | 'status' | 'default';
  level?: string;
  className?: string;
}

export function Badge({ children, variant = 'default', level = '', className = '' }: BadgeProps) {
  let colorClass = 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200';

  if (variant === 'priority' && level) {
    colorClass = PRIORITY_COLORS[level] ?? colorClass;
  } else if (variant === 'category' && level) {
    colorClass = CATEGORY_COLORS[level] ?? colorClass;
  } else if (variant === 'status' && level) {
    colorClass = STATUS_COLORS[level] ?? colorClass;
  }

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium uppercase tracking-wide ${colorClass} ${className}`}
    >
      {children}
    </span>
  );
}
