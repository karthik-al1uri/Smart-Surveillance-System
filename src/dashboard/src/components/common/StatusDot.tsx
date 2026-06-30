import React from 'react';

type Status = 'online' | 'warning' | 'offline' | 'disabled';

interface StatusDotProps {
  status: Status;
  label?: string;
}

const DOT_CLASSES: Record<Status, string> = {
  online: 'bg-green-500 animate-pulse',
  warning: 'bg-yellow-500',
  offline: 'bg-red-500',
  disabled: 'bg-gray-400',
};

export function StatusDot({ status, label }: StatusDotProps) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={`inline-block w-2 h-2 rounded-full ${DOT_CLASSES[status]}`} />
      {label && <span className="text-sm text-gray-600 dark:text-gray-400">{label}</span>}
    </span>
  );
}
