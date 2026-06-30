import { useAppStore } from '../../stores/appStore';
import { XMarkIcon } from '@heroicons/react/20/solid';
import {
  CheckCircleIcon,
  ExclamationCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
} from '@heroicons/react/24/outline';

const TOAST_STYLES = {
  success: { bg: 'bg-green-50 dark:bg-green-900/50 border-green-200 dark:border-green-700', icon: CheckCircleIcon, iconColor: 'text-green-500' },
  error: { bg: 'bg-red-50 dark:bg-red-900/50 border-red-200 dark:border-red-700', icon: ExclamationCircleIcon, iconColor: 'text-red-500' },
  warning: { bg: 'bg-yellow-50 dark:bg-yellow-900/50 border-yellow-200 dark:border-yellow-700', icon: ExclamationTriangleIcon, iconColor: 'text-yellow-500' },
  info: { bg: 'bg-blue-50 dark:bg-blue-900/50 border-blue-200 dark:border-blue-700', icon: InformationCircleIcon, iconColor: 'text-blue-500' },
};

export function ToastContainer() {
  const { toasts, removeToast } = useAppStore();

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 w-80">
      {toasts.map((toast) => {
        const style = TOAST_STYLES[toast.type];
        const Icon = style.icon;
        return (
          <div
            key={toast.id}
            className={`flex items-start gap-3 p-3 rounded-lg border shadow-md ${style.bg}`}
          >
            <Icon className={`w-5 h-5 flex-shrink-0 mt-0.5 ${style.iconColor}`} />
            <p className="flex-1 text-sm text-gray-800 dark:text-gray-200">{toast.message}</p>
            <button onClick={() => removeToast(toast.id)}>
              <XMarkIcon className="w-4 h-4 text-gray-400 hover:text-gray-600" />
            </button>
          </div>
        );
      })}
    </div>
  );
}
