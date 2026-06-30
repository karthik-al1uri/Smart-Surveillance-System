interface PaginationProps {
  total: number;
  limit: number;
  offset: number;
  onChange: (offset: number) => void;
}

export function Pagination({ total, limit, offset, onChange }: PaginationProps) {
  const totalPages = Math.ceil(total / limit);
  const currentPage = Math.floor(offset / limit) + 1;

  if (totalPages <= 1) return null;

  const pages = Array.from({ length: totalPages }, (_, i) => i + 1).filter(
    (p) => p === 1 || p === totalPages || Math.abs(p - currentPage) <= 2
  );

  return (
    <div className="flex items-center justify-between py-3">
      <p className="text-sm text-gray-500 dark:text-gray-400">
        Showing {offset + 1}–{Math.min(offset + limit, total)} of {total}
      </p>
      <div className="flex items-center gap-1">
        <button
          disabled={currentPage === 1}
          onClick={() => onChange(Math.max(0, offset - limit))}
          className="px-3 py-1 text-sm rounded border border-gray-300 dark:border-gray-600 disabled:opacity-40 hover:bg-gray-50 dark:hover:bg-gray-700"
        >
          ←
        </button>
        {pages.map((p, i) => {
          const prevP = pages[i - 1];
          return (
            <span key={p} className="flex items-center gap-1">
              {prevP && p - prevP > 1 && (
                <span className="px-1 text-gray-400">…</span>
              )}
              <button
                onClick={() => onChange((p - 1) * limit)}
                className={`px-3 py-1 text-sm rounded border ${
                  p === currentPage
                    ? 'bg-indigo-600 text-white border-indigo-600'
                    : 'border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'
                }`}
              >
                {p}
              </button>
            </span>
          );
        })}
        <button
          disabled={currentPage === totalPages}
          onClick={() => onChange(offset + limit)}
          className="px-3 py-1 text-sm rounded border border-gray-300 dark:border-gray-600 disabled:opacity-40 hover:bg-gray-50 dark:hover:bg-gray-700"
        >
          →
        </button>
      </div>
    </div>
  );
}
