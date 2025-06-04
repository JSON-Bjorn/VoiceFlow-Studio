import React from 'react';

interface ProgressProps {
    value: number;
    className?: string;
    max?: number;
}

export function Progress({ value, className = '', max = 100 }: ProgressProps) {
    const percentage = Math.min(100, Math.max(0, (value / max) * 100));

    return (
        <div className={`w-full bg-gray-200 rounded-full overflow-hidden ${className}`}>
            <div
                className="h-full bg-blue-600 transition-all duration-300 ease-out"
                style={{ width: `${percentage}%` }}
            />
        </div>
    );
} 