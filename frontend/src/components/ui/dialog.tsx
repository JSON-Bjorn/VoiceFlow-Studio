import React, { ReactNode } from 'react';
import { createPortal } from 'react-dom';

interface DialogProps {
    open: boolean;
    onOpenChange?: (open: boolean) => void;
    children: ReactNode;
}

interface DialogContentProps {
    className?: string;
    children: ReactNode;
}

interface DialogHeaderProps {
    children: ReactNode;
}

interface DialogTitleProps {
    className?: string;
    children: ReactNode;
}

interface DialogDescriptionProps {
    className?: string;
    children: ReactNode;
}

interface DialogTriggerProps {
    asChild?: boolean;
    children: ReactNode;
    onClick?: () => void;
}

export function Dialog({ open, onOpenChange, children }: DialogProps) {
    if (!open) return null;

    const handleBackdropClick = (e: React.MouseEvent) => {
        if (e.target === e.currentTarget) {
            onOpenChange?.(false);
        }
    };

    return createPortal(
        <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50"
            onClick={handleBackdropClick}
        >
            {children}
        </div>,
        document.body
    );
}

export function DialogTrigger({ asChild, children, onClick }: DialogTriggerProps) {
    if (asChild && React.isValidElement(children)) {
        // For asChild, we expect the parent to handle the onClick
        return children;
    }

    return (
        <button onClick={onClick}>
            {children}
        </button>
    );
}

export function DialogContent({ className = '', children }: DialogContentProps) {
    return (
        <div className={`bg-white rounded-lg shadow-xl max-h-[90vh] overflow-y-auto ${className}`}>
            {children}
        </div>
    );
}

export function DialogHeader({ children }: DialogHeaderProps) {
    return (
        <div className="px-6 py-4 border-b border-gray-200">
            {children}
        </div>
    );
}

export function DialogTitle({ className = '', children }: DialogTitleProps) {
    return (
        <h2 className={`text-lg font-semibold text-gray-900 ${className}`}>
            {children}
        </h2>
    );
}

export function DialogDescription({ className = '', children }: DialogDescriptionProps) {
    return (
        <p className={`text-sm text-gray-600 ${className}`}>
            {children}
        </p>
    );
} 