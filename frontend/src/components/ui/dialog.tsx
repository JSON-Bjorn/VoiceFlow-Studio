"use client"

import React, { ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';

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

interface DialogFooterProps {
    children: ReactNode;
    className?: string;
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
            className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 backdrop-blur-sm"
            onClick={handleBackdropClick}
        >
            {children}
        </div>,
        document.body
    );
}

export function DialogTrigger({ asChild, children, onClick }: DialogTriggerProps) {
    if (asChild && React.isValidElement(children)) {
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
        <div className={`relative bg-slate-800 border border-slate-700 rounded-lg shadow-xl max-h-[90vh] overflow-y-auto ${className}`}>
            {children}
        </div>
    );
}

export function DialogHeader({ children }: DialogHeaderProps) {
    return (
        <div className="px-6 py-4 border-b border-slate-700">
            {children}
        </div>
    );
}

export function DialogTitle({ className = '', children }: DialogTitleProps) {
    return (
        <h2 className={`text-lg font-semibold text-white ${className}`}>
            {children}
        </h2>
    );
}

export function DialogDescription({ className = '', children }: DialogDescriptionProps) {
    return (
        <p className={`text-sm text-slate-300 mt-2 ${className}`}>
            {children}
        </p>
    );
}

export function DialogFooter({ children, className = '' }: DialogFooterProps) {
    return (
        <div className={`px-6 py-4 border-t border-slate-700 flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2 ${className}`}>
            {children}
        </div>
    );
} 