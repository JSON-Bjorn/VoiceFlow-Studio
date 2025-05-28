import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { CreditCard, AlertTriangle, Plus } from 'lucide-react'
import { cn } from '@/lib/utils'

interface CreditBalanceProps {
    credits: number
    variant?: 'compact' | 'card' | 'header'
    showBuyButton?: boolean
    onBuyCredits?: () => void
    className?: string
}

export function CreditBalance({
    credits,
    variant = 'compact',
    showBuyButton = false,
    onBuyCredits,
    className
}: CreditBalanceProps) {
    const isLowCredits = credits <= 2
    const isOutOfCredits = credits === 0

    if (variant === 'header') {
        return (
            <div className={cn("flex items-center space-x-2", className)}>
                <div className="text-white">
                    <span className="text-sm text-gray-300">Credits: </span>
                    <span className={cn(
                        "font-semibold",
                        isOutOfCredits ? "text-red-400" :
                            isLowCredits ? "text-yellow-400" :
                                "text-purple-400"
                    )}>
                        {credits}
                    </span>
                </div>
                {isLowCredits && (
                    <AlertTriangle className="h-4 w-4 text-yellow-400" />
                )}
            </div>
        )
    }

    if (variant === 'compact') {
        return (
            <div className={cn("flex items-center justify-between p-3 bg-slate-800/50 border border-slate-700 rounded-lg", className)}>
                <div className="flex items-center space-x-2">
                    <CreditCard className={cn(
                        "h-5 w-5",
                        isOutOfCredits ? "text-red-400" :
                            isLowCredits ? "text-yellow-400" :
                                "text-purple-400"
                    )} />
                    <div>
                        <p className="text-white font-medium">{credits} Credits</p>
                        {isLowCredits && (
                            <p className="text-xs text-yellow-400">
                                {isOutOfCredits ? "No credits remaining" : "Low credits"}
                            </p>
                        )}
                    </div>
                </div>
                {showBuyButton && (
                    <Button
                        size="sm"
                        onClick={onBuyCredits}
                        className="bg-purple-600 hover:bg-purple-700"
                    >
                        <Plus className="h-4 w-4 mr-1" />
                        Buy
                    </Button>
                )}
            </div>
        )
    }

    // Card variant
    return (
        <Card className={cn("bg-slate-800/50 border-slate-700", className)}>
            <CardHeader>
                <CardTitle className="text-white flex items-center justify-between">
                    <span className="flex items-center">
                        <CreditCard className={cn(
                            "h-5 w-5 mr-2",
                            isOutOfCredits ? "text-red-400" :
                                isLowCredits ? "text-yellow-400" :
                                    "text-purple-400"
                        )} />
                        Credits
                    </span>
                    {isLowCredits && (
                        <AlertTriangle className="h-5 w-5 text-yellow-400" />
                    )}
                </CardTitle>
            </CardHeader>
            <CardContent>
                <div className="space-y-3">
                    <div className="flex justify-between items-center">
                        <span className="text-gray-300">Available:</span>
                        <span className={cn(
                            "font-semibold text-lg",
                            isOutOfCredits ? "text-red-400" :
                                isLowCredits ? "text-yellow-400" :
                                    "text-purple-400"
                        )}>
                            {credits}
                        </span>
                    </div>

                    {isLowCredits && (
                        <div className={cn(
                            "p-3 rounded-md border",
                            isOutOfCredits
                                ? "bg-red-500/10 border-red-500/20"
                                : "bg-yellow-500/10 border-yellow-500/20"
                        )}>
                            <p className={cn(
                                "text-sm",
                                isOutOfCredits ? "text-red-400" : "text-yellow-400"
                            )}>
                                {isOutOfCredits
                                    ? "You're out of credits! Purchase more to create podcasts."
                                    : "You're running low on credits. Consider purchasing more."
                                }
                            </p>
                        </div>
                    )}

                    {showBuyButton && (
                        <Button
                            onClick={onBuyCredits}
                            className="w-full bg-purple-600 hover:bg-purple-700"
                        >
                            <Plus className="h-4 w-4 mr-2" />
                            Buy More Credits
                        </Button>
                    )}
                </div>
            </CardContent>
        </Card>
    )
} 