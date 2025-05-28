'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { CreditBalance } from '@/components/ui/credit-balance'
import { Mic, ArrowLeft, CreditCard, Plus, History } from 'lucide-react'
import { apiClient, User as UserType, CreditSummary, CreditTransaction } from '@/lib/api'

export default function CreditsPage() {
    const router = useRouter()
    const [user, setUser] = useState<UserType | null>(null)
    const [creditSummary, setCreditSummary] = useState<CreditSummary | null>(null)
    const [transactions, setTransactions] = useState<CreditTransaction[]>([])
    const [isLoading, setIsLoading] = useState(true)
    const [isPurchasing, setIsPurchasing] = useState(false)

    useEffect(() => {
        const loadData = async () => {
            try {
                const userData = await apiClient.getCurrentUser()
                setUser(userData)

                const summary = await apiClient.getCreditSummary()
                setCreditSummary(summary)

                const transactionHistory = await apiClient.getCreditTransactions(20)
                setTransactions(transactionHistory)
            } catch (error) {
                router.push('/auth/login')
            } finally {
                setIsLoading(false)
            }
        }

        loadData()
    }, [router])

    const handlePurchase = async (amount: number) => {
        if (!user) return

        setIsPurchasing(true)
        try {
            await apiClient.purchaseCredits({
                amount,
                payment_reference: `test_purchase_${Date.now()}`
            })

            // Reload data
            const userData = await apiClient.getCurrentUser()
            setUser(userData)

            const summary = await apiClient.getCreditSummary()
            setCreditSummary(summary)

            const transactionHistory = await apiClient.getCreditTransactions(20)
            setTransactions(transactionHistory)
        } catch (error) {
            console.error('Purchase failed:', error)
        } finally {
            setIsPurchasing(false)
        }
    }

    const handleTestUsage = async () => {
        if (!user) return

        try {
            await apiClient.useCredits({
                description: "Test podcast generation",
                amount: 1
            })

            // Reload data
            const userData = await apiClient.getCurrentUser()
            setUser(userData)

            const summary = await apiClient.getCreditSummary()
            setCreditSummary(summary)

            const transactionHistory = await apiClient.getCreditTransactions(20)
            setTransactions(transactionHistory)
        } catch (error) {
            console.error('Usage failed:', error)
        }
    }

    if (isLoading) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
                <div className="text-white">Loading...</div>
            </div>
        )
    }

    if (!user) {
        return null
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
            {/* Header */}
            <header className="border-b border-slate-700 bg-slate-800/50">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between items-center h-16">
                        <div className="flex items-center space-x-4">
                            <Link href="/dashboard" className="flex items-center space-x-2 text-gray-300 hover:text-white">
                                <ArrowLeft className="h-5 w-5" />
                                <span>Back to Dashboard</span>
                            </Link>
                        </div>

                        <div className="flex items-center space-x-2">
                            <Mic className="h-8 w-8 text-purple-400" />
                            <span className="text-xl font-bold text-white">VoiceFlow Studio</span>
                        </div>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-white mb-2">Credit Management</h1>
                    <p className="text-gray-300">Purchase credits and view your transaction history</p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Credit Balance and Purchase */}
                    <div className="lg:col-span-2 space-y-6">
                        {/* Current Balance */}
                        <CreditBalance
                            credits={user.credits}
                            variant="card"
                            showBuyButton={false}
                        />

                        {/* Purchase Options */}
                        <Card className="bg-slate-800/50 border-slate-700">
                            <CardHeader>
                                <CardTitle className="text-white flex items-center">
                                    <CreditCard className="h-5 w-5 mr-2 text-green-400" />
                                    Purchase Credits
                                </CardTitle>
                                <CardDescription className="text-gray-300">
                                    Buy credits to create more AI podcasts
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                    {[
                                        { credits: 1, price: 5, popular: false },
                                        { credits: 5, price: 20, popular: true },
                                        { credits: 10, price: 35, popular: false },
                                    ].map((plan, index) => (
                                        <Card key={index} className={`bg-slate-700/50 border-slate-600 ${plan.popular ? 'ring-2 ring-purple-400' : ''}`}>
                                            <CardHeader className="text-center pb-2">
                                                {plan.popular && (
                                                    <div className="bg-purple-600 text-white text-xs font-medium px-2 py-1 rounded-full mb-2 inline-block">
                                                        Most Popular
                                                    </div>
                                                )}
                                                <CardTitle className="text-white text-lg">{plan.credits} Credit{plan.credits > 1 ? 's' : ''}</CardTitle>
                                                <div className="text-2xl font-bold text-white">${plan.price}</div>
                                                <CardDescription className="text-gray-300 text-sm">
                                                    ${plan.price / plan.credits} per podcast
                                                </CardDescription>
                                            </CardHeader>
                                            <CardContent className="pt-2">
                                                <Button
                                                    className="w-full bg-purple-600 hover:bg-purple-700"
                                                    onClick={() => handlePurchase(plan.credits)}
                                                    disabled={isPurchasing}
                                                >
                                                    {isPurchasing ? 'Processing...' : 'Purchase'}
                                                </Button>
                                            </CardContent>
                                        </Card>
                                    ))}
                                </div>
                            </CardContent>
                        </Card>

                        {/* Test Usage */}
                        <Card className="bg-slate-800/50 border-slate-700">
                            <CardHeader>
                                <CardTitle className="text-white">Test Credit Usage</CardTitle>
                                <CardDescription className="text-gray-300">
                                    Test the credit system by simulating podcast generation
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <Button
                                    onClick={handleTestUsage}
                                    disabled={user.credits === 0}
                                    className="bg-blue-600 hover:bg-blue-700"
                                >
                                    <Plus className="h-4 w-4 mr-2" />
                                    Use 1 Credit (Test)
                                </Button>
                            </CardContent>
                        </Card>
                    </div>

                    {/* Credit Summary and History */}
                    <div className="space-y-6">
                        {/* Credit Summary */}
                        {creditSummary && (
                            <Card className="bg-slate-800/50 border-slate-700">
                                <CardHeader>
                                    <CardTitle className="text-white text-lg">Credit Summary</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-3">
                                        <div className="flex justify-between">
                                            <span className="text-gray-300">Current Balance:</span>
                                            <span className="text-purple-400 font-semibold">{creditSummary.current_balance}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-300">Total Purchased:</span>
                                            <span className="text-white">{creditSummary.total_purchased}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-300">Total Used:</span>
                                            <span className="text-white">{creditSummary.total_used}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-300">Bonus Credits:</span>
                                            <span className="text-green-400">{creditSummary.total_bonus}</span>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        )}

                        {/* Transaction History */}
                        <Card className="bg-slate-800/50 border-slate-700">
                            <CardHeader>
                                <CardTitle className="text-white flex items-center">
                                    <History className="h-5 w-5 mr-2 text-blue-400" />
                                    Transaction History
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-3 max-h-96 overflow-y-auto">
                                    {transactions.length > 0 ? (
                                        transactions.map((transaction) => (
                                            <div key={transaction.id} className="flex justify-between items-start py-2 border-b border-slate-700 last:border-b-0">
                                                <div className="flex-1">
                                                    <p className="text-white text-sm">{transaction.description}</p>
                                                    <p className="text-gray-400 text-xs">
                                                        {new Date(transaction.created_at).toLocaleString()}
                                                    </p>
                                                    <p className="text-gray-500 text-xs capitalize">
                                                        {transaction.transaction_type}
                                                    </p>
                                                </div>
                                                <span className={`font-semibold text-sm ${transaction.amount > 0 ? 'text-green-400' : 'text-red-400'
                                                    }`}>
                                                    {transaction.amount > 0 ? '+' : ''}{transaction.amount}
                                                </span>
                                            </div>
                                        ))
                                    ) : (
                                        <p className="text-gray-400 text-sm">No transactions yet</p>
                                    )}
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </div>
            </main>
        </div>
    )
} 