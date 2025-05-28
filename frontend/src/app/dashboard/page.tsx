'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { CreditBalance } from '@/components/ui/credit-balance'
import { Mic, Plus, User, CreditCard, History, LogOut, Settings } from 'lucide-react'
import { apiClient, User as UserType, CreditSummary } from '@/lib/api'

export default function DashboardPage() {
    const router = useRouter()
    const [user, setUser] = useState<UserType | null>(null)
    const [creditSummary, setCreditSummary] = useState<CreditSummary | null>(null)
    const [isLoading, setIsLoading] = useState(true)

    useEffect(() => {
        const loadUserData = async () => {
            try {
                const userData = await apiClient.getCurrentUser()
                setUser(userData)

                // Load credit summary
                const summary = await apiClient.getCreditSummary()
                setCreditSummary(summary)
            } catch (error) {
                // If we can't get user data, redirect to login
                router.push('/auth/login')
            } finally {
                setIsLoading(false)
            }
        }

        loadUserData()
    }, [router])

    const handleLogout = () => {
        apiClient.logout()
        router.push('/')
    }

    const handleBuyCredits = () => {
        router.push('/dashboard/credits')
    }

    if (isLoading) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
                <div className="text-white">Loading...</div>
            </div>
        )
    }

    if (!user) {
        return null // Will redirect to login
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
            {/* Header */}
            <header className="border-b border-slate-700 bg-slate-800/50">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between items-center h-16">
                        <div className="flex items-center space-x-2">
                            <Mic className="h-8 w-8 text-purple-400" />
                            <span className="text-xl font-bold text-white">VoiceFlow Studio</span>
                        </div>

                        <div className="flex items-center space-x-4">
                            <CreditBalance
                                credits={user.credits}
                                variant="header"
                            />
                            <Link href="/dashboard/profile">
                                <Button
                                    variant="outline"
                                    size="sm"
                                    className="border-slate-600 text-white hover:bg-slate-700"
                                >
                                    <Settings className="h-4 w-4 mr-2" />
                                    Profile
                                </Button>
                            </Link>
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={handleLogout}
                                className="border-slate-600 text-white hover:bg-slate-700"
                            >
                                <LogOut className="h-4 w-4 mr-2" />
                                Logout
                            </Button>
                        </div>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-white mb-2">
                        Welcome back, {user.email}!
                    </h1>
                    <p className="text-gray-300">
                        Ready to create your next AI-powered podcast?
                    </p>
                </div>

                {/* Credit Balance Card */}
                <div className="mb-8">
                    <CreditBalance
                        credits={user.credits}
                        variant="compact"
                        showBuyButton={true}
                        onBuyCredits={handleBuyCredits}
                    />
                </div>

                {/* Quick Actions */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                    <Card className="bg-slate-800/50 border-slate-700 hover:bg-slate-800/70 transition-colors cursor-pointer">
                        <CardHeader>
                            <CardTitle className="text-white flex items-center">
                                <Plus className="h-5 w-5 mr-2 text-purple-400" />
                                Create New Podcast
                            </CardTitle>
                            <CardDescription className="text-gray-300">
                                Generate a new AI podcast from your topic
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <Button
                                className="w-full bg-purple-600 hover:bg-purple-700"
                                disabled={user.credits === 0}
                            >
                                {user.credits === 0 ? 'No Credits' : 'Start Creating'}
                            </Button>
                        </CardContent>
                    </Card>

                    <Card className="bg-slate-800/50 border-slate-700 hover:bg-slate-800/70 transition-colors cursor-pointer">
                        <CardHeader>
                            <CardTitle className="text-white flex items-center">
                                <CreditCard className="h-5 w-5 mr-2 text-green-400" />
                                Buy Credits
                            </CardTitle>
                            <CardDescription className="text-gray-300">
                                Purchase more credits to create podcasts
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <Button
                                variant="outline"
                                className="w-full border-slate-600 text-white hover:bg-slate-700"
                                onClick={handleBuyCredits}
                            >
                                View Plans
                            </Button>
                        </CardContent>
                    </Card>

                    <Link href="/dashboard/library">
                        <Card className="bg-slate-800/50 border-slate-700 hover:bg-slate-800/70 transition-colors cursor-pointer h-full">
                            <CardHeader>
                                <CardTitle className="text-white flex items-center">
                                    <History className="h-5 w-5 mr-2 text-blue-400" />
                                    Podcast History
                                </CardTitle>
                                <CardDescription className="text-gray-300">
                                    View and manage your created podcasts
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <Button variant="outline" className="w-full border-slate-600 text-white hover:bg-slate-700">
                                    View History
                                </Button>
                            </CardContent>
                        </Card>
                    </Link>

                    <Link href="/dashboard/profile">
                        <Card className="bg-slate-800/50 border-slate-700 hover:bg-slate-800/70 transition-colors cursor-pointer h-full">
                            <CardHeader>
                                <CardTitle className="text-white flex items-center">
                                    <Settings className="h-5 w-5 mr-2 text-orange-400" />
                                    Profile Settings
                                </CardTitle>
                                <CardDescription className="text-gray-300">
                                    Manage your account and preferences
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <Button variant="outline" className="w-full border-slate-600 text-white hover:bg-slate-700">
                                    Manage Profile
                                </Button>
                            </CardContent>
                        </Card>
                    </Link>
                </div>

                {/* Stats and Recent Activity */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Account Stats */}
                    <div className="space-y-6">
                        <Card className="bg-slate-800/50 border-slate-700">
                            <CardHeader>
                                <CardTitle className="text-white text-lg">Account Status</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-2">
                                    <div className="flex justify-between">
                                        <span className="text-gray-300">Status:</span>
                                        <span className="text-green-400">Active</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-300">Member since:</span>
                                        <span className="text-white">
                                            {new Date(user.created_at).toLocaleDateString()}
                                        </span>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>

                        {creditSummary && (
                            <Card className="bg-slate-800/50 border-slate-700">
                                <CardHeader>
                                    <CardTitle className="text-white text-lg">Credit Summary</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-2">
                                        <div className="flex justify-between">
                                            <span className="text-gray-300">Available:</span>
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
                    </div>

                    {/* Recent Credit Activity */}
                    {creditSummary && creditSummary.recent_transactions.length > 0 && (
                        <Card className="bg-slate-800/50 border-slate-700">
                            <CardHeader>
                                <CardTitle className="text-white text-lg">Recent Credit Activity</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-3">
                                    {creditSummary.recent_transactions.map((transaction) => (
                                        <div key={transaction.id} className="flex justify-between items-center py-2 border-b border-slate-700 last:border-b-0">
                                            <div>
                                                <p className="text-white text-sm">{transaction.description}</p>
                                                <p className="text-gray-400 text-xs">
                                                    {new Date(transaction.created_at).toLocaleDateString()}
                                                </p>
                                            </div>
                                            <span className={`font-semibold ${transaction.amount > 0 ? 'text-green-400' : 'text-red-400'
                                                }`}>
                                                {transaction.amount > 0 ? '+' : ''}{transaction.amount}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </CardContent>
                        </Card>
                    )}
                </div>
            </main>
        </div>
    )
} 