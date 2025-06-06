'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { CreditBalance } from '@/components/ui/credit-balance'
import { Mic, Plus, User, CreditCard, History, LogOut, Settings } from 'lucide-react'
import { api, User as UserType, CreditSummary } from '@/lib/api'

export default function DashboardPage() {
    const router = useRouter()
    const [user, setUser] = useState<UserType | null>(null)
    const [creditSummary, setCreditSummary] = useState<CreditSummary | null>(null)
    const [isLoading, setIsLoading] = useState(true)

    useEffect(() => {
        const loadUserData = async () => {
            try {
                console.log('Dashboard loading, checking authentication...')

                // Check localStorage directly
                const directToken = localStorage.getItem('access_token')
                console.log('Direct localStorage check - token exists:', directToken ? 'Yes' : 'No')
                console.log('Direct token length:', directToken?.length || 0)

                // Ensure we have the token from localStorage
                api.refreshTokenFromStorage()

                // Check if we have a token before making API calls
                if (!api.hasToken()) {
                    console.log('No authentication token found, redirecting to login')
                    router.push('/auth/login')
                    return
                }

                console.log('Token found, making API call to /api/users/me')
                const userData = await api.getCurrentUser()
                console.log('User data received:', userData)
                setUser(userData)

                // Load credit summary
                const summary = await api.getCreditSummary()
                setCreditSummary(summary)
            } catch (error) {
                // If we can't get user data, redirect to login
                console.error('Dashboard auth error:', error)
                router.push('/auth/login')
            } finally {
                setIsLoading(false)
            }
        }

        loadUserData()
    }, [router])

    const handleLogout = () => {
        api.logout()
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
                                    className="border-orange-400 text-orange-400 hover:bg-orange-400 hover:text-slate-900"
                                >
                                    <Settings className="h-4 w-4 mr-2" />
                                    Profile
                                </Button>
                            </Link>
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={handleLogout}
                                className="border-red-400 text-red-400 hover:bg-red-400 hover:text-slate-900"
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
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
                    <Link href="/dashboard/library">
                        <Card className="bg-slate-800/50 border-slate-700 hover:bg-slate-800/70 transition-colors cursor-pointer h-full">
                            <CardHeader>
                                <CardTitle className="text-white flex items-center">
                                    <History className="h-5 w-5 mr-2 text-purple-400" />
                                    My Podcasts
                                </CardTitle>
                                <CardDescription className="text-gray-300">
                                    View, manage, and create new AI-powered podcasts
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <Button variant="outline" className="w-full border-purple-500 text-purple-500 hover:bg-purple-500 hover:text-slate-900 font-semibold">
                                    Open Studio
                                </Button>
                            </CardContent>
                        </Card>
                    </Link>

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
                                className="w-full border-green-400 text-green-400 hover:bg-green-400 hover:text-slate-900"
                                onClick={handleBuyCredits}
                            >
                                View Plans
                            </Button>
                        </CardContent>
                    </Card>

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
                                <Button variant="outline" className="w-full border-orange-400 text-orange-400 hover:bg-orange-400 hover:text-slate-900">
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