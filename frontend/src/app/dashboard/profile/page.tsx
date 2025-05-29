'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Mic, ArrowLeft, User, Mail, Lock, Save, Eye, EyeOff, Loader2 } from 'lucide-react'
import { api, User as UserType } from '@/lib/api'

export default function ProfilePage() {
    const router = useRouter()
    const [user, setUser] = useState<UserType | null>(null)
    const [isLoading, setIsLoading] = useState(true)
    const [isSaving, setIsSaving] = useState(false)
    const [showCurrentPassword, setShowCurrentPassword] = useState(false)
    const [showNewPassword, setShowNewPassword] = useState(false)
    const [showConfirmPassword, setShowConfirmPassword] = useState(false)
    const [error, setError] = useState('')
    const [success, setSuccess] = useState('')

    const [profileData, setProfileData] = useState({
        email: '',
        currentPassword: '',
        newPassword: '',
        confirmPassword: ''
    })

    useEffect(() => {
        const loadUser = async () => {
            try {
                const userData = await api.getCurrentUser()
                setUser(userData)
                setProfileData(prev => ({ ...prev, email: userData.email }))
            } catch (error) {
                router.push('/auth/login')
            } finally {
                setIsLoading(false)
            }
        }

        loadUser()
    }, [router])

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setProfileData(prev => ({
            ...prev,
            [e.target.name]: e.target.value
        }))
    }

    const handleEmailUpdate = async (e: React.FormEvent) => {
        e.preventDefault()
        setIsSaving(true)
        setError('')
        setSuccess('')

        try {
            const updatedUser = await api.updateEmail({ email: profileData.email })
            setUser(updatedUser)
            setSuccess('Email updated successfully!')
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to update email')
        } finally {
            setIsSaving(false)
        }
    }

    const handlePasswordUpdate = async (e: React.FormEvent) => {
        e.preventDefault()
        setIsSaving(true)
        setError('')
        setSuccess('')

        if (profileData.newPassword !== profileData.confirmPassword) {
            setError('New passwords do not match')
            setIsSaving(false)
            return
        }

        if (profileData.newPassword.length < 6) {
            setError('New password must be at least 6 characters')
            setIsSaving(false)
            return
        }

        try {
            await api.updatePassword({
                current_password: profileData.currentPassword,
                new_password: profileData.newPassword
            })
            setSuccess('Password updated successfully!')
            setProfileData(prev => ({
                ...prev,
                currentPassword: '',
                newPassword: '',
                confirmPassword: ''
            }))
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to update password')
        } finally {
            setIsSaving(false)
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
            <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-white mb-2">Profile Settings</h1>
                    <p className="text-gray-300">Manage your account information and preferences</p>
                </div>

                {error && (
                    <div className="mb-6 bg-red-500/10 border border-red-500/20 rounded-md p-4">
                        <p className="text-red-400">{error}</p>
                    </div>
                )}

                {success && (
                    <div className="mb-6 bg-green-500/10 border border-green-500/20 rounded-md p-4">
                        <p className="text-green-400">{success}</p>
                    </div>
                )}

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Account Overview */}
                    <div className="lg:col-span-1">
                        <Card className="bg-slate-800/50 border-slate-700">
                            <CardHeader>
                                <CardTitle className="text-white flex items-center">
                                    <User className="h-5 w-5 mr-2 text-purple-400" />
                                    Account Overview
                                </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div>
                                    <label className="text-sm text-gray-300">Email</label>
                                    <p className="text-white font-medium">{user.email}</p>
                                </div>
                                <div>
                                    <label className="text-sm text-gray-300">Credits</label>
                                    <p className="text-purple-400 font-semibold text-lg">{user.credits}</p>
                                </div>
                                <div>
                                    <label className="text-sm text-gray-300">Member Since</label>
                                    <p className="text-white">{new Date(user.created_at).toLocaleDateString()}</p>
                                </div>
                                <div>
                                    <label className="text-sm text-gray-300">Account Status</label>
                                    <p className="text-green-400">Active</p>
                                </div>
                            </CardContent>
                        </Card>
                    </div>

                    {/* Profile Settings */}
                    <div className="lg:col-span-2 space-y-6">
                        {/* Email Settings */}
                        <Card className="bg-slate-800/50 border-slate-700">
                            <CardHeader>
                                <CardTitle className="text-white flex items-center">
                                    <Mail className="h-5 w-5 mr-2 text-blue-400" />
                                    Email Settings
                                </CardTitle>
                                <CardDescription className="text-gray-300">
                                    Update your email address
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <form onSubmit={handleEmailUpdate} className="space-y-4">
                                    <div>
                                        <label htmlFor="email" className="block text-sm font-medium text-white mb-2">
                                            Email Address
                                        </label>
                                        <input
                                            id="email"
                                            name="email"
                                            type="email"
                                            required
                                            value={profileData.email}
                                            onChange={handleChange}
                                            className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                            disabled={isSaving}
                                        />
                                    </div>
                                    <Button
                                        type="submit"
                                        className="bg-blue-600 hover:bg-blue-700"
                                        disabled={isSaving || profileData.email === user.email}
                                    >
                                        {isSaving ? (
                                            <>
                                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                                Updating...
                                            </>
                                        ) : (
                                            <>
                                                <Save className="mr-2 h-4 w-4" />
                                                Update Email
                                            </>
                                        )}
                                    </Button>
                                </form>
                            </CardContent>
                        </Card>

                        {/* Password Settings */}
                        <Card className="bg-slate-800/50 border-slate-700">
                            <CardHeader>
                                <CardTitle className="text-white flex items-center">
                                    <Lock className="h-5 w-5 mr-2 text-red-400" />
                                    Password Settings
                                </CardTitle>
                                <CardDescription className="text-gray-300">
                                    Change your account password
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <form onSubmit={handlePasswordUpdate} className="space-y-4">
                                    <div>
                                        <label htmlFor="currentPassword" className="block text-sm font-medium text-white mb-2">
                                            Current Password
                                        </label>
                                        <div className="relative">
                                            <input
                                                id="currentPassword"
                                                name="currentPassword"
                                                type={showCurrentPassword ? 'text' : 'password'}
                                                required
                                                value={profileData.currentPassword}
                                                onChange={handleChange}
                                                className="w-full px-3 py-2 pr-10 bg-slate-700 border border-slate-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                                placeholder="Enter current password"
                                                disabled={isSaving}
                                            />
                                            <button
                                                type="button"
                                                onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                                                className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-white"
                                                disabled={isSaving}
                                            >
                                                {showCurrentPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                                            </button>
                                        </div>
                                    </div>

                                    <div>
                                        <label htmlFor="newPassword" className="block text-sm font-medium text-white mb-2">
                                            New Password
                                        </label>
                                        <div className="relative">
                                            <input
                                                id="newPassword"
                                                name="newPassword"
                                                type={showNewPassword ? 'text' : 'password'}
                                                required
                                                value={profileData.newPassword}
                                                onChange={handleChange}
                                                className="w-full px-3 py-2 pr-10 bg-slate-700 border border-slate-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                                placeholder="Enter new password"
                                                disabled={isSaving}
                                            />
                                            <button
                                                type="button"
                                                onClick={() => setShowNewPassword(!showNewPassword)}
                                                className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-white"
                                                disabled={isSaving}
                                            >
                                                {showNewPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                                            </button>
                                        </div>
                                    </div>

                                    <div>
                                        <label htmlFor="confirmPassword" className="block text-sm font-medium text-white mb-2">
                                            Confirm New Password
                                        </label>
                                        <div className="relative">
                                            <input
                                                id="confirmPassword"
                                                name="confirmPassword"
                                                type={showConfirmPassword ? 'text' : 'password'}
                                                required
                                                value={profileData.confirmPassword}
                                                onChange={handleChange}
                                                className="w-full px-3 py-2 pr-10 bg-slate-700 border border-slate-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                                placeholder="Confirm new password"
                                                disabled={isSaving}
                                            />
                                            <button
                                                type="button"
                                                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                                                className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-white"
                                                disabled={isSaving}
                                            >
                                                {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                                            </button>
                                        </div>
                                    </div>

                                    <Button
                                        type="submit"
                                        className="bg-red-600 hover:bg-red-700"
                                        disabled={isSaving}
                                    >
                                        {isSaving ? (
                                            <>
                                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                                Updating...
                                            </>
                                        ) : (
                                            <>
                                                <Lock className="mr-2 h-4 w-4" />
                                                Update Password
                                            </>
                                        )}
                                    </Button>
                                </form>
                            </CardContent>
                        </Card>
                    </div>
                </div>
            </main>
        </div>
    )
} 