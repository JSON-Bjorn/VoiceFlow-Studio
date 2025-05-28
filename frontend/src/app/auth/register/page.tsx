'use client'

import { useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Mic, Eye, EyeOff, Check, Loader2 } from 'lucide-react'
import { apiClient } from '@/lib/api'

export default function RegisterPage() {
    const router = useRouter()
    const [showPassword, setShowPassword] = useState(false)
    const [showConfirmPassword, setShowConfirmPassword] = useState(false)
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState('')
    const [success, setSuccess] = useState('')
    const [formData, setFormData] = useState({
        email: '',
        password: '',
        confirmPassword: '',
        acceptTerms: false
    })

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setIsLoading(true)
        setError('')
        setSuccess('')

        if (formData.password !== formData.confirmPassword) {
            setError('Passwords do not match')
            setIsLoading(false)
            return
        }

        if (!formData.acceptTerms) {
            setError('Please accept the terms and conditions')
            setIsLoading(false)
            return
        }

        try {
            await apiClient.register({
                email: formData.email,
                password: formData.password
            })

            setSuccess('Account created successfully! Please sign in.')
            // Redirect to login after a short delay
            setTimeout(() => {
                router.push('/auth/login')
            }, 2000)
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Registration failed')
        } finally {
            setIsLoading(false)
        }
    }

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, type, checked, value } = e.target
        setFormData(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : value
        }))
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center p-4">
            <div className="w-full max-w-md">
                {/* Logo */}
                <div className="text-center mb-8">
                    <Link href="/" className="inline-flex items-center space-x-2">
                        <Mic className="h-8 w-8 text-purple-400" />
                        <span className="text-2xl font-bold text-white">VoiceFlow Studio</span>
                    </Link>
                </div>

                {/* Register Card */}
                <Card className="bg-slate-800/50 border-slate-700">
                    <CardHeader className="text-center">
                        <CardTitle className="text-2xl text-white">Create Account</CardTitle>
                        <CardDescription className="text-gray-300">
                            Join VoiceFlow Studio and start creating AI podcasts
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <form onSubmit={handleSubmit} className="space-y-4">
                            {error && (
                                <div className="bg-red-500/10 border border-red-500/20 rounded-md p-3">
                                    <p className="text-red-400 text-sm">{error}</p>
                                </div>
                            )}

                            {success && (
                                <div className="bg-green-500/10 border border-green-500/20 rounded-md p-3">
                                    <p className="text-green-400 text-sm">{success}</p>
                                </div>
                            )}

                            <div className="space-y-2">
                                <label htmlFor="email" className="text-sm font-medium text-white">
                                    Email
                                </label>
                                <input
                                    id="email"
                                    name="email"
                                    type="email"
                                    required
                                    value={formData.email}
                                    onChange={handleChange}
                                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                    placeholder="Enter your email"
                                    disabled={isLoading}
                                />
                            </div>

                            <div className="space-y-2">
                                <label htmlFor="password" className="text-sm font-medium text-white">
                                    Password
                                </label>
                                <div className="relative">
                                    <input
                                        id="password"
                                        name="password"
                                        type={showPassword ? 'text' : 'password'}
                                        required
                                        value={formData.password}
                                        onChange={handleChange}
                                        className="w-full px-3 py-2 pr-10 bg-slate-700 border border-slate-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                        placeholder="Create a password"
                                        disabled={isLoading}
                                    />
                                    <button
                                        type="button"
                                        onClick={() => setShowPassword(!showPassword)}
                                        className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-white"
                                        disabled={isLoading}
                                    >
                                        {showPassword ? (
                                            <EyeOff className="h-4 w-4" />
                                        ) : (
                                            <Eye className="h-4 w-4" />
                                        )}
                                    </button>
                                </div>
                                <p className="text-xs text-gray-400">
                                    Must be at least 8 characters with uppercase, lowercase, and number
                                </p>
                            </div>

                            <div className="space-y-2">
                                <label htmlFor="confirmPassword" className="text-sm font-medium text-white">
                                    Confirm Password
                                </label>
                                <div className="relative">
                                    <input
                                        id="confirmPassword"
                                        name="confirmPassword"
                                        type={showConfirmPassword ? 'text' : 'password'}
                                        required
                                        value={formData.confirmPassword}
                                        onChange={handleChange}
                                        className="w-full px-3 py-2 pr-10 bg-slate-700 border border-slate-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                        placeholder="Confirm your password"
                                        disabled={isLoading}
                                    />
                                    <button
                                        type="button"
                                        onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                                        className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-white"
                                        disabled={isLoading}
                                    >
                                        {showConfirmPassword ? (
                                            <EyeOff className="h-4 w-4" />
                                        ) : (
                                            <Eye className="h-4 w-4" />
                                        )}
                                    </button>
                                </div>
                            </div>

                            <div className="flex items-start space-x-2">
                                <div className="relative">
                                    <input
                                        id="acceptTerms"
                                        name="acceptTerms"
                                        type="checkbox"
                                        required
                                        checked={formData.acceptTerms}
                                        onChange={handleChange}
                                        className="sr-only"
                                        disabled={isLoading}
                                    />
                                    <div
                                        className={`w-4 h-4 rounded border-2 flex items-center justify-center cursor-pointer ${formData.acceptTerms
                                                ? 'bg-purple-600 border-purple-600'
                                                : 'border-slate-600 bg-slate-700'
                                            } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                                        onClick={() => !isLoading && setFormData(prev => ({ ...prev, acceptTerms: !prev.acceptTerms }))}
                                    >
                                        {formData.acceptTerms && <Check className="h-3 w-3 text-white" />}
                                    </div>
                                </div>
                                <label htmlFor="acceptTerms" className="text-sm text-gray-300 leading-4">
                                    I agree to the{' '}
                                    <Link href="/terms" className="text-purple-400 hover:text-purple-300">
                                        Terms of Service
                                    </Link>{' '}
                                    and{' '}
                                    <Link href="/privacy" className="text-purple-400 hover:text-purple-300">
                                        Privacy Policy
                                    </Link>
                                </label>
                            </div>

                            <Button
                                type="submit"
                                className="w-full bg-purple-600 hover:bg-purple-700"
                                disabled={isLoading}
                            >
                                {isLoading ? (
                                    <>
                                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                        Creating Account...
                                    </>
                                ) : (
                                    'Create Account'
                                )}
                            </Button>
                        </form>

                        <div className="mt-6 text-center">
                            <p className="text-gray-300">
                                Already have an account?{' '}
                                <Link href="/auth/login" className="text-purple-400 hover:text-purple-300 font-medium">
                                    Sign in
                                </Link>
                            </p>
                        </div>
                    </CardContent>
                </Card>

                {/* Welcome Bonus */}
                <div className="mt-4 text-center">
                    <p className="text-sm text-green-400">
                        🎉 Get 1 free podcast credit when you sign up!
                    </p>
                </div>
            </div>
        </div>
    )
} 