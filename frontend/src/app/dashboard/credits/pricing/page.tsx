'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { CheckCircle, CreditCard, ArrowLeft } from 'lucide-react'
import { api } from '@/lib/api'
import { getStripe } from '@/lib/stripe'
import type { CreditBundle } from '@/lib/stripe'

export default function PricingPage() {
    const [bundles, setBundles] = useState<CreditBundle[]>([])
    const [loading, setLoading] = useState(true)
    const [processingBundle, setProcessingBundle] = useState<string | null>(null)
    const router = useRouter()

    useEffect(() => {
        loadBundles()
    }, [])

    const loadBundles = async () => {
        try {
            const data = await api.getCreditBundles()
            setBundles(data)
        } catch (error) {
            console.error('Failed to load credit bundles:', error)
        } finally {
            setLoading(false)
        }
    }

    const handlePurchase = async (bundleId: string) => {
        setProcessingBundle(bundleId)

        try {
            // Create payment intent
            const paymentIntent = await api.createPaymentIntent(bundleId)

            // Get Stripe instance
            const stripe = await getStripe()
            if (!stripe) {
                throw new Error('Stripe failed to load')
            }

            // Redirect to Stripe Checkout or handle payment
            const { error } = await stripe.confirmPayment({
                clientSecret: paymentIntent.client_secret,
                confirmParams: {
                    return_url: `${window.location.origin}/dashboard/credits/success?payment_intent=${paymentIntent.payment_intent_id}`,
                },
            })

            if (error) {
                console.error('Payment failed:', error)
                alert('Payment failed: ' + error.message)
            }
        } catch (error) {
            console.error('Failed to process payment:', error)
            alert('Failed to process payment. Please try again.')
        } finally {
            setProcessingBundle(null)
        }
    }

    if (loading) {
        return (
            <div className="container mx-auto px-4 py-8">
                <div className="flex items-center gap-4 mb-8">
                    <Button variant="ghost" onClick={() => router.back()}>
                        <ArrowLeft className="h-4 w-4 mr-2" />
                        Back
                    </Button>
                    <h1 className="text-3xl font-bold">Purchase Credits</h1>
                </div>
                <div className="text-center">Loading credit bundles...</div>
            </div>
        )
    }

    return (
        <div className="container mx-auto px-4 py-8">
            <div className="flex items-center gap-4 mb-8">
                <Button variant="ghost" onClick={() => router.back()}>
                    <ArrowLeft className="h-4 w-4 mr-2" />
                    Back
                </Button>
                <div>
                    <h1 className="text-3xl font-bold">Purchase Credits</h1>
                    <p className="text-muted-foreground">Choose a credit bundle to continue creating podcasts</p>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {bundles.map((bundle) => (
                    <Card key={bundle.id} className={`relative ${bundle.popular ? 'border-purple-500 shadow-lg' : ''}`}>
                        {bundle.popular && (
                            <Badge className="absolute -top-2 left-1/2 transform -translate-x-1/2 bg-purple-500">
                                Most Popular
                            </Badge>
                        )}

                        <CardHeader className="text-center">
                            <CardTitle className="text-xl">{bundle.name}</CardTitle>
                            <CardDescription>{bundle.description}</CardDescription>
                            <div className="mt-4">
                                <span className="text-3xl font-bold">{bundle.price_formatted}</span>
                                <div className="text-sm text-muted-foreground mt-1">
                                    {bundle.credits} credits
                                </div>
                                <div className="text-xs text-muted-foreground">
                                    ${(bundle.price / bundle.credits / 100).toFixed(2)} per credit
                                </div>
                            </div>
                        </CardHeader>

                        <CardContent>
                            <div className="space-y-2">
                                <div className="flex items-center gap-2">
                                    <CheckCircle className="h-4 w-4 text-green-500" />
                                    <span className="text-sm">Generate {bundle.credits} podcasts</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <CheckCircle className="h-4 w-4 text-green-500" />
                                    <span className="text-sm">High-quality AI voices</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <CheckCircle className="h-4 w-4 text-green-500" />
                                    <span className="text-sm">Instant download</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <CheckCircle className="h-4 w-4 text-green-500" />
                                    <span className="text-sm">No expiration</span>
                                </div>
                            </div>
                        </CardContent>

                        <CardFooter>
                            <Button
                                className="w-full"
                                onClick={() => handlePurchase(bundle.id)}
                                disabled={processingBundle === bundle.id}
                                variant={bundle.popular ? "default" : "outline"}
                            >
                                {processingBundle === bundle.id ? (
                                    'Processing...'
                                ) : (
                                    <>
                                        <CreditCard className="h-4 w-4 mr-2" />
                                        Purchase Now
                                    </>
                                )}
                            </Button>
                        </CardFooter>
                    </Card>
                ))}
            </div>

            <div className="mt-12 text-center">
                <div className="bg-muted/50 rounded-lg p-6 max-w-2xl mx-auto">
                    <h3 className="font-semibold mb-2">Secure Payment</h3>
                    <p className="text-sm text-muted-foreground">
                        All payments are processed securely through Stripe. We never store your payment information.
                        Credits never expire and can be used anytime to generate podcasts.
                    </p>
                </div>
            </div>
        </div>
    )
} 