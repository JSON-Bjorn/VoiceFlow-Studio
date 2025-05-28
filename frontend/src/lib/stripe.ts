import { loadStripe, Stripe } from '@stripe/stripe-js';

let stripePromise: Promise<Stripe | null>;

export const getStripe = () => {
    if (!stripePromise) {
        // We'll get the publishable key from the backend
        stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY || '');
    }
    return stripePromise;
};

export interface CreditBundle {
    id: string;
    name: string;
    credits: number;
    price: number;
    price_formatted: string;
    description: string;
    popular: boolean;
}

export interface PaymentIntentResponse {
    client_secret: string;
    payment_intent_id: string;
    amount: number;
    currency: string;
    status: string;
    credits: number;
    bundle_name: string;
}

export interface PaymentStatus {
    payment_intent_id: string;
    status: string;
    amount: number;
    currency: string;
    credits: number;
} 