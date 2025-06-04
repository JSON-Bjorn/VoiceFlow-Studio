const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface User {
    id: number
    email: string
    credits: number
    is_active: boolean
    created_at: string
}

export interface LoginRequest {
    email: string
    password: string
}

export interface RegisterRequest {
    email: string
    password: string
}

export interface UpdateEmailRequest {
    email: string
}

export interface UpdatePasswordRequest {
    current_password: string
    new_password: string
}

export interface AuthResponse {
    access_token: string
    token_type: string
}

export interface CreditTransaction {
    id: number
    user_id: number
    amount: number
    transaction_type: 'purchase' | 'usage' | 'refund' | 'bonus'
    description: string | null
    reference_id: string | null
    created_at: string
}

export interface CreditSummary {
    current_balance: number
    total_purchased: number
    total_used: number
    total_bonus: number
    recent_transactions: CreditTransaction[]
}

export interface CreditUsage {
    description: string
    amount?: number
}

export interface CreditPurchase {
    amount: number
    payment_reference?: string
}

// Podcast types
export interface Podcast {
    id: number
    user_id: number
    title: string
    topic: string
    length: number
    status: 'pending' | 'generating' | 'completed' | 'failed'
    audio_url?: string
    script?: string
    created_at: string
    updated_at?: string
}

export interface PodcastCreate {
    title: string
    topic: string
    length: number
}

export interface PodcastListResponse {
    podcasts: Podcast[]
    total: number
    page: number
    per_page: number
    total_pages: number
}

export interface PodcastSummary {
    total_podcasts: number
    completed_podcasts: number
    pending_podcasts: number
    failed_podcasts: number
}

class ApiClient {
    private baseUrl: string
    private token: string | null = null

    constructor(baseUrl: string = API_BASE_URL) {
        this.baseUrl = baseUrl
        // Try to get token from localStorage on initialization
        if (typeof window !== 'undefined') {
            this.token = localStorage.getItem('access_token')
        }
    }

    setToken(token: string) {
        this.token = token
        if (typeof window !== 'undefined') {
            localStorage.setItem('access_token', token)
        }
    }

    clearToken() {
        this.token = null
        if (typeof window !== 'undefined') {
            localStorage.removeItem('access_token')
        }
    }

    async request<T>(
        endpoint: string,
        options: RequestInit = {}
    ): Promise<T> {
        const url = `${this.baseUrl}${endpoint}`

        const headers: Record<string, string> = {
            'Content-Type': 'application/json',
            ...(options.headers as Record<string, string>),
        }

        if (this.token) {
            headers.Authorization = `Bearer ${this.token}`
        }

        const response = await fetch(url, {
            ...options,
            headers,
        })

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}))
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
        }

        return response.json()
    }

    // Authentication endpoints
    async login(credentials: LoginRequest): Promise<AuthResponse> {
        const response = await this.request<AuthResponse>('/api/auth/login', {
            method: 'POST',
            body: JSON.stringify(credentials),
        })

        this.setToken(response.access_token)
        return response
    }

    async register(userData: RegisterRequest): Promise<User> {
        return this.request<User>('/api/auth/register', {
            method: 'POST',
            body: JSON.stringify(userData),
        })
    }

    async getCurrentUser(): Promise<User> {
        return this.request<User>('/api/users/me')
    }

    // Profile management endpoints
    async updateEmail(emailData: UpdateEmailRequest): Promise<User> {
        return this.request<User>('/api/users/me/email', {
            method: 'PUT',
            body: JSON.stringify(emailData),
        })
    }

    async updatePassword(passwordData: UpdatePasswordRequest): Promise<User> {
        return this.request<User>('/api/users/me/password', {
            method: 'PUT',
            body: JSON.stringify(passwordData),
        })
    }

    // Credit management endpoints
    async getCreditSummary(): Promise<CreditSummary> {
        return this.request<CreditSummary>('/api/credits/summary')
    }

    async getCreditTransactions(limit: number = 10): Promise<CreditTransaction[]> {
        return this.request<CreditTransaction[]>(`/api/credits/transactions?limit=${limit}`)
    }

    async useCredits(usageData: CreditUsage): Promise<CreditTransaction> {
        return this.request<CreditTransaction>('/api/credits/use', {
            method: 'POST',
            body: JSON.stringify(usageData),
        })
    }

    async purchaseCredits(purchaseData: CreditPurchase): Promise<CreditTransaction> {
        return this.request<CreditTransaction>('/api/credits/purchase', {
            method: 'POST',
            body: JSON.stringify(purchaseData),
        })
    }

    async canAffordCredits(amount: number): Promise<{ can_afford: boolean; current_balance: number; required: number }> {
        return this.request(`/api/credits/can-afford/${amount}`)
    }

    logout() {
        this.clearToken()
    }

    // Podcast API functions
    async createPodcast(data: PodcastCreate): Promise<Podcast> {
        const response = await fetch(`${this.baseUrl}/api/podcasts/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.token}`,
            },
            body: JSON.stringify(data),
        })

        if (!response.ok) {
            const error = await response.json()
            throw new Error(error.detail || 'Failed to create podcast')
        }

        return response.json()
    }

    async getPodcasts(
        page: number = 1,
        perPage: number = 10,
        status?: string
    ): Promise<PodcastListResponse> {
        const params = new URLSearchParams({
            page: page.toString(),
            per_page: perPage.toString(),
        })

        if (status) {
            params.append('status', status)
        }

        const response = await fetch(`${this.baseUrl}/api/podcasts/?${params}`, {
            headers: {
                'Authorization': `Bearer ${this.token}`,
            },
        })

        if (!response.ok) {
            throw new Error('Failed to fetch podcasts')
        }

        return response.json()
    }

    async getPodcastSummary(): Promise<PodcastSummary> {
        const response = await fetch(`${this.baseUrl}/api/podcasts/summary`, {
            headers: {
                'Authorization': `Bearer ${this.token}`,
            },
        })

        if (!response.ok) {
            throw new Error('Failed to fetch podcast summary')
        }

        return response.json()
    }

    async getRecentPodcasts(limit: number = 5): Promise<Podcast[]> {
        const response = await fetch(`${this.baseUrl}/api/podcasts/recent?limit=${limit}`, {
            headers: {
                'Authorization': `Bearer ${this.token}`,
            },
        })

        if (!response.ok) {
            throw new Error('Failed to fetch recent podcasts')
        }

        return response.json()
    }

    async getPodcastById(id: number): Promise<Podcast> {
        const response = await fetch(`${this.baseUrl}/api/podcasts/${id}`, {
            headers: {
                'Authorization': `Bearer ${this.token}`,
            },
        })

        if (!response.ok) {
            throw new Error('Failed to fetch podcast')
        }

        return response.json()
    }

    async deletePodcast(id: number): Promise<void> {
        const response = await fetch(`${this.baseUrl}/api/podcasts/${id}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${this.token}`,
            },
        })

        if (!response.ok) {
            throw new Error('Failed to delete podcast')
        }
    }

    async simulatePodcastGeneration(id: number): Promise<Podcast> {
        return this.request<Podcast>(`/api/podcasts/${id}/generate`, {
            method: 'POST',
        })
    }

    // Enhanced AI Pipeline with Voice Generation
    async generatePodcastWithAI(id: number, options?: {
        generateVoice?: boolean;
        hostPersonalities?: {
            host_1?: { name: string; personality: string; voice_id?: string; role?: string };
            host_2?: { name: string; personality: string; voice_id?: string; role?: string };
        };
        stylePreferences?: {
            tone?: string;
            complexity?: string;
            humor_level?: string;
            pacing?: string;
        };
        contentPreferences?: {
            focus_areas?: string[];
            avoid_topics?: string[];
            target_audience?: string;
        };
    }): Promise<{
        success: boolean;
        generation_id?: string;
        message: string;
        result?: any;
        voice_generated?: boolean;
        total_duration?: number;
        cost_estimate?: number;
    }> {
        const requestBody = {
            podcast_id: id,
            custom_settings: {
                generate_voice: options?.generateVoice || false,
                hosts: options?.hostPersonalities || {
                    host_1: {
                        name: "Felix",
                        personality: "analytical, thoughtful, engaging",
                        role: "primary_questioner"
                    },
                    host_2: {
                        name: "Bjorn",
                        personality: "warm, curious, conversational",
                        role: "storyteller"
                    }
                },
                style_preferences: options?.stylePreferences || {
                    tone: "conversational",
                    complexity: "accessible",
                    humor_level: "light",
                    pacing: "moderate"
                },
                content_preferences: options?.contentPreferences || {
                    focus_areas: ["practical applications", "recent developments"],
                    target_audience: "general public"
                }
            }
        };

        return this.request<{
            success: boolean;
            generation_id?: string;
            message: string;
            result?: any;
            voice_generated?: boolean;
            total_duration?: number;
            cost_estimate?: number;
        }>(`/api/ai-pipeline/enhanced/generate/podcast`, {
            method: 'POST',
            body: JSON.stringify(requestBody),
        });
    }

    // Voice generation APIs (Updated for Chatterbox)
    async getVoiceProfiles(): Promise<{
        voices: Array<{
            voice_id: string;
            name: string;
            description: string;
            gender: string;
            style: string;
            is_custom: boolean;
        }>;
    }> {
        return this.request<{
            voices: Array<{
                voice_id: string;
                name: string;
                description: string;
                gender: string;
                style: string;
                is_custom: boolean;
            }>;
        }>(`/api/chatterbox/voices`);
    }

    async estimateVoiceCost(text: string): Promise<{
        character_count: number;
        estimated_processing_time: number;
        computational_cost: string;
        api_cost: number;
        total_cost: number;
    }> {
        return this.request<{
            character_count: number;
            estimated_processing_time: number;
            computational_cost: string;
            api_cost: number;
            total_cost: number;
        }>(`/api/chatterbox/estimate-cost`, {
            method: 'POST',
            body: JSON.stringify({ text }),
        });
    }

    // Additional Chatterbox methods
    async testChatterboxConnection(): Promise<{
        status: string;
        message: string;
        details: any;
    }> {
        return this.request(`/api/chatterbox/test`);
    }

    async getChatterboxHealth(): Promise<{
        status: string;
        service: string;
        details: any;
    }> {
        return this.request(`/api/chatterbox/health`);
    }

    async getPodcastVoiceProfiles(): Promise<{
        voices: Record<string, {
            id: string;
            name: string;
            role: string;
            personality: string;
            voice_settings: any;
        }>;
    }> {
        return this.request(`/api/chatterbox/podcast-voices`);
    }

    async generateTTSAudio(data: {
        text: string;
        voice_id?: string;
        audio_prompt_path?: string;
        speed?: number;
        stability?: number;
        similarity_boost?: number;
        style?: number;
    }): Promise<{
        success: boolean;
        audio_url?: string;
        duration: number;
        voice_id: string;
        processing_time?: number;
        cost_estimate?: any;
        error_message?: string;
    }> {
        return this.request(`/api/chatterbox/generate`, {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    async generatePodcastSegmentAudio(data: {
        text: string;
        speaker_id: string;
        segment_type?: string;
        voice_settings?: any;
    }): Promise<{
        success: boolean;
        audio_url?: string;
        duration: number;
        voice_id: string;
        processing_time?: number;
        cost_estimate?: any;
        error_message?: string;
    }> {
        return this.request(`/api/chatterbox/podcast-segment`, {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    async getChatterboxConfig(): Promise<{
        service: string;
        available: boolean;
        device: string;
        sample_rate: number;
        voice_profiles: any;
        health: any;
    }> {
        return this.request(`/api/chatterbox/config`);
    }

    // Stripe payment endpoints
    async getStripeConfig(): Promise<{ publishable_key: string }> {
        return this.request<{ publishable_key: string }>('/api/stripe/config')
    }

    async getCreditBundles(): Promise<CreditBundle[]> {
        return this.request<CreditBundle[]>('/api/stripe/bundles')
    }

    async createPaymentIntent(bundle: string): Promise<PaymentIntentResponse> {
        return this.request<PaymentIntentResponse>('/api/stripe/create-payment-intent', {
            method: 'POST',
            body: JSON.stringify({ bundle }),
        })
    }

    async getPaymentStatus(paymentIntentId: string): Promise<PaymentStatus> {
        return this.request<PaymentStatus>(`/api/stripe/payment-status/${paymentIntentId}`)
    }
}

// Import types for Stripe
import type { CreditBundle, PaymentIntentResponse, PaymentStatus } from './stripe'

export const api = new ApiClient()

// Replace ElevenLabs voices endpoint with Chatterbox
export const getAvailableVoices = () =>
    api.getVoiceProfiles();

// Replace ElevenLabs cost estimation with Chatterbox
export const estimateAudioCost = (text: string) =>
    api.estimateVoiceCost(text);

// New Chatterbox endpoints
export const testChatterboxConnection = () =>
    api.testChatterboxConnection();

export const getChatterboxHealth = () =>
    api.getChatterboxHealth();

export const getPodcastVoices = () =>
    api.getPodcastVoiceProfiles();

export const generateTTS = (data: {
    text: string;
    voice_id?: string;
    audio_prompt_path?: string;
    speed?: number;
    stability?: number;
    similarity_boost?: number;
    style?: number;
}) =>
    api.generateTTSAudio(data);

export const generatePodcastSegment = (data: {
    text: string;
    speaker_id: string;
    segment_type?: string;
    voice_settings?: any;
}) =>
    api.generatePodcastSegmentAudio(data);

export const getChatterboxConfig = () =>
    api.getChatterboxConfig(); 