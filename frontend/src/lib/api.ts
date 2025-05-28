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

    private async request<T>(
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
        const response = await fetch(`${this.baseUrl}/api/podcasts/${id}/generate`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.token}`,
            },
        })

        if (!response.ok) {
            throw new Error('Failed to generate podcast')
        }

        return response.json()
    }
}

export const apiClient = new ApiClient() 