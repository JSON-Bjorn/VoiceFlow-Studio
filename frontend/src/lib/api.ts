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

export interface AuthResponse {
    access_token: string
    token_type: string
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

    logout() {
        this.clearToken()
    }
}

export const apiClient = new ApiClient() 