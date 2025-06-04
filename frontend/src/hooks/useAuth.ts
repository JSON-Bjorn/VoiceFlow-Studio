import { useState, useEffect } from 'react';

interface User {
    id: number;
    email: string;
    name: string;
}

interface AuthState {
    user: User | null;
    token: string | null;
    isLoading: boolean;
}

export function useAuth() {
    const [authState, setAuthState] = useState<AuthState>({
        user: null,
        token: null,
        isLoading: true
    });

    useEffect(() => {
        // Get token from localStorage
        const token = localStorage.getItem('token');
        const userStr = localStorage.getItem('user');

        if (token && userStr) {
            try {
                const user = JSON.parse(userStr);
                setAuthState({
                    user,
                    token,
                    isLoading: false
                });
            } catch (error) {
                console.error('Error parsing user data:', error);
                setAuthState({
                    user: null,
                    token: null,
                    isLoading: false
                });
            }
        } else {
            setAuthState({
                user: null,
                token: null,
                isLoading: false
            });
        }
    }, []);

    const login = (token: string, user: User) => {
        localStorage.setItem('token', token);
        localStorage.setItem('user', JSON.stringify(user));
        setAuthState({ user, token, isLoading: false });
    };

    const logout = () => {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        setAuthState({ user: null, token: null, isLoading: false });
    };

    return {
        ...authState,
        login,
        logout,
        isAuthenticated: !!authState.token
    };
} 