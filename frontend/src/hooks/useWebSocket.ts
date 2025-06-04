import { useEffect, useRef, useState, useCallback } from 'react';

export interface ProgressUpdate {
    type: 'progress_update' | 'generation_complete' | 'generation_error' | 'pong' | 'status_response';
    generation_id?: string;
    phase?: string;
    progress?: number;
    message?: string;
    timestamp?: string;
    metadata?: any;
    success?: boolean;
    result?: any;
    error_message?: string;
    error_details?: any;
    status?: any;
}

export interface UseWebSocketOptions {
    onProgressUpdate?: (update: ProgressUpdate) => void;
    onGenerationComplete?: (result: ProgressUpdate) => void;
    onError?: (error: ProgressUpdate) => void;
    autoReconnect?: boolean;
    reconnectInterval?: number;
}

export function useWebSocket(token: string | null, options: UseWebSocketOptions = {}) {
    const {
        onProgressUpdate,
        onGenerationComplete,
        onError,
        autoReconnect = true,
        reconnectInterval = 5000
    } = options;

    const [isConnected, setIsConnected] = useState(false);
    const [connectionError, setConnectionError] = useState<string | null>(null);
    const [lastUpdate, setLastUpdate] = useState<ProgressUpdate | null>(null);

    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const reconnectAttemptsRef = useRef(0);
    const maxReconnectAttempts = 5;

    const connect = useCallback(() => {
        if (!token) {
            setConnectionError('No authentication token provided');
            return;
        }

        try {
            const wsUrl = `ws://localhost:8000/ws/progress/${token}`;
            const ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                console.log('WebSocket connected');
                setIsConnected(true);
                setConnectionError(null);
                reconnectAttemptsRef.current = 0;

                // Send initial ping
                ws.send(JSON.stringify({ type: 'ping' }));
            };

            ws.onmessage = (event) => {
                try {
                    const data: ProgressUpdate = JSON.parse(event.data);
                    setLastUpdate(data);

                    switch (data.type) {
                        case 'progress_update':
                            onProgressUpdate?.(data);
                            break;
                        case 'generation_complete':
                            onGenerationComplete?.(data);
                            break;
                        case 'generation_error':
                            onError?.(data);
                            break;
                        case 'pong':
                            // Handle ping/pong for connection health
                            break;
                        case 'status_response':
                            // Handle status response
                            break;
                        default:
                            console.log('Unknown message type:', data.type);
                    }
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };

            ws.onclose = (event) => {
                console.log('WebSocket disconnected:', event.code, event.reason);
                setIsConnected(false);
                wsRef.current = null;

                // Attempt to reconnect if enabled and not too many attempts
                if (autoReconnect &&
                    reconnectAttemptsRef.current < maxReconnectAttempts &&
                    event.code !== 1000) { // Don't reconnect on normal closure

                    reconnectAttemptsRef.current += 1;
                    const delay = reconnectInterval * Math.pow(2, reconnectAttemptsRef.current - 1);

                    console.log(`Attempting to reconnect in ${delay}ms (attempt ${reconnectAttemptsRef.current})`);

                    reconnectTimeoutRef.current = setTimeout(() => {
                        connect();
                    }, delay);
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                setConnectionError('WebSocket connection failed');
            };

            wsRef.current = ws;

        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            setConnectionError('Failed to create WebSocket connection');
        }
    }, [token, autoReconnect, reconnectInterval, onProgressUpdate, onGenerationComplete, onError]);

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }

        if (wsRef.current) {
            wsRef.current.close(1000, 'Client disconnect');
            wsRef.current = null;
        }

        setIsConnected(false);
        reconnectAttemptsRef.current = 0;
    }, []);

    const sendMessage = useCallback((message: any) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(message));
            return true;
        }
        return false;
    }, []);

    const requestStatus = useCallback((generationId: string) => {
        return sendMessage({
            type: 'get_status',
            generation_id: generationId
        });
    }, [sendMessage]);

    const ping = useCallback(() => {
        return sendMessage({ type: 'ping' });
    }, [sendMessage]);

    // Connect on mount and token change
    useEffect(() => {
        if (token) {
            connect();
        }

        return () => {
            disconnect();
        };
    }, [token, connect, disconnect]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            disconnect();
        };
    }, [disconnect]);

    // Periodic ping to keep connection alive
    useEffect(() => {
        if (isConnected) {
            const pingInterval = setInterval(() => {
                ping();
            }, 30000); // Ping every 30 seconds

            return () => clearInterval(pingInterval);
        }
    }, [isConnected, ping]);

    return {
        isConnected,
        connectionError,
        lastUpdate,
        connect,
        disconnect,
        sendMessage,
        requestStatus,
        ping,
        reconnectAttempts: reconnectAttemptsRef.current
    };
} 