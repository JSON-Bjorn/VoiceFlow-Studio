import React, { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { AlertCircle, CheckCircle, Loader2, X, Wifi, WifiOff } from 'lucide-react';
import { useWebSocket, ProgressUpdate } from '../hooks/useWebSocket';
import { useAuth } from '../hooks/useAuth';

interface GenerationProgressModalProps {
    isOpen: boolean;
    onClose: () => void;
    podcastTitle: string;
    generationId?: string;
    onGenerationComplete?: (result: any) => void;
    onGenerationError?: (error: string) => void;
}

interface RetryInfo {
    attempt: number;
    max_attempts: number;
    last_error?: string;
    recovered?: boolean;
    attempts_used?: number;
}

interface ErrorDetails {
    error_code: string;
    category: string;
    severity: string;
    suggested_action: string;
    user_message: string;
    retry_recommended: boolean;
    phase?: string;
    technical_error?: string;
}

export function GenerationProgressModal({
    isOpen,
    onClose,
    podcastTitle,
    generationId,
    onGenerationComplete,
    onGenerationError
}: GenerationProgressModalProps) {
    const { token } = useAuth();
    const [currentPhase, setCurrentPhase] = useState<string>('initialization');
    const [progress, setProgress] = useState<number>(0);
    const [message, setMessage] = useState<string>('Preparing generation...');
    const [isCompleted, setIsCompleted] = useState<boolean>(false);
    const [hasError, setHasError] = useState<boolean>(false);
    const [errorMessage, setErrorMessage] = useState<string>('');
    const [errorDetails, setErrorDetails] = useState<ErrorDetails | null>(null);
    const [retryInfo, setRetryInfo] = useState<RetryInfo | null>(null);
    const [isRetrying, setIsRetrying] = useState<boolean>(false);
    const [recoveryActions, setRecoveryActions] = useState<string[]>([]);
    const [generationResult, setGenerationResult] = useState<any>(null);
    const [startTime, setStartTime] = useState<Date | null>(null);
    const [elapsedTime, setElapsedTime] = useState<number>(0);

    // Enhanced phase definitions with retry and recovery descriptions
    const GENERATION_PHASES = {
        'initialization': { name: 'Initialization', description: 'Setting up generation pipeline' },
        'research': { name: 'Research', description: 'Gathering and analyzing information' },
        'planning': { name: 'Planning', description: 'Creating content structure' },
        'script_generation': { name: 'Script Generation', description: 'Writing dialogue and content' },
        'voice_generation': { name: 'Voice Generation', description: 'Creating audio from text' },
        'audio_assembly': { name: 'Audio Assembly', description: 'Combining audio elements' },
        'validation': { name: 'Validation', description: 'Quality checking' },
        'saving': { name: 'Saving', description: 'Finalizing and storing' },
        'error_recovery': { name: 'Error Recovery', description: 'Handling errors and retrying' },
        'completed': { name: 'Completed', description: 'Generation finished successfully' }
    };

    // Update elapsed time
    useEffect(() => {
        if (!startTime || isCompleted || hasError) return;

        const interval = setInterval(() => {
            setElapsedTime(Math.floor((new Date().getTime() - startTime.getTime()) / 1000));
        }, 1000);

        return () => clearInterval(interval);
    }, [startTime, isCompleted, hasError]);

    const handleProgressUpdate = (update: ProgressUpdate) => {
        if (update.generation_id !== generationId) return;

        setProgress(update.progress || 0);
        setCurrentPhase(update.phase || 'initialization');
        setMessage(update.message || 'Processing...');

        // Handle retry information
        if (update.metadata?.retry_info) {
            const retry = update.metadata.retry_info as RetryInfo;
            setRetryInfo(retry);
            setIsRetrying(true);

            if (retry.recovered) {
                setIsRetrying(false);
                setRetryInfo(null);
                setRecoveryActions(prev => [...prev, `Recovered after ${retry.attempts_used} attempts`]);
            }
        } else if (currentPhase !== 'error_recovery') {
            setIsRetrying(false);
            setRetryInfo(null);
        }

        // Reset error state on successful progress
        if (hasError && update.phase !== 'error_recovery') {
            setHasError(false);
            setErrorMessage('');
            setErrorDetails(null);
        }
    };

    const handleGenerationComplete = (result: ProgressUpdate) => {
        if (result.generation_id !== generationId) return;

        setIsCompleted(true);
        setProgress(100);
        setCurrentPhase('completed');
        setMessage('Generation completed successfully!');
        setGenerationResult(result.result);
        setIsRetrying(false);
        setRetryInfo(null);
        onGenerationComplete?.(result.result);
    };

    const handleGenerationError = (error: ProgressUpdate) => {
        if (error.generation_id !== generationId) return;

        setHasError(true);
        setIsRetrying(false);
        setRetryInfo(null);
        setErrorMessage(error.error_message || 'An unknown error occurred');

        // Enhanced error details
        if (error.error_details) {
            setErrorDetails(error.error_details as ErrorDetails);
        }

        onGenerationError?.(error.error_message || 'Generation failed');
    };

    const { isConnected, connectionError } = useWebSocket(token, {
        onProgressUpdate: handleProgressUpdate,
        onGenerationComplete: handleGenerationComplete,
        onError: handleGenerationError,
        autoReconnect: true
    });

    // Initialize start time when modal opens
    useEffect(() => {
        if (isOpen && !startTime) {
            setStartTime(new Date());
        }
    }, [isOpen, startTime]);

    const formatTime = (seconds: number): string => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const getPhaseStatus = (phase: string) => {
        const phases = Object.keys(GENERATION_PHASES);
        const currentIndex = phases.indexOf(currentPhase);
        const phaseIndex = phases.indexOf(phase);

        if (phaseIndex < currentIndex) return 'completed';
        if (phaseIndex === currentIndex) return 'active';
        return 'pending';
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'completed': return 'bg-green-500';
            case 'active': return 'bg-blue-500';
            default: return 'bg-gray-300';
        }
    };

    const getErrorSeverityColor = (severity?: string) => {
        switch (severity) {
            case 'critical': return 'text-red-600 bg-red-50 border-red-200';
            case 'high': return 'text-orange-600 bg-orange-50 border-orange-200';
            case 'medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
            case 'low': return 'text-blue-600 bg-blue-50 border-blue-200';
            default: return 'text-gray-600 bg-gray-50 border-gray-200';
        }
    };

    const handleRetryGeneration = () => {
        // Reset states for retry
        setHasError(false);
        setErrorMessage('');
        setErrorDetails(null);
        setIsCompleted(false);
        setProgress(0);
        setCurrentPhase('initialization');
        setMessage('Retrying generation...');
        setStartTime(new Date());
        setElapsedTime(0);
        setRecoveryActions([]);

        // Trigger retry through parent component or API call
        // This would typically involve calling the generation API again
        console.log('Retry generation requested');
    };

    if (!isOpen) return null;

    return (
        <Dialog open={isOpen} onOpenChange={onClose}>
            <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        {hasError ? (
                            <AlertCircle className="h-5 w-5 text-red-500" />
                        ) : isCompleted ? (
                            <CheckCircle className="h-5 w-5 text-green-500" />
                        ) : (
                            <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
                        )}
                        <span>
                            {hasError ? 'Generation Error' :
                                isCompleted ? 'Generation Complete' :
                                    'Generating Podcast'}
                        </span>
                    </DialogTitle>
                </DialogHeader>

                <div className="space-y-6">
                    {/* Connection Status */}
                    <div className="flex items-center gap-2 text-sm">
                        {isConnected ? (
                            <>
                                <Wifi className="h-4 w-4 text-green-500" />
                                <span className="text-green-600">Connected</span>
                            </>
                        ) : (
                            <>
                                <WifiOff className="h-4 w-4 text-red-500" />
                                <span className="text-red-600">
                                    {connectionError || 'Disconnected'}
                                </span>
                            </>
                        )}

                        {startTime && (
                            <span className="ml-auto text-gray-500">
                                Elapsed: {formatTime(elapsedTime)}
                            </span>
                        )}
                    </div>

                    {/* Podcast Info */}
                    <div className="bg-gray-50 p-4 rounded-lg">
                        <h3 className="font-medium text-gray-900">{podcastTitle}</h3>
                        {generationId && (
                            <p className="text-sm text-gray-500 mt-1">
                                Generation ID: {generationId}
                            </p>
                        )}
                    </div>

                    {/* Error Display */}
                    {hasError && (
                        <div className={`p-4 rounded-lg border ${getErrorSeverityColor(errorDetails?.severity)}`}>
                            <div className="flex items-start gap-3">
                                <AlertCircle className="h-5 w-5 mt-0.5 flex-shrink-0" />
                                <div className="flex-1">
                                    <h4 className="font-medium">
                                        {errorDetails?.user_message || errorMessage}
                                    </h4>

                                    {errorDetails && (
                                        <div className="mt-2 space-y-2 text-sm">
                                            <div className="flex gap-4">
                                                <span>Category: <Badge variant="outline">{errorDetails.category}</Badge></span>
                                                <span>Severity: <Badge variant="outline">{errorDetails.severity}</Badge></span>
                                            </div>

                                            {errorDetails.suggested_action && (
                                                <p className="text-gray-600">
                                                    <strong>Suggested Action:</strong> {errorDetails.suggested_action}
                                                </p>
                                            )}

                                            {errorDetails.phase && (
                                                <p className="text-gray-600">
                                                    <strong>Failed Phase:</strong> {errorDetails.phase}
                                                </p>
                                            )}
                                        </div>
                                    )}

                                    {errorDetails?.retry_recommended && (
                                        <Button
                                            onClick={handleRetryGeneration}
                                            className="mt-3"
                                            variant="outline"
                                        >
                                            Retry Generation
                                        </Button>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Retry Information */}
                    {isRetrying && retryInfo && (
                        <div className="bg-yellow-50 border border-yellow-200 p-4 rounded-lg">
                            <div className="flex items-center gap-2">
                                <Loader2 className="h-4 w-4 animate-spin text-yellow-600" />
                                <span className="font-medium text-yellow-800">
                                    Retrying... (Attempt {retryInfo.attempt} of {retryInfo.max_attempts})
                                </span>
                            </div>
                            {retryInfo.last_error && (
                                <p className="text-sm text-yellow-700 mt-1">
                                    Previous error: {retryInfo.last_error}
                                </p>
                            )}
                        </div>
                    )}

                    {/* Recovery Actions */}
                    {recoveryActions.length > 0 && (
                        <div className="bg-blue-50 border border-blue-200 p-4 rounded-lg">
                            <h4 className="font-medium text-blue-800 mb-2">Recovery Actions</h4>
                            <ul className="text-sm text-blue-700 space-y-1">
                                {recoveryActions.map((action, index) => (
                                    <li key={index} className="flex items-center gap-2">
                                        <CheckCircle className="h-3 w-3" />
                                        {action}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* Progress Section */}
                    {!hasError && (
                        <>
                            <div className="space-y-2">
                                <div className="flex justify-between items-center">
                                    <span className="text-sm font-medium text-gray-700">
                                        {GENERATION_PHASES[currentPhase as keyof typeof GENERATION_PHASES]?.name || currentPhase}
                                    </span>
                                    <span className="text-sm text-gray-500">{progress}%</span>
                                </div>
                                <Progress value={progress} className="h-2" />
                                <p className="text-sm text-gray-600">{message}</p>
                            </div>

                            {/* Phase Status List */}
                            <div className="space-y-2">
                                <h4 className="font-medium text-gray-900">Generation Phases</h4>
                                <div className="space-y-1">
                                    {Object.entries(GENERATION_PHASES)
                                        .filter(([key]) => key !== 'error_recovery') // Hide error recovery unless active
                                        .map(([phase, info]) => {
                                            const status = getPhaseStatus(phase);
                                            return (
                                                <div key={phase} className="flex items-center gap-3 p-2 rounded">
                                                    <div
                                                        className={`w-3 h-3 rounded-full ${getStatusColor(status)}`}
                                                    />
                                                    <span className={`text-sm ${status === 'active' ? 'font-medium text-gray-900' : 'text-gray-600'
                                                        }`}>
                                                        {info.name}
                                                    </span>
                                                    <span className="text-xs text-gray-400 ml-auto">
                                                        {info.description}
                                                    </span>
                                                </div>
                                            );
                                        })}
                                </div>
                            </div>
                        </>
                    )}

                    {/* Generation Result Summary */}
                    {isCompleted && generationResult && (
                        <div className="bg-green-50 border border-green-200 p-4 rounded-lg">
                            <h4 className="font-medium text-green-800 mb-2">Generation Summary</h4>
                            <div className="text-sm text-green-700 space-y-1">
                                {generationResult.errors_encountered?.length > 0 && (
                                    <p>Errors encountered: {generationResult.errors_encountered.length}</p>
                                )}
                                {generationResult.recovery_actions?.length > 0 && (
                                    <p>Recovery actions applied: {generationResult.recovery_actions.length}</p>
                                )}
                                {generationResult.voice_data && (
                                    <p>Voice generation: Successful</p>
                                )}
                                {generationResult.audio_data && (
                                    <p>Audio assembly: Successful</p>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Close Button */}
                    <div className="flex justify-end">
                        <Button onClick={onClose} variant="outline">
                            {isCompleted || hasError ? 'Close' : 'Cancel'}
                        </Button>
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    );
} 