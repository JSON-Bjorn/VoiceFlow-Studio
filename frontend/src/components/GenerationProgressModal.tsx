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

interface PhaseInfo {
    name: string;
    description: string;
    estimatedDuration: number; // in seconds
    icon: React.ReactNode;
}

const GENERATION_PHASES: Record<string, PhaseInfo> = {
    research: {
        name: 'Research',
        description: 'Analyzing topic and gathering information',
        estimatedDuration: 30,
        icon: <Loader2 className="h-4 w-4" />
    },
    planning: {
        name: 'Content Planning',
        description: 'Creating strategic content outline',
        estimatedDuration: 20,
        icon: <Loader2 className="h-4 w-4" />
    },
    script_generation: {
        name: 'Script Generation',
        description: 'Generating and refining dialogue',
        estimatedDuration: 45,
        icon: <Loader2 className="h-4 w-4" />
    },
    voice_generation: {
        name: 'Voice Generation',
        description: 'Creating audio with Chatterbox TTS',
        estimatedDuration: 60,
        icon: <Loader2 className="h-4 w-4" />
    },
    audio_assembly: {
        name: 'Audio Assembly',
        description: 'Assembling final podcast episode',
        estimatedDuration: 25,
        icon: <Loader2 className="h-4 w-4" />
    },
    validation: {
        name: 'Quality Validation',
        description: 'Final quality checks and validation',
        estimatedDuration: 15,
        icon: <Loader2 className="h-4 w-4" />
    },
    saving: {
        name: 'Saving',
        description: 'Saving generated content',
        estimatedDuration: 10,
        icon: <Loader2 className="h-4 w-4" />
    },
    completed: {
        name: 'Completed',
        description: 'Generation completed successfully',
        estimatedDuration: 0,
        icon: <CheckCircle className="h-4 w-4 text-green-500" />
    }
};

export function GenerationProgressModal({
    isOpen,
    onClose,
    podcastTitle,
    generationId,
    onGenerationComplete,
    onGenerationError
}: GenerationProgressModalProps) {
    const { token } = useAuth();
    const [currentPhase, setCurrentPhase] = useState<string>('research');
    const [progress, setProgress] = useState(0);
    const [message, setMessage] = useState('Initializing generation...');
    const [isCompleted, setIsCompleted] = useState(false);
    const [hasError, setHasError] = useState(false);
    const [errorMessage, setErrorMessage] = useState('');
    const [startTime, setStartTime] = useState<Date | null>(null);
    const [estimatedTimeRemaining, setEstimatedTimeRemaining] = useState<number | null>(null);
    const [generationResult, setGenerationResult] = useState<any>(null);

    const handleProgressUpdate = (update: ProgressUpdate) => {
        if (update.generation_id !== generationId) return;

        setCurrentPhase(update.phase || 'research');
        setProgress(update.progress || 0);
        setMessage(update.message || 'Processing...');

        // Calculate estimated time remaining
        if (update.phase && GENERATION_PHASES[update.phase] && startTime) {
            const phases = Object.keys(GENERATION_PHASES);
            const currentPhaseIndex = phases.indexOf(update.phase);
            let remainingTime = 0;

            for (let i = currentPhaseIndex; i < phases.length; i++) {
                remainingTime += GENERATION_PHASES[phases[i]].estimatedDuration;
            }

            // Adjust based on current progress in phase
            if (update.progress && update.progress > 0) {
                const phaseProgress = (update.progress % 100) / 100;
                remainingTime -= GENERATION_PHASES[update.phase].estimatedDuration * phaseProgress;
            }

            setEstimatedTimeRemaining(Math.max(0, remainingTime));
        }
    };

    const handleGenerationComplete = (result: ProgressUpdate) => {
        if (result.generation_id !== generationId) return;

        setIsCompleted(true);
        setProgress(100);
        setCurrentPhase('completed');
        setMessage('Generation completed successfully!');
        setGenerationResult(result.result);
        onGenerationComplete?.(result.result);
    };

    const handleGenerationError = (error: ProgressUpdate) => {
        if (error.generation_id !== generationId) return;

        setHasError(true);
        setErrorMessage(error.error_message || 'An unknown error occurred');
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

    if (!isOpen) return null;

    return (
        <Dialog open={isOpen} onOpenChange={() => { }}>
            <DialogContent className="max-w-2xl">
                <DialogHeader>
                    <DialogTitle className="flex items-center justify-between">
                        <span>Generating Podcast: {podcastTitle}</span>
                        <div className="flex items-center gap-2">
                            {isConnected ? (
                                <Wifi className="h-4 w-4 text-green-500" />
                            ) : (
                                <WifiOff className="h-4 w-4 text-red-500" />
                            )}
                            {!isCompleted && !hasError && (
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={onClose}
                                    className="h-8 w-8 p-0"
                                >
                                    <X className="h-4 w-4" />
                                </Button>
                            )}
                        </div>
                    </DialogTitle>
                </DialogHeader>

                <div className="space-y-6">
                    {/* Connection Status */}
                    {connectionError && (
                        <div className="flex items-center gap-2 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                            <AlertCircle className="h-4 w-4 text-yellow-600" />
                            <span className="text-sm text-yellow-700">
                                Connection issue: {connectionError}. Progress updates may be delayed.
                            </span>
                        </div>
                    )}

                    {/* Error State */}
                    {hasError && (
                        <div className="flex items-center gap-2 p-4 bg-red-50 border border-red-200 rounded-lg">
                            <AlertCircle className="h-5 w-5 text-red-600" />
                            <div>
                                <p className="font-medium text-red-800">Generation Failed</p>
                                <p className="text-sm text-red-600">{errorMessage}</p>
                            </div>
                        </div>
                    )}

                    {/* Progress Overview */}
                    {!hasError && (
                        <div className="space-y-4">
                            {/* Overall Progress */}
                            <div className="space-y-2">
                                <div className="flex justify-between items-center">
                                    <span className="text-sm font-medium">Overall Progress</span>
                                    <span className="text-sm text-gray-500">{progress}%</span>
                                </div>
                                <Progress value={progress} className="h-2" />
                            </div>

                            {/* Current Status */}
                            <div className="flex items-center gap-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                                <Loader2 className="h-5 w-5 text-blue-600 animate-spin" />
                                <div>
                                    <p className="font-medium text-blue-800">{message}</p>
                                    {estimatedTimeRemaining !== null && (
                                        <p className="text-sm text-blue-600">
                                            Estimated time remaining: {formatTime(estimatedTimeRemaining)}
                                        </p>
                                    )}
                                </div>
                            </div>

                            {/* Phase Timeline */}
                            <div className="space-y-3">
                                <h4 className="font-medium text-gray-900">Generation Phases</h4>
                                <div className="space-y-2">
                                    {Object.entries(GENERATION_PHASES).map(([phase, info]) => {
                                        const status = getPhaseStatus(phase);
                                        return (
                                            <div key={phase} className="flex items-center gap-3">
                                                <div className={`w-3 h-3 rounded-full ${getStatusColor(status)}`} />
                                                <div className="flex-1">
                                                    <div className="flex items-center gap-2">
                                                        <span className={`font-medium ${status === 'active' ? 'text-blue-700' :
                                                                status === 'completed' ? 'text-green-700' :
                                                                    'text-gray-500'
                                                            }`}>
                                                            {info.name}
                                                        </span>
                                                        {status === 'active' && (
                                                            <Badge variant="outline" className="text-xs">
                                                                Active
                                                            </Badge>
                                                        )}
                                                        {status === 'completed' && (
                                                            <CheckCircle className="h-4 w-4 text-green-500" />
                                                        )}
                                                    </div>
                                                    <p className="text-sm text-gray-600">{info.description}</p>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Completion Actions */}
                    {(isCompleted || hasError) && (
                        <div className="flex justify-end gap-3 pt-4 border-t">
                            {isCompleted && generationResult && (
                                <Button
                                    onClick={() => {
                                        // You could trigger a download or redirect here
                                        console.log('Generation result:', generationResult);
                                    }}
                                    className="bg-green-600 hover:bg-green-700"
                                >
                                    View Result
                                </Button>
                            )}
                            <Button onClick={onClose} variant="outline">
                                {isCompleted ? 'Close' : 'Dismiss'}
                            </Button>
                        </div>
                    )}
                </div>
            </DialogContent>
        </Dialog>
    );
} 