'use client';

import { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
    DialogFooter
} from '@/components/ui/dialog';
import {
    Upload,
    Mic,
    Play,
    Pause,
    RefreshCw,
    Check,
    Loader2,
    Volume2,
    Download
} from 'lucide-react';
import { api } from '@/lib/api';

interface VoiceCloneModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSuccess?: (voiceId: string) => void;
}

interface CloneProgress {
    stage: 'uploading' | 'processing' | 'testing' | 'complete';
    message: string;
    progress: number;
}

export default function VoiceCloneModal({ isOpen, onClose, onSuccess }: VoiceCloneModalProps) {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [voiceName, setVoiceName] = useState('');
    const [voiceDescription, setVoiceDescription] = useState('');
    const [isRecording, setIsRecording] = useState(false);
    const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
    const [cloneProgress, setCloneProgress] = useState<CloneProgress | null>(null);
    const [isCloning, setIsCloning] = useState(false);
    const [testAudioUrl, setTestAudioUrl] = useState<string | null>(null);
    const [isPlayingTest, setIsPlayingTest] = useState(false);
    const [audioInputMethod, setAudioInputMethod] = useState<'upload' | 'record' | null>(null);

    const fileInputRef = useRef<HTMLInputElement>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioRef = useRef<HTMLAudioElement>(null);

    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            // Validate file type and size
            if (!file.type.startsWith('audio/')) {
                alert('Please select an audio file');
                return;
            }
            if (file.size > 10 * 1024 * 1024) { // 10MB limit
                alert('File size must be less than 10MB');
                return;
            }
            setSelectedFile(file);
            setRecordedBlob(null); // Clear recorded audio if file is selected
        }
    };

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;

            const chunks: BlobPart[] = [];
            mediaRecorder.ondataavailable = (event) => {
                chunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                const blob = new Blob(chunks, { type: 'audio/wav' });
                setRecordedBlob(blob);
                setSelectedFile(null); // Clear file if recording is made

                // Stop all tracks to release microphone
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            setIsRecording(true);
        } catch (error) {
            console.error('Error starting recording:', error);
            alert('Could not access microphone. Please check permissions.');
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    const playTestAudio = () => {
        if (audioRef.current && testAudioUrl) {
            if (isPlayingTest) {
                audioRef.current.pause();
                setIsPlayingTest(false);
            } else {
                audioRef.current.play();
                setIsPlayingTest(true);
            }
        }
    };

    const handleCloneVoice = async () => {
        if (!voiceName.trim()) {
            alert('Please enter a voice name');
            return;
        }

        if (!audioInputMethod) {
            alert('Please choose how to provide your voice sample');
            return;
        }

        if (audioInputMethod === 'upload' && !selectedFile) {
            alert('Please upload an audio file');
            return;
        }

        if (audioInputMethod === 'record' && !recordedBlob) {
            alert('Please record your voice');
            return;
        }

        setIsCloning(true);

        try {
            // Check authentication before starting
            try {
                api.refreshTokenFromStorage();
                if (!api.hasToken()) {
                    throw new Error('Please log in again to use voice cloning');
                }
            } catch (authError) {
                throw new Error('Authentication required. Please log in again.');
            }

            // Stage 1: Upload audio
            setCloneProgress({
                stage: 'uploading',
                message: 'Uploading audio sample...',
                progress: 25
            });

            const formData = new FormData();
            if (selectedFile) {
                formData.append('audio', selectedFile);
            } else if (recordedBlob) {
                formData.append('audio', recordedBlob, 'recorded_voice.wav');
            }
            formData.append('voice_name', voiceName);
            formData.append('description', voiceDescription || `Custom voice: ${voiceName}`);

            // Stage 2: Processing
            setCloneProgress({
                stage: 'processing',
                message: 'Processing voice characteristics...',
                progress: 50
            });

            const response = await api.request('/api/chatterbox/clone-voice', {
                method: 'POST',
                body: formData,
            }) as {
                voice_id: string;
                voice_name: string;
                test_audio_url?: string;
                voice_sample_url?: string;
                message: string;
            };

            // Stage 3: Testing (if test audio was generated)
            setCloneProgress({
                stage: 'testing',
                message: 'Voice cloned! Test sample ready...',
                progress: 75
            });

            // Set the test audio URL if available
            if (response.test_audio_url) {
                // Ensure the URL is properly formatted for the frontend
                const fullUrl = response.test_audio_url.startsWith('http')
                    ? response.test_audio_url
                    : `http://localhost:8000${response.test_audio_url}`;
                setTestAudioUrl(fullUrl);
                console.log("Test audio available at:", fullUrl);
            } else {
                console.warn("No test audio URL returned from server");
            }

            // Stage 4: Complete
            setCloneProgress({
                stage: 'complete',
                message: 'Voice cloning complete! Test your cloned voice below.',
                progress: 100
            });

            // Call the success callback to refresh user voices in the parent component
            if (onSuccess) {
                onSuccess(response.voice_id);
            }

            // Don't auto-close - let user listen to test sample and close manually
            // Auto-close is disabled so users can test their voice before closing

        } catch (error: unknown) {
            console.error('Voice cloning failed:', error);

            let errorMessage = 'Voice cloning failed. Please try again.';

            if (error instanceof Error) {
                errorMessage = error.message;
            } else if (typeof error === 'string') {
                errorMessage = error;
            } else if (error && typeof error === 'object' && 'message' in error) {
                errorMessage = String((error as any).message);
            }

            // Show user-friendly error
            alert(`Voice Cloning Failed\n\n${errorMessage}`);
            setCloneProgress(null);
        } finally {
            setIsCloning(false);
        }
    };

    const resetModal = () => {
        setSelectedFile(null);
        setVoiceName('');
        setVoiceDescription('');
        setRecordedBlob(null);
        setCloneProgress(null);
        setTestAudioUrl(null);
        setIsPlayingTest(false);
        setAudioInputMethod(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const handleClose = () => {
        if (!isCloning) {
            resetModal();
            onClose();
        }
    };

    return (
        <Dialog open={isOpen} onOpenChange={handleClose}>
            <DialogContent className="sm:max-w-[600px] p-0">
                <DialogHeader>
                    <DialogTitle>Clone Your Voice</DialogTitle>
                    <DialogDescription>
                        Upload an audio sample or record your voice to create a custom voice profile
                    </DialogDescription>
                </DialogHeader>

                <div className="px-6 py-4 space-y-6">
                    {/* Voice Information */}
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-slate-300 mb-2">
                                Voice Name *
                            </label>
                            <input
                                type="text"
                                value={voiceName}
                                onChange={(e) => setVoiceName(e.target.value)}
                                placeholder="e.g., My Voice, John's Voice"
                                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white placeholder-slate-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                disabled={isCloning}
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-slate-300 mb-2">
                                Description (Optional)
                            </label>
                            <input
                                type="text"
                                value={voiceDescription}
                                onChange={(e) => setVoiceDescription(e.target.value)}
                                placeholder="Describe the voice characteristics"
                                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white placeholder-slate-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                disabled={isCloning}
                            />
                        </div>
                    </div>

                    {/* Audio Input Methods */}
                    <div className="space-y-4">
                        <h3 className="text-lg font-medium text-white">Audio Sample</h3>

                        {!audioInputMethod ? (
                            /* Method Selection */
                            <div className="space-y-3">
                                <p className="text-sm text-slate-300 mb-4">
                                    Choose how you want to provide your voice sample:
                                </p>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                    <Button
                                        variant="outline"
                                        onClick={() => setAudioInputMethod('upload')}
                                        disabled={isCloning}
                                        className="h-20 border-slate-500 text-slate-300 hover:bg-slate-600 hover:border-purple-400 transition-all duration-200"
                                    >
                                        <div className="flex flex-col items-center gap-2">
                                            <Upload className="h-6 w-6" />
                                            <span className="font-medium">Upload Audio File</span>
                                            <span className="text-xs text-slate-400">MP3, WAV, M4A</span>
                                        </div>
                                    </Button>

                                    <Button
                                        variant="outline"
                                        onClick={() => setAudioInputMethod('record')}
                                        disabled={isCloning}
                                        className="h-20 border-slate-500 text-slate-300 hover:bg-slate-600 hover:border-purple-400 transition-all duration-200"
                                    >
                                        <div className="flex flex-col items-center gap-2">
                                            <Mic className="h-6 w-6" />
                                            <span className="font-medium">Record Your Voice</span>
                                            <span className="text-xs text-slate-400">30+ seconds recommended</span>
                                        </div>
                                    </Button>
                                </div>
                            </div>
                        ) : audioInputMethod === 'upload' ? (
                            /* File Upload */
                            <Card className="bg-slate-700/50 border-slate-600">
                                <CardContent className="p-4">
                                    <div className="flex items-center justify-between mb-3">
                                        <h4 className="font-medium text-slate-200">Upload Audio File</h4>
                                        <div className="flex items-center gap-2">
                                            <Upload className="h-5 w-5 text-slate-400" />
                                            <Button
                                                variant="ghost"
                                                size="sm"
                                                onClick={() => {
                                                    setAudioInputMethod(null);
                                                    setSelectedFile(null);
                                                    if (fileInputRef.current) fileInputRef.current.value = '';
                                                }}
                                                disabled={isCloning}
                                                className="text-slate-400 hover:text-slate-200"
                                            >
                                                <RefreshCw className="h-4 w-4" />
                                            </Button>
                                        </div>
                                    </div>

                                    <input
                                        ref={fileInputRef}
                                        type="file"
                                        accept="audio/*"
                                        onChange={handleFileSelect}
                                        className="hidden"
                                        disabled={isCloning}
                                    />

                                    <Button
                                        variant="outline"
                                        onClick={() => fileInputRef.current?.click()}
                                        disabled={isCloning}
                                        className="w-full border-slate-500 text-slate-300 hover:bg-slate-600"
                                    >
                                        {selectedFile ? (
                                            <div className="flex items-center gap-2">
                                                <Check className="h-4 w-4 text-green-400" />
                                                {selectedFile.name}
                                            </div>
                                        ) : (
                                            'Choose Audio File'
                                        )}
                                    </Button>

                                    <p className="text-xs text-slate-400 mt-2">
                                        Supported: MP3, WAV, M4A (max 10MB, 30+ seconds recommended)
                                    </p>
                                </CardContent>
                            </Card>
                        ) : (
                            /* Recording */
                            <Card className="bg-slate-700/50 border-slate-600">
                                <CardContent className="p-4">
                                    <div className="flex items-center justify-between mb-3">
                                        <h4 className="font-medium text-slate-200">Record Your Voice</h4>
                                        <div className="flex items-center gap-2">
                                            <Mic className="h-5 w-5 text-slate-400" />
                                            <Button
                                                variant="ghost"
                                                size="sm"
                                                onClick={() => {
                                                    setAudioInputMethod(null);
                                                    setRecordedBlob(null);
                                                    setIsRecording(false);
                                                }}
                                                disabled={isCloning || isRecording}
                                                className="text-slate-400 hover:text-slate-200"
                                            >
                                                <RefreshCw className="h-4 w-4" />
                                            </Button>
                                        </div>
                                    </div>

                                    <div className="space-y-3">
                                        <Button
                                            variant={isRecording ? "destructive" : "outline"}
                                            onClick={isRecording ? stopRecording : startRecording}
                                            disabled={isCloning}
                                            className={`w-full ${isRecording ? "" : "border-slate-500 text-slate-300 hover:bg-slate-600"}`}
                                        >
                                            {isRecording ? (
                                                <>
                                                    <Pause className="h-4 w-4 mr-2" />
                                                    Stop Recording
                                                </>
                                            ) : (
                                                <>
                                                    <Mic className="h-4 w-4 mr-2" />
                                                    {recordedBlob ? 'Record Again' : 'Start Recording'}
                                                </>
                                            )}
                                        </Button>

                                        {recordedBlob && (
                                            <div className="text-sm text-green-400 flex items-center justify-center">
                                                <Check className="h-4 w-4 mr-2" />
                                                Recording captured successfully
                                            </div>
                                        )}
                                    </div>

                                    <p className="text-xs text-slate-400 mt-2">
                                        Record 30+ seconds of clear speech for best results
                                    </p>
                                </CardContent>
                            </Card>
                        )}
                    </div>

                    {/* Progress Display */}
                    {cloneProgress && (
                        <Card className="bg-slate-700/50 border-slate-600">
                            <CardContent className="p-4">
                                <div className="flex items-center mb-3">
                                    {cloneProgress.stage === 'complete' ? (
                                        <Check className="h-5 w-5 text-green-400 mr-2" />
                                    ) : (
                                        <Loader2 className="h-5 w-5 animate-spin text-purple-400 mr-2" />
                                    )}
                                    <span className="font-medium text-white">{cloneProgress.message}</span>
                                </div>

                                <div className="w-full bg-slate-600 rounded-full h-2 mb-3">
                                    <div
                                        className="bg-gradient-to-r from-purple-500 to-purple-600 h-2 rounded-full transition-all duration-300"
                                        style={{ width: `${cloneProgress.progress}%` }}
                                    />
                                </div>

                                {testAudioUrl && (
                                    <div className="mt-4 p-3 bg-slate-600/50 rounded-lg">
                                        <div className="flex items-center justify-between mb-3">
                                            <span className="text-sm font-medium text-slate-200">🎧 Test Your Cloned Voice</span>
                                            <div className="flex items-center gap-2">
                                                <Button
                                                    variant="outline"
                                                    size="sm"
                                                    onClick={playTestAudio}
                                                    className="border-slate-500 text-slate-300 hover:bg-slate-600"
                                                >
                                                    {isPlayingTest ? (
                                                        <>
                                                            <Pause className="h-4 w-4 mr-1" />
                                                            Pause
                                                        </>
                                                    ) : (
                                                        <>
                                                            <Play className="h-4 w-4 mr-1" />
                                                            Play
                                                        </>
                                                    )}
                                                </Button>
                                                <Button
                                                    variant="outline"
                                                    size="sm"
                                                    onClick={() => window.open(testAudioUrl, '_blank')}
                                                    className="border-slate-500 text-slate-300 hover:bg-slate-600"
                                                >
                                                    <Download className="h-4 w-4 mr-1" />
                                                    Download
                                                </Button>
                                            </div>
                                        </div>
                                        <audio
                                            ref={audioRef}
                                            src={testAudioUrl}
                                            onEnded={() => setIsPlayingTest(false)}
                                            className="hidden"
                                        />
                                        <p className="text-xs text-slate-400">
                                            Listen to how your cloned voice sounds with the test message
                                        </p>
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    )}
                </div>

                <DialogFooter>
                    <Button
                        variant="outline"
                        onClick={handleClose}
                        disabled={isCloning}
                        className="border-slate-600 text-slate-300 hover:bg-slate-700"
                    >
                        {isCloning ? 'Processing...' : 'Cancel'}
                    </Button>
                    <Button
                        onClick={handleCloneVoice}
                        disabled={
                            isCloning ||
                            !voiceName.trim() ||
                            !audioInputMethod ||
                            (audioInputMethod === 'upload' && !selectedFile) ||
                            (audioInputMethod === 'record' && !recordedBlob)
                        }
                        className="bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white"
                    >
                        {isCloning ? (
                            <>
                                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                Cloning Voice...
                            </>
                        ) : (
                            <>
                                <Volume2 className="h-4 w-4 mr-2" />
                                Clone Voice
                            </>
                        )}
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
} 