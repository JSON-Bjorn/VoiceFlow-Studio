'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import PodcastCard from '@/components/PodcastCard';
import { CreditBalance } from '@/components/ui/credit-balance';
import { GenerationProgressModal } from '@/components/GenerationProgressModal';
import { Badge } from '@/components/ui/badge';
import {
    Plus,
    Filter,
    Search,
    Loader2,
    RefreshCw,
    Mic,
    Volume2,
    Settings,
    Clock,
    X,
    Play,
    Pause,
    ArrowLeft,
    AlertCircle
} from 'lucide-react';
import {
    Podcast,
    PodcastListResponse,
    PodcastSummary,
    PodcastCreate,
    CreditSummary,
    api
} from '@/lib/api';
import VoiceCloneModal from '@/components/VoiceCloneModal';

// Queue management types
interface GenerationQueueItem {
    id: string;
    podcastId: number;
    podcastTitle: string;
    startTime: Date;
    status: 'active' | 'queued' | 'completed' | 'error' | 'cancelled';
    phase?: string;
    progress?: number;
    estimatedCompletion?: Date;
    errorMessage?: string;
    result?: any;
}

export default function PodcastLibrary() {
    const router = useRouter();
    const [podcasts, setPodcasts] = useState<Podcast[]>([]);
    const [summary, setSummary] = useState<PodcastSummary | null>(null);
    const [creditSummary, setCreditSummary] = useState<CreditSummary | null>(null);
    const [loading, setLoading] = useState(true);
    const [creating, setCreating] = useState(false);
    const [generating, setGenerating] = useState<number | null>(null);
    const [currentPage, setCurrentPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [statusFilter, setStatusFilter] = useState<string>('');
    const [searchTerm, setSearchTerm] = useState('');
    const [showCreateForm, setShowCreateForm] = useState(false);
    // Removed showGenerationOptions - only one generation when creating podcasts
    const [showCreditConfirmation, setShowCreditConfirmation] = useState(false);
    const [pendingPodcastData, setPendingPodcastData] = useState<PodcastCreate | null>(null);

    // Queue management state
    const [generationQueue, setGenerationQueue] = useState<GenerationQueueItem[]>([]);
    const [showProgressModal, setShowProgressModal] = useState(false);
    const [currentGenerationId, setCurrentGenerationId] = useState<string | null>(null);
    const [showQueuePanel, setShowQueuePanel] = useState(false);
    const [maxConcurrentGenerations] = useState(3); // Allow up to 3 concurrent generations

    const [newPodcast, setNewPodcast] = useState<PodcastCreate>({
        title: '',
        topic: '',
        length: 10,
        voice_settings: undefined
    });

    // Generation options state
    const [generationOptions, setGenerationOptions] = useState({
        generateVoice: true,
        assemble_audio: true,
        hostPersonalities: {
            host_1: {
                name: 'Host 1',
                personality: 'analytical, thoughtful, engaging',
                role: 'primary_questioner',
                voice_id: '' // Add voice_id support
            },
            host_2: {
                name: 'Host 2',
                personality: 'warm, curious, conversational',
                role: 'storyteller',
                voice_id: '' // Add voice_id support
            }
        },
        stylePreferences: {
            tone: 'conversational',
            complexity: 'accessible',
            humor_level: 'light',
            pacing: 'moderate'
        },
        contentPreferences: {
            focus_areas: ['practical applications', 'recent developments'],
            avoid_topics: [] as string[],
            target_audience: 'general public'
        },
        audio_options: {
            add_intro: true,
            add_outro: true,
            intro_style: 'overlay', // 'overlay' or 'sequential'
            outro_style: 'overlay',
            intro_asset_id: 'default_intro',
            outro_asset_id: 'default_outro',
            add_transitions: false,
            transition_asset_id: 'default_transition',
            add_background_music: false,
            background_asset_id: ''
        }
    });

    // Voice cloning state
    const [showVoiceCloneModal, setShowVoiceCloneModal] = useState(false);

    // Voice selection state
    const [userVoices, setUserVoices] = useState<any[]>([]);
    const [systemVoices, setSystemVoices] = useState<any[]>([]);
    const [voiceSettings, setVoiceSettings] = useState({
        host1_voice_id: '',
        host2_voice_id: ''
    });

    // Queue management functions
    const addToQueue = (podcastId: number, podcastTitle: string): string => {
        const generationId = `gen_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const activeGenerations = generationQueue.filter(item => item.status === 'active').length;

        const queueItem: GenerationQueueItem = {
            id: generationId,
            podcastId,
            podcastTitle,
            startTime: new Date(),
            status: activeGenerations < maxConcurrentGenerations ? 'active' : 'queued',
            progress: 0
        };

        setGenerationQueue(prev => [...prev, queueItem]);
        return generationId;
    };

    const updateQueueItem = (generationId: string, updates: Partial<GenerationQueueItem>) => {
        setGenerationQueue(prev =>
            prev.map(item =>
                item.id === generationId ? { ...item, ...updates } : item
            )
        );
    };

    const removeFromQueue = (generationId: string) => {
        setGenerationQueue(prev => prev.filter(item => item.id !== generationId));

        // Start next queued generation if available
        const nextQueued = generationQueue.find(item => item.status === 'queued');
        if (nextQueued) {
            updateQueueItem(nextQueued.id, { status: 'active', startTime: new Date() });
        }
    };

    const cancelGeneration = (generationId: string) => {
        updateQueueItem(generationId, { status: 'cancelled' });
        // In a real implementation, you'd also send a cancellation request to the backend
        setTimeout(() => removeFromQueue(generationId), 2000);
    };

    // Get queue statistics
    const getQueueStats = () => {
        const active = generationQueue.filter(item => item.status === 'active').length;
        const queued = generationQueue.filter(item => item.status === 'queued').length;
        const completed = generationQueue.filter(item => item.status === 'completed').length;
        const errors = generationQueue.filter(item => item.status === 'error').length;

        return { active, queued, completed, errors, total: generationQueue.length };
    };

    // Load podcasts and summary
    const loadData = async () => {
        try {
            setLoading(true);
            const [podcastsResponse, summaryResponse, creditResponse] = await Promise.all([
                api.getPodcasts(currentPage, 12, statusFilter || undefined),
                api.getPodcastSummary(),
                api.getCreditSummary()
            ]);

            setPodcasts(podcastsResponse.podcasts);
            setTotalPages(podcastsResponse.total_pages);
            setSummary(summaryResponse);
            setCreditSummary(creditResponse);
        } catch (error) {
            console.error('Failed to load data:', error);
        } finally {
            setLoading(false);
        }
    };

    // Load user voices
    const loadUserVoices = async () => {
        try {
            const response = await api.getUserVoices();
            setUserVoices(response.voices);
        } catch (error) {
            console.error('Failed to load user voices:', error);
        }
    };

    // Load system voices
    const loadSystemVoices = async () => {
        try {
            const response = await fetch('http://localhost:8000/api/chatterbox/system-voices');
            const data = await response.json();
            if (data.success) {
                setSystemVoices(data.system_voices);
            }
        } catch (error) {
            console.error('Failed to load system voices:', error);
        }
    };

    useEffect(() => {
        loadData();
        loadUserVoices();
        loadSystemVoices();
    }, [currentPage, statusFilter]);

    // Auto-refresh data when there are active generations
    useEffect(() => {
        const activeGenerations = generationQueue.filter(item =>
            item.status === 'active' || item.status === 'queued'
        ).length;

        if (activeGenerations > 0) {
            const interval = setInterval(() => {
                loadData(); // Refresh podcast data
            }, 10000); // Every 10 seconds

            return () => clearInterval(interval);
        }
    }, [generationQueue]);

    // Create new podcast with confirmation
    const handleCreatePodcast = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!newPodcast.title.trim() || !newPodcast.topic.trim()) return;

        // MANDATORY: Validate voice selections
        if (!voiceSettings.host1_voice_id || !voiceSettings.host2_voice_id) {
            alert('❌ Voice Selection Required: You must select voices for both Host 1 and Host 2 before creating a podcast. No default voices are available.');
            return;
        }

        // Prepare podcast data with voice settings (always included now)
        const podcastData = {
            ...newPodcast,
            voice_settings: voiceSettings // Always include voice settings
        };

        // Store the podcast data and show credit confirmation
        setPendingPodcastData(podcastData);
        setShowCreateForm(false);
        setShowCreditConfirmation(true);
    };

    // Confirm and create podcast with immediate generation
    const handleConfirmCreateAndGenerate = async () => {
        if (!pendingPodcastData) return;

        try {
            setCreating(true);
            setShowCreditConfirmation(false);

            console.log('Creating podcast with data:', pendingPodcastData);
            console.log('Current voice settings at creation time:', voiceSettings);

            // Create the podcast
            const podcast = await api.createPodcast(pendingPodcastData);
            console.log('Created podcast:', podcast);
            setPodcasts(prev => [podcast, ...prev]);

            // Clear form data
            setNewPodcast({ title: '', topic: '', length: 10 });
            setPendingPodcastData(null);

            // Start background generation immediately
            startBackgroundGeneration(podcast);

            // Refresh summaries
            const [summaryResponse, creditResponse] = await Promise.all([
                api.getPodcastSummary(),
                api.getCreditSummary()
            ]);
            setSummary(summaryResponse);
            setCreditSummary(creditResponse);

        } catch (error: any) {
            alert(error.message || 'Failed to create podcast');
        } finally {
            setCreating(false);
        }
    };

    // Cancel creation
    const handleCancelCreate = () => {
        setShowCreditConfirmation(false);
        setPendingPodcastData(null);
        setShowCreateForm(true); // Go back to form
    };

    // Start background generation (only used for new podcast creation)
    const startBackgroundGeneration = async (podcast: Podcast) => {
        // Add to generation queue
        const generationId = addToQueue(podcast.id, podcast.title);
        setCurrentGenerationId(generationId);
        setShowProgressModal(true);

        // Update podcast status to generating immediately
        setPodcasts(prev => prev.map(p =>
            p.id === podcast.id ? { ...p, status: 'generating' } : p
        ));

        // Start the generation process in background (don't await)
        handleBackgroundGeneration(podcast, generationId);
    };

    // Handle background generation (async, doesn't block UI)
    const handleBackgroundGeneration = async (podcast: Podcast, generationId: string) => {
        try {
            console.log('Starting background generation for podcast:', podcast.id);
            console.log('Current voiceSettings:', voiceSettings);
            console.log('Current generationOptions.hostPersonalities:', generationOptions.hostPersonalities);

            // Check if we have voice IDs from either source
            const host1VoiceId = voiceSettings.host1_voice_id || generationOptions.hostPersonalities.host_1.voice_id;
            const host2VoiceId = voiceSettings.host2_voice_id || generationOptions.hostPersonalities.host_2.voice_id;

            console.log('Resolved voice IDs:', { host1VoiceId, host2VoiceId });

            // Determine final voice IDs (with fallback to podcast.voice_settings if needed)
            let finalHost1VoiceId = host1VoiceId || '';
            let finalHost2VoiceId = host2VoiceId || '';

            // Last resort: try to get voice settings from the podcast if available
            if ((!finalHost1VoiceId || !finalHost2VoiceId) && podcast.voice_settings) {
                console.log('Using voice settings from podcast:', podcast.voice_settings);
                finalHost1VoiceId = finalHost1VoiceId || podcast.voice_settings.host1_voice_id || '';
                finalHost2VoiceId = finalHost2VoiceId || podcast.voice_settings.host2_voice_id || '';
                console.log('Final resolved voice IDs:', { finalHost1VoiceId, finalHost2VoiceId });
            }

            // Validate that we have both voice IDs before proceeding
            if (!finalHost1VoiceId || !finalHost2VoiceId) {
                const missingVoices = [];
                if (!finalHost1VoiceId) missingVoices.push('Host 1');
                if (!finalHost2VoiceId) missingVoices.push('Host 2');

                const errorMessage = `❌ Voice Selection Error: Missing voice selections for ${missingVoices.join(' and ')}. This should not happen - please try creating the podcast again.`;
                console.error(errorMessage);

                // Update queue item as error
                updateQueueItem(generationId, {
                    status: 'error',
                    errorMessage: errorMessage
                });
                return;
            }

            // Sync voice settings with generation options before making API call
            const optionsWithVoiceSettings = {
                ...generationOptions,
                hostPersonalities: {
                    host_1: {
                        ...generationOptions.hostPersonalities.host_1,
                        voice_id: finalHost1VoiceId
                    },
                    host_2: {
                        ...generationOptions.hostPersonalities.host_2,
                        voice_id: finalHost2VoiceId
                    }
                }
            };

            console.log('Generation options with voice settings:', optionsWithVoiceSettings);

            // Start the generation process
            const result = await api.generatePodcastWithAI(podcast.id, optionsWithVoiceSettings);

            console.log('Generation result received:', result);

            if (result.success) {
                console.log('Generation successful, updating UI state');

                // Update queue item as completed
                updateQueueItem(generationId, {
                    status: 'completed',
                    progress: 100,
                    result: result
                });

                // Update podcast status in the list
                setPodcasts(prev => prev.map(p =>
                    p.id === podcast.id ? { ...p, status: 'completed' } : p
                ));

                // Force refresh of podcast data from server
                console.log('Refreshing podcast data from server');
                await loadData();
            } else {
                // Update queue item as error
                updateQueueItem(generationId, {
                    status: 'error',
                    errorMessage: result.message || 'Generation failed'
                });

                // Update podcast status in the list
                setPodcasts(prev => prev.map(p =>
                    p.id === podcast.id ? { ...p, status: 'failed' } : p
                ));

                // Still refresh data on failure to get latest status
                await loadData();
            }

        } catch (error: any) {
            console.error('Generation error:', error);

            // Update queue item as error
            updateQueueItem(generationId, {
                status: 'error',
                errorMessage: error.message || 'Generation failed'
            });

            // Update podcast status in the list
            setPodcasts(prev => prev.map(p =>
                p.id === podcast.id ? { ...p, status: 'failed' } : p
            ));

            // Refresh data to get latest status even on error
            await loadData();
        }
    };

    // Note: Removed handleGenerateEnhanced - only one generation when creating podcasts

    // Progress modal handlers
    const handleGenerationComplete = (result: any) => {
        if (currentGenerationId) {
            updateQueueItem(currentGenerationId, {
                status: 'completed',
                progress: 100,
                result: result
            });
        }
        // Refresh podcasts list
        loadData();
    };

    const handleGenerationError = (error: string) => {
        if (currentGenerationId) {
            updateQueueItem(currentGenerationId, {
                status: 'error',
                errorMessage: error
            });
        }
    };

    const handleCloseProgressModal = () => {
        setShowProgressModal(false);
        // Don't clear currentGenerationId - generation continues in background
        // setCurrentGenerationId(null);
    };

    // Note: Removed showGenerationModal - only one generation when creating podcasts

    // Delete podcast
    const handleDelete = async (podcast: Podcast) => {
        if (!confirm(`Are you sure you want to delete "${podcast.title}"?`)) return;

        try {
            await api.deletePodcast(podcast.id);
            setPodcasts(prev => prev.filter(p => p.id !== podcast.id));

            // Refresh summary
            const summaryResponse = await api.getPodcastSummary();
            setSummary(summaryResponse);

            // Refresh credit summary
            const creditResponse = await api.getCreditSummary();
            setCreditSummary(creditResponse);
        } catch (error: any) {
            alert(error.message || 'Failed to delete podcast');
        }
    };

    // Filter podcasts by search term
    const filteredPodcasts = podcasts.filter(podcast =>
        podcast.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        podcast.topic.toLowerCase().includes(searchTerm.toLowerCase())
    );

    // Get queue statistics for display
    const queueStats = getQueueStats();

    const handleVoiceCloneSuccess = async (voiceId: string) => {
        console.log('Voice cloned successfully:', voiceId);

        // Give a small delay to ensure the API has fully processed the voice
        setTimeout(async () => {
            // Refresh user voices to include the new voice
            await loadUserVoices();
            // Also refresh system voices in case the voice was promoted
            await loadSystemVoices();
        }, 500);

        alert(`Voice cloned successfully! You can now use it in podcast generation.`);
        // Don't close the modal here - let the user close it manually after testing
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
            {/* Header */}
            <header className="border-b border-slate-700 bg-slate-800/50">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between items-center h-16">
                        <div className="flex items-center space-x-4">
                            <Button
                                variant="ghost"
                                onClick={() => router.push('/dashboard')}
                                className="flex items-center space-x-2 text-gray-300 hover:text-white"
                            >
                                <ArrowLeft className="h-5 w-5" />
                                <span>Back to Dashboard</span>
                            </Button>
                        </div>

                        <div className="flex items-center space-x-2">
                            <Mic className="h-8 w-8 text-purple-400" />
                            <span className="text-xl font-bold text-white">VoiceFlow Studio</span>
                        </div>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {/* Page Header Section */}
                <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between mb-8">
                    <div>
                        <h1 className="text-3xl font-bold text-white mb-2">My Podcast Library</h1>
                        <p className="text-gray-300">Create, manage, and listen to your AI-generated podcasts</p>
                    </div>

                    <div className="mt-4 lg:mt-0 flex items-center gap-4">
                        <CreditBalance
                            credits={creditSummary?.current_balance || 0}
                            variant="header"
                        />

                        {/* Queue Status Indicator */}
                        {queueStats.total > 0 && (
                            <Button
                                variant="outline"
                                onClick={() => setShowQueuePanel(!showQueuePanel)}
                                className="flex items-center gap-2 border-purple-500/50 text-purple-300 hover:bg-purple-500/20 hover:text-purple-200 hover:border-purple-400 transition-all duration-200"
                            >
                                <Clock className="h-4 w-4" />
                                <span className="hidden sm:inline">Queue</span>
                                <Badge variant="secondary" className="ml-1 bg-purple-500/20 text-purple-300">
                                    {queueStats.active + queueStats.queued}
                                </Badge>
                            </Button>
                        )}

                        <Button
                            onClick={() => setShowCreateForm(true)}
                            className="bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white border-0 font-semibold shadow-lg hover:shadow-purple-500/25 transition-all duration-200"
                        >
                            <Plus className="h-4 w-4 mr-2" />
                            New Podcast
                        </Button>

                        {/* Add Clone Voice button */}
                        <Button
                            onClick={() => setShowVoiceCloneModal(true)}
                            variant="outline"
                            className="border-purple-500/50 text-purple-300 hover:bg-purple-500/20 hover:text-purple-200 hover:border-purple-400 transition-all duration-200"
                        >
                            <Mic className="h-4 w-4 mr-2" />
                            Clone Your Voice
                        </Button>
                    </div>
                </div>

                {/* Queue Management Panel */}
                {showQueuePanel && queueStats.total > 0 && (
                    <Card className="mb-6 bg-slate-800/50 border-slate-700">
                        <CardHeader>
                            <CardTitle className="flex items-center justify-between text-white">
                                <div className="flex items-center gap-2">
                                    <Clock className="h-5 w-5 text-purple-400" />
                                    Generation Queue
                                </div>
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => setShowQueuePanel(false)}
                                    className="text-slate-300 hover:text-white hover:bg-slate-700"
                                >
                                    <X className="h-4 w-4" />
                                </Button>
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-3">
                                {generationQueue.map((item) => (
                                    <div
                                        key={item.id}
                                        className="flex items-center justify-between p-3 border rounded-lg"
                                    >
                                        <div className="flex items-center gap-3">
                                            <div className="flex items-center gap-2">
                                                {item.status === 'active' && (
                                                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                                                )}
                                                {item.status === 'queued' && (
                                                    <div className="w-2 h-2 bg-yellow-500 rounded-full" />
                                                )}
                                                {item.status === 'completed' && (
                                                    <div className="w-2 h-2 bg-blue-500 rounded-full" />
                                                )}
                                                {item.status === 'error' && (
                                                    <div className="w-2 h-2 bg-red-500 rounded-full" />
                                                )}
                                                {item.status === 'cancelled' && (
                                                    <div className="w-2 h-2 bg-gray-500 rounded-full" />
                                                )}
                                            </div>

                                            <div>
                                                <p className="font-medium text-sm">{item.podcastTitle}</p>
                                                <div className="flex items-center gap-2 text-xs text-gray-500">
                                                    <Badge
                                                        variant={
                                                            item.status === 'active' ? 'default' :
                                                                item.status === 'queued' ? 'secondary' :
                                                                    item.status === 'completed' ? 'outline' :
                                                                        item.status === 'error' ? 'destructive' : 'secondary'
                                                        }
                                                        className="text-xs"
                                                    >
                                                        {item.status.charAt(0).toUpperCase() + item.status.slice(1)}
                                                    </Badge>
                                                    {item.phase && (
                                                        <span>{item.phase.replace('_', ' ')}</span>
                                                    )}
                                                    {item.progress !== undefined && item.progress > 0 && (
                                                        <span>{item.progress}%</span>
                                                    )}
                                                </div>
                                            </div>
                                        </div>

                                        <div className="flex items-center gap-2">
                                            {item.status === 'active' && (
                                                <Button
                                                    variant="ghost"
                                                    size="sm"
                                                    onClick={() => {
                                                        setCurrentGenerationId(item.id);
                                                        setShowProgressModal(true);
                                                    }}
                                                >
                                                    <Play className="h-3 w-3" />
                                                </Button>
                                            )}

                                            {(item.status === 'active' || item.status === 'queued') && (
                                                <Button
                                                    variant="ghost"
                                                    size="sm"
                                                    onClick={() => cancelGeneration(item.id)}
                                                    className="text-red-600 hover:text-red-700"
                                                >
                                                    <X className="h-3 w-3" />
                                                </Button>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>

                            {/* Queue Statistics */}
                            <div className="mt-4 pt-3 border-t flex justify-between text-sm text-gray-600">
                                <span>Active: {queueStats.active}</span>
                                <span>Queued: {queueStats.queued}</span>
                                <span>Completed: {queueStats.completed}</span>
                                {queueStats.errors > 0 && (
                                    <span className="text-red-600">Errors: {queueStats.errors}</span>
                                )}
                            </div>
                        </CardContent>
                    </Card>
                )}

                {/* Summary Cards */}
                {summary && (
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
                        <Card className="bg-slate-800/50 border-slate-700">
                            <CardContent className="p-4">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm font-medium text-gray-300">Total Podcasts</p>
                                        <p className="text-2xl font-bold text-white">{summary.total_podcasts}</p>
                                    </div>
                                    <Mic className="h-8 w-8 text-purple-400" />
                                </div>
                            </CardContent>
                        </Card>

                        <Card className="bg-slate-800/50 border-slate-700">
                            <CardContent className="p-4">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm font-medium text-gray-300">Completed</p>
                                        <p className="text-2xl font-bold text-green-400">{summary.completed_podcasts}</p>
                                    </div>
                                    <Volume2 className="h-8 w-8 text-green-400" />
                                </div>
                            </CardContent>
                        </Card>

                        <Card className="bg-slate-800/50 border-slate-700">
                            <CardContent className="p-4">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm font-medium text-gray-300">Generating</p>
                                        <p className="text-2xl font-bold text-blue-400">{summary.pending_podcasts}</p>
                                    </div>
                                    <Loader2 className="h-8 w-8 text-blue-400" />
                                </div>
                            </CardContent>
                        </Card>

                        <Card className="bg-slate-800/50 border-slate-700">
                            <CardContent className="p-4">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm font-medium text-gray-300">Failed</p>
                                        <p className="text-2xl font-bold text-red-400">{summary.failed_podcasts}</p>
                                    </div>
                                    <Clock className="h-8 w-8 text-red-400" />
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                )}

                {/* Search and Filter Controls */}
                <div className="flex flex-col lg:flex-row gap-4 mb-6">
                    <div className="flex-1">
                        <div className="relative">
                            <Search className="h-4 w-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                            <input
                                type="text"
                                placeholder="Search podcasts..."
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                className="w-full pl-10 pr-4 py-2 bg-slate-800/50 border border-slate-600 text-white placeholder-gray-400 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                            />
                        </div>
                    </div>

                    <div className="flex gap-2">
                        <select
                            value={statusFilter}
                            onChange={(e) => setStatusFilter(e.target.value)}
                            className="px-3 py-2 bg-slate-800/50 border border-slate-600 text-white rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                        >
                            <option value="">All Status</option>
                            <option value="completed">Completed</option>
                            <option value="generating">Generating</option>
                            <option value="pending">Pending</option>
                            <option value="failed">Failed</option>
                        </select>

                        <Button
                            variant="outline"
                            onClick={loadData}
                            disabled={loading}
                            className="border-purple-500/50 text-purple-300 hover:bg-purple-500/20 hover:text-purple-200 hover:border-purple-400 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                        </Button>
                    </div>
                </div>

                {/* Loading State */}
                {loading && (
                    <div className="flex items-center justify-center py-12">
                        <Loader2 className="h-8 w-8 animate-spin text-purple-400" />
                    </div>
                )}

                {/* Podcasts Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {filteredPodcasts.map((podcast) => {
                        // Find if this podcast is in the generation queue
                        const queueItem = generationQueue.find(item => item.podcastId === podcast.id);
                        const isGenerating = queueItem && (queueItem.status === 'active' || queueItem.status === 'queued');
                        const generationProgress = queueItem?.progress || 0;
                        const generationPhase = queueItem?.phase;

                        return (
                            <PodcastCard
                                key={podcast.id}
                                podcast={podcast}
                                onDelete={() => handleDelete(podcast)}
                                isGenerating={!!isGenerating}
                                generationProgress={generationProgress}
                                generationPhase={generationPhase}
                                onViewProgress={() => {
                                    if (queueItem) {
                                        setCurrentGenerationId(queueItem.id);
                                        setShowProgressModal(true);
                                    }
                                }}
                            />
                        );
                    })}
                </div>

                {/* Pagination */}
                {totalPages > 1 && (
                    <div className="flex justify-center mt-8">
                        <div className="flex items-center space-x-2">
                            <Button
                                variant="outline"
                                onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                                disabled={currentPage === 1}
                                className="border-purple-500/50 text-purple-300 hover:bg-purple-500/20 hover:text-purple-200 hover:border-purple-400 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                Previous
                            </Button>

                            <span className="px-4 py-2 text-sm text-purple-300">
                                Page {currentPage} of {totalPages}
                            </span>

                            <Button
                                variant="outline"
                                onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                                disabled={currentPage === totalPages}
                                className="border-purple-500/50 text-purple-300 hover:bg-purple-500/20 hover:text-purple-200 hover:border-purple-400 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                Next
                            </Button>
                        </div>
                    </div>
                )}

                {/* Empty State */}
                {!loading && filteredPodcasts.length === 0 && (
                    <div className="text-center py-12">
                        <Mic className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                        <h3 className="text-lg font-medium text-white mb-2">No podcasts found</h3>
                        <p className="text-gray-300 mb-4">
                            {searchTerm ? 'Try adjusting your search terms' : 'Create your first AI-generated podcast'}
                        </p>
                        {!searchTerm && (
                            <Button
                                onClick={() => setShowCreateForm(true)}
                                className="bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white border-0 font-semibold shadow-lg hover:shadow-purple-500/25 transition-all duration-200"
                            >
                                <Plus className="h-4 w-4 mr-2" />
                                Create Podcast
                            </Button>
                        )}
                    </div>
                )}

                {/* Generation Progress Modal */}
                <GenerationProgressModal
                    isOpen={showProgressModal}
                    onClose={handleCloseProgressModal}
                    podcastTitle={
                        currentGenerationId
                            ? generationQueue.find(item => item.id === currentGenerationId)?.podcastTitle || 'Unknown Podcast'
                            : 'Unknown Podcast'
                    }
                    generationId={currentGenerationId || undefined}
                    onGenerationComplete={handleGenerationComplete}
                    onGenerationError={handleGenerationError}
                />

                {/* Create Podcast Modal */}
                {showCreateForm && (
                    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                        <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 w-full max-w-md mx-4">
                            <h2 className="text-xl font-bold mb-4 text-white">Create New Podcast</h2>
                            <form onSubmit={handleCreatePodcast}>
                                <div className="space-y-4">
                                    <div>
                                        <label className="block text-sm font-medium mb-1 text-gray-300">Title</label>
                                        <input
                                            type="text"
                                            value={newPodcast.title}
                                            onChange={(e) => setNewPodcast(prev => ({ ...prev, title: e.target.value }))}
                                            className="w-full bg-slate-700 border border-slate-600 text-white placeholder-gray-400 rounded-md px-3 py-2 focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                                            placeholder="Enter podcast title"
                                            required
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium mb-1 text-gray-300">Topic</label>
                                        <textarea
                                            value={newPodcast.topic}
                                            onChange={(e) => setNewPodcast(prev => ({ ...prev, topic: e.target.value }))}
                                            className="w-full bg-slate-700 border border-slate-600 text-white placeholder-gray-400 rounded-md px-3 py-2 focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                                            placeholder="Describe the topic for your podcast"
                                            rows={3}
                                            required
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium mb-1 text-gray-300">Length (minutes)</label>
                                        <select
                                            value={newPodcast.length}
                                            onChange={(e) => setNewPodcast(prev => ({ ...prev, length: parseInt(e.target.value) }))}
                                            className="w-full bg-slate-700 border border-slate-600 text-white rounded-md px-3 py-2 focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                                        >
                                            <option value={5}>5 minutes</option>
                                            <option value={10}>10 minutes</option>
                                            <option value={15}>15 minutes</option>
                                            <option value={20}>20 minutes</option>
                                            <option value={30}>30 minutes</option>
                                        </select>
                                    </div>

                                    {/* Voice Selection Section */}
                                    <div className="border-t border-slate-600 pt-4">
                                        <div className="space-y-3">
                                            <div className="mb-4">
                                                <div className="flex items-center justify-between mb-2">
                                                    <h4 className="text-sm font-medium text-gray-300 flex items-center">
                                                        <Volume2 className="h-4 w-4 mr-2 text-purple-400" />
                                                        Voice Selection for Hosts
                                                    </h4>
                                                    <Button
                                                        type="button"
                                                        variant="outline"
                                                        size="sm"
                                                        onClick={() => setShowVoiceCloneModal(true)}
                                                        className="border-purple-500/50 text-purple-300 hover:bg-purple-500/20 hover:text-purple-200 hover:border-purple-400 transition-all duration-200"
                                                    >
                                                        <Mic className="h-3 w-3 mr-1" />
                                                        Clone Voice
                                                    </Button>
                                                </div>
                                                <p className="text-xs text-gray-400">
                                                    Choose voices for your podcast hosts from our curated selection
                                                </p>
                                            </div>

                                            <div className="space-y-3 pl-6 border-l-2 border-purple-500/30">
                                                {userVoices.length === 0 && systemVoices.length === 0 ? (
                                                    <p className="text-sm text-gray-400 italic">
                                                        No voices available. Clone your voice or contact support!
                                                    </p>
                                                ) : (
                                                    <>
                                                        <div>
                                                            <label className="block text-xs font-medium mb-1 text-gray-400">Host 1 Voice</label>
                                                            <select
                                                                value={voiceSettings.host1_voice_id}
                                                                onChange={(e) => setVoiceSettings(prev => ({ ...prev, host1_voice_id: e.target.value }))}
                                                                className="w-full bg-slate-700 border border-slate-600 text-white text-sm rounded-md px-2 py-1 focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                                                            >
                                                                <option value="">⚠️  Select Host 1 voice (required)</option>

                                                                {/* System Voices Section */}
                                                                {systemVoices.length > 0 && (
                                                                    <optgroup label="📢 Podcast Voices (Available to All)">
                                                                        {systemVoices
                                                                            .filter(voice => voice.gender === 'male')
                                                                            .map(voice => (
                                                                                <option key={voice.voice_id} value={voice.voice_id}>
                                                                                    👨 {voice.name}
                                                                                </option>
                                                                            ))}
                                                                        {systemVoices
                                                                            .filter(voice => voice.gender === 'female')
                                                                            .map(voice => (
                                                                                <option key={voice.voice_id} value={voice.voice_id}>
                                                                                    👩 {voice.name}
                                                                                </option>
                                                                            ))}
                                                                    </optgroup>
                                                                )}

                                                                {/* User Custom Voices Section */}
                                                                {userVoices.length > 0 && (
                                                                    <optgroup label="🎤 My Custom Voices">
                                                                        {userVoices.map(voice => (
                                                                            <option key={voice.voice_id} value={voice.voice_id}>
                                                                                🔒 {voice.name}
                                                                            </option>
                                                                        ))}
                                                                    </optgroup>
                                                                )}
                                                            </select>
                                                        </div>

                                                        <div>
                                                            <label className="block text-xs font-medium mb-1 text-gray-400">Host 2 Voice</label>
                                                            <select
                                                                value={voiceSettings.host2_voice_id}
                                                                onChange={(e) => setVoiceSettings(prev => ({ ...prev, host2_voice_id: e.target.value }))}
                                                                className="w-full bg-slate-700 border border-slate-600 text-white text-sm rounded-md px-2 py-1 focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                                                            >
                                                                <option value="">⚠️  Select Host 2 voice (required)</option>

                                                                {/* System Voices Section */}
                                                                {systemVoices.length > 0 && (
                                                                    <optgroup label="📢 Podcast Voices (Available to All)">
                                                                        {systemVoices
                                                                            .filter(voice => voice.gender === 'male')
                                                                            .map(voice => (
                                                                                <option key={voice.voice_id} value={voice.voice_id}>
                                                                                    👨 {voice.name}
                                                                                </option>
                                                                            ))}
                                                                        {systemVoices
                                                                            .filter(voice => voice.gender === 'female')
                                                                            .map(voice => (
                                                                                <option key={voice.voice_id} value={voice.voice_id}>
                                                                                    👩 {voice.name}
                                                                                </option>
                                                                            ))}
                                                                    </optgroup>
                                                                )}

                                                                {/* User Custom Voices Section */}
                                                                {userVoices.length > 0 && (
                                                                    <optgroup label="🎤 My Custom Voices">
                                                                        {userVoices.map(voice => (
                                                                            <option key={voice.voice_id} value={voice.voice_id}>
                                                                                🔒 {voice.name}
                                                                            </option>
                                                                        ))}
                                                                    </optgroup>
                                                                )}
                                                            </select>
                                                        </div>
                                                    </>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div className="flex justify-end space-x-3 mt-6">
                                    <Button
                                        type="button"
                                        variant="outline"
                                        onClick={() => setShowCreateForm(false)}
                                        className="border-slate-600 text-slate-300 hover:bg-slate-700 hover:text-white transition-all duration-200"
                                    >
                                        Cancel
                                    </Button>
                                    <Button
                                        type="submit"
                                        className="bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white border-0 shadow-lg transition-all duration-200"
                                    >
                                        <ArrowLeft className="h-4 w-4 mr-2 rotate-180" />
                                        Continue
                                    </Button>
                                </div>
                            </form>
                        </div>
                    </div>
                )}

                {/* Credit Confirmation Modal */}
                {showCreditConfirmation && pendingPodcastData && (
                    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                        <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 w-full max-w-md mx-4">
                            <div className="text-center">
                                <div className="flex items-center justify-center w-12 h-12 mx-auto mb-4 bg-yellow-500/20 rounded-full">
                                    <AlertCircle className="h-6 w-6 text-yellow-400" />
                                </div>
                                <h2 className="text-xl font-bold mb-4 text-white">Confirm Podcast Generation</h2>

                                <div className="bg-slate-700/50 rounded-lg p-4 mb-6 text-left">
                                    <h3 className="font-semibold text-white mb-2">Podcast Details:</h3>
                                    <p className="text-sm text-gray-300 mb-1">
                                        <span className="font-medium">Title:</span> {pendingPodcastData.title}
                                    </p>
                                    <p className="text-sm text-gray-300 mb-1">
                                        <span className="font-medium">Duration:</span> {pendingPodcastData.length} minutes
                                    </p>
                                    <p className="text-sm text-gray-300">
                                        <span className="font-medium">Topic:</span> {pendingPodcastData.topic}
                                    </p>
                                </div>

                                <div className="bg-purple-500/20 border border-purple-500/30 rounded-lg p-4 mb-6">
                                    <div className="flex items-center gap-2 mb-2">
                                        <Volume2 className="h-4 w-4 text-purple-400" />
                                        <span className="text-sm font-medium text-purple-300">Credit Usage</span>
                                    </div>
                                    <p className="text-sm text-purple-200 mb-2">
                                        Your podcast will be generated with high-quality AI voices and content.
                                    </p>
                                    <p className="text-sm text-purple-200">
                                        <span className="font-medium">Credits will be deducted</span> only after successful generation and completion.
                                    </p>
                                </div>

                                <div className="flex flex-col gap-3">
                                    <Button
                                        onClick={handleConfirmCreateAndGenerate}
                                        disabled={creating}
                                        className="w-full bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white border-0 shadow-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        {creating ? (
                                            <>
                                                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                                Creating & Generating...
                                            </>
                                        ) : (
                                            <>
                                                <Mic className="h-4 w-4 mr-2" />
                                                Confirm & Generate Podcast
                                            </>
                                        )}
                                    </Button>

                                    <Button
                                        variant="outline"
                                        onClick={handleCancelCreate}
                                        disabled={creating}
                                        className="w-full border-slate-600 text-slate-300 hover:bg-slate-700 hover:text-white transition-all duration-200"
                                    >
                                        Go Back to Edit
                                    </Button>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Removed Generation Options Modal - only one generation when creating podcasts */}
            </main>

            {/* Voice Clone Modal */}
            <VoiceCloneModal
                isOpen={showVoiceCloneModal}
                onClose={() => setShowVoiceCloneModal(false)}
                onSuccess={handleVoiceCloneSuccess}
            />
        </div>
    );
} 