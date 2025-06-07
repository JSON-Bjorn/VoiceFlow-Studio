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
    const [showGenerationOptions, setShowGenerationOptions] = useState<number | null>(null);
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
        length: 10
    });

    // Generation options state
    const [generationOptions, setGenerationOptions] = useState({
        generateVoice: true,
        assemble_audio: true,
        hostPersonalities: {
            host_1: { name: 'Felix', personality: 'analytical, thoughtful, engaging', role: 'primary_questioner' },
            host_2: { name: 'Bjorn', personality: 'warm, curious, conversational', role: 'storyteller' }
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

    useEffect(() => {
        loadData();
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

        // Store the podcast data and show credit confirmation
        setPendingPodcastData({ ...newPodcast });
        setShowCreateForm(false);
        setShowCreditConfirmation(true);
    };

    // Confirm and create podcast with immediate generation
    const handleConfirmCreateAndGenerate = async () => {
        if (!pendingPodcastData) return;

        try {
            setCreating(true);
            setShowCreditConfirmation(false);

            // Create the podcast
            const podcast = await api.createPodcast(pendingPodcastData);
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

    // Start background generation
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

            // Start the generation process
            const result = await api.generatePodcastWithAI(podcast.id, generationOptions);

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

    // Enhanced generation with queue management (for existing podcasts)
    const handleGenerateEnhanced = async (podcast: Podcast) => {
        // Check if this podcast is already generating
        const existingQueueItem = generationQueue.find(item => item.podcastId === podcast.id &&
            (item.status === 'active' || item.status === 'queued'));

        if (existingQueueItem) {
            // Show the progress modal for existing generation
            setCurrentGenerationId(existingQueueItem.id);
            setShowProgressModal(true);
            return;
        }

        // Start new background generation
        startBackgroundGeneration(podcast);
    };

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

    // Show generation options modal
    const showGenerationModal = (podcast: Podcast) => {
        setShowGenerationOptions(podcast.id);
    };

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
                                onGenerate={() => showGenerationModal(podcast)}
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

                {/* Generation Options Modal */}
                {showGenerationOptions && (
                    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                        <div className="bg-white rounded-lg p-6 w-full max-w-2xl mx-4 max-h-[90vh] overflow-y-auto">
                            <h2 className="text-xl font-bold mb-4">Generation Options</h2>

                            <div className="space-y-6">
                                {/* Audio generation is always enabled for podcasts */}
                                <div className="border rounded-lg p-4 bg-green-50">
                                    <div className="flex items-center gap-2 mb-2">
                                        <Volume2 className="h-4 w-4 text-green-600" />
                                        <span className="text-sm font-medium text-green-800">Audio Generation Enabled</span>
                                    </div>
                                    <p className="text-xs text-green-700">
                                        Your podcast will include high-quality voice audio using Chatterbox TTS
                                    </p>
                                </div>

                                {/* Host Personalities */}
                                <div className="border rounded-lg p-4">
                                    <h3 className="font-medium mb-3">Host Personalities</h3>
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                        {Object.entries(generationOptions.hostPersonalities).map(([key, host]) => (
                                            <div key={key} className="space-y-2">
                                                <label className="block text-sm font-medium">
                                                    {key.replace('_', ' ').toUpperCase()}
                                                </label>
                                                <input
                                                    type="text"
                                                    value={host.name}
                                                    onChange={(e) => setGenerationOptions(prev => ({
                                                        ...prev,
                                                        hostPersonalities: {
                                                            ...prev.hostPersonalities,
                                                            [key]: { ...host, name: e.target.value }
                                                        }
                                                    }))}
                                                    className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
                                                    placeholder="Host name"
                                                />
                                                <textarea
                                                    value={host.personality}
                                                    onChange={(e) => setGenerationOptions(prev => ({
                                                        ...prev,
                                                        hostPersonalities: {
                                                            ...prev.hostPersonalities,
                                                            [key]: { ...host, personality: e.target.value }
                                                        }
                                                    }))}
                                                    className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
                                                    placeholder="Personality traits"
                                                    rows={2}
                                                />
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* Audio Options */}
                                <div className="border rounded-lg p-4">
                                    <h3 className="font-medium mb-3">Audio Options</h3>
                                    <div className="grid grid-cols-2 gap-4">
                                        <label className="flex items-center space-x-2">
                                            <input
                                                type="checkbox"
                                                checked={generationOptions.audio_options.add_intro}
                                                onChange={(e) => setGenerationOptions(prev => ({
                                                    ...prev,
                                                    audio_options: { ...prev.audio_options, add_intro: e.target.checked }
                                                }))}
                                            />
                                            <span className="text-sm">Add Intro</span>
                                        </label>
                                        <label className="flex items-center space-x-2">
                                            <input
                                                type="checkbox"
                                                checked={generationOptions.audio_options.add_outro}
                                                onChange={(e) => setGenerationOptions(prev => ({
                                                    ...prev,
                                                    audio_options: { ...prev.audio_options, add_outro: e.target.checked }
                                                }))}
                                            />
                                            <span className="text-sm">Add Outro</span>
                                        </label>
                                    </div>
                                </div>
                            </div>

                            <div className="flex justify-end space-x-3 mt-6">
                                <Button
                                    variant="outline"
                                    onClick={() => setShowGenerationOptions(null)}
                                    className="border-slate-600 text-slate-700 hover:bg-slate-100 hover:text-slate-800 transition-all duration-200"
                                >
                                    Cancel
                                </Button>
                                <Button
                                    onClick={() => {
                                        const podcast = podcasts.find(p => p.id === showGenerationOptions);
                                        if (podcast) {
                                            handleGenerateEnhanced(podcast);
                                        }
                                    }}
                                    disabled={generating !== null}
                                    className="bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white border-0 shadow-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    {generating ? (
                                        <>
                                            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                            Generating...
                                        </>
                                    ) : (
                                        <>
                                            <Mic className="h-4 w-4 mr-2" />
                                            Start Generation
                                        </>
                                    )}
                                </Button>
                            </div>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
} 