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
    Pause
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
        generateVoice: false,
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

    // Create new podcast
    const handleCreatePodcast = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!newPodcast.title.trim() || !newPodcast.topic.trim()) return;

        try {
            setCreating(true);
            const podcast = await api.createPodcast(newPodcast);
            setPodcasts(prev => [podcast, ...prev]);
            setNewPodcast({ title: '', topic: '', length: 10 });
            setShowCreateForm(false);

            // Refresh summary
            const summaryResponse = await api.getPodcastSummary();
            setSummary(summaryResponse);

            // Refresh credit summary
            const creditResponse = await api.getCreditSummary();
            setCreditSummary(creditResponse);
        } catch (error: any) {
            alert(error.message || 'Failed to create podcast');
        } finally {
            setCreating(false);
        }
    };

    // Enhanced generation with queue management
    const handleGenerateEnhanced = async (podcast: Podcast) => {
        try {
            // Add to generation queue
            const generationId = addToQueue(podcast.id, podcast.title);
            setCurrentGenerationId(generationId);
            setShowProgressModal(true);
            setGenerating(podcast.id);

            // Start the generation process
            const result = await api.generatePodcastWithAI(podcast.id, generationOptions);

            if (result.success) {
                // Update queue item as completed
                updateQueueItem(generationId, {
                    status: 'completed',
                    progress: 100,
                    result: result
                });

                // Update the podcast in the list
                setPodcasts(prev =>
                    prev.map(p => p.id === podcast.id ? { ...p, status: 'completed' } : p)
                );

                // Show success message
                const message = result.voice_generated
                    ? `Podcast generated successfully with voice! Duration: ${result.total_duration?.toFixed(1)}s, Cost: $${result.cost_estimate?.toFixed(4)}`
                    : 'Podcast script generated successfully!';

                alert(message);
            } else {
                // Update queue item as error
                updateQueueItem(generationId, {
                    status: 'error',
                    errorMessage: result.message || 'Generation failed'
                });
                alert(`Generation failed: ${result.message}`);
            }

            // Refresh data
            const [summaryResponse, creditResponse] = await Promise.all([
                api.getPodcastSummary(),
                api.getCreditSummary()
            ]);
            setSummary(summaryResponse);
            setCreditSummary(creditResponse);

        } catch (error: any) {
            // Update queue item as error
            if (currentGenerationId) {
                updateQueueItem(currentGenerationId, {
                    status: 'error',
                    errorMessage: error.message || 'Generation failed'
                });
            }
            alert(error.message || 'Failed to generate podcast');
        } finally {
            setGenerating(null);
            setShowGenerationOptions(null);
        }
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
        setCurrentGenerationId(null);
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
        <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-pink-50">
            <div className="container mx-auto px-4 py-8">
                {/* Header Section */}
                <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between mb-8">
                    <div>
                        <h1 className="text-3xl font-bold text-gray-900 mb-2">My Podcast Library</h1>
                        <p className="text-gray-600">Create, manage, and listen to your AI-generated podcasts</p>
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
                                className="flex items-center gap-2"
                            >
                                <Clock className="h-4 w-4" />
                                <span className="hidden sm:inline">Queue</span>
                                <Badge variant="secondary" className="ml-1">
                                    {queueStats.active + queueStats.queued}
                                </Badge>
                            </Button>
                        )}

                        <Button
                            onClick={() => setShowCreateForm(true)}
                            className="bg-purple-600 hover:bg-purple-700 text-white"
                        >
                            <Plus className="h-4 w-4 mr-2" />
                            New Podcast
                        </Button>
                    </div>
                </div>

                {/* Queue Management Panel */}
                {showQueuePanel && queueStats.total > 0 && (
                    <Card className="mb-6 border-purple-200">
                        <CardHeader>
                            <CardTitle className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <Clock className="h-5 w-5 text-purple-600" />
                                    Generation Queue
                                </div>
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => setShowQueuePanel(false)}
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
                        <Card>
                            <CardContent className="p-4">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm font-medium text-gray-600">Total Podcasts</p>
                                        <p className="text-2xl font-bold text-gray-900">{summary.total_podcasts}</p>
                                    </div>
                                    <Mic className="h-8 w-8 text-purple-600" />
                                </div>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardContent className="p-4">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm font-medium text-gray-600">Completed</p>
                                        <p className="text-2xl font-bold text-green-600">{summary.completed_podcasts}</p>
                                    </div>
                                    <Volume2 className="h-8 w-8 text-green-600" />
                                </div>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardContent className="p-4">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm font-medium text-gray-600">Generating</p>
                                        <p className="text-2xl font-bold text-blue-600">{summary.pending_podcasts}</p>
                                    </div>
                                    <Loader2 className="h-8 w-8 text-blue-600" />
                                </div>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardContent className="p-4">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm font-medium text-gray-600">Failed</p>
                                        <p className="text-2xl font-bold text-red-600">{summary.failed_podcasts}</p>
                                    </div>
                                    <Clock className="h-8 w-8 text-red-600" />
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
                                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                            />
                        </div>
                    </div>

                    <div className="flex gap-2">
                        <select
                            value={statusFilter}
                            onChange={(e) => setStatusFilter(e.target.value)}
                            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
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
                        >
                            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                        </Button>
                    </div>
                </div>

                {/* Loading State */}
                {loading && (
                    <div className="flex items-center justify-center py-12">
                        <Loader2 className="h-8 w-8 animate-spin text-purple-600" />
                    </div>
                )}

                {/* Podcasts Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {filteredPodcasts.map((podcast) => (
                        <PodcastCard
                            key={podcast.id}
                            podcast={podcast}
                            onGenerate={() => showGenerationModal(podcast)}
                            onDelete={() => handleDelete(podcast)}
                            isGenerating={generating === podcast.id}
                        />
                    ))}
                </div>

                {/* Pagination */}
                {totalPages > 1 && (
                    <div className="flex justify-center mt-8">
                        <div className="flex items-center space-x-2">
                            <Button
                                variant="outline"
                                onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                                disabled={currentPage === 1}
                            >
                                Previous
                            </Button>

                            <span className="px-4 py-2 text-sm text-gray-600">
                                Page {currentPage} of {totalPages}
                            </span>

                            <Button
                                variant="outline"
                                onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                                disabled={currentPage === totalPages}
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
                        <h3 className="text-lg font-medium text-gray-900 mb-2">No podcasts found</h3>
                        <p className="text-gray-600 mb-4">
                            {searchTerm ? 'Try adjusting your search terms' : 'Create your first AI-generated podcast'}
                        </p>
                        {!searchTerm && (
                            <Button
                                onClick={() => setShowCreateForm(true)}
                                className="bg-purple-600 hover:bg-purple-700 text-white"
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
                        <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
                            <h2 className="text-xl font-bold mb-4">Create New Podcast</h2>
                            <form onSubmit={handleCreatePodcast}>
                                <div className="space-y-4">
                                    <div>
                                        <label className="block text-sm font-medium mb-1">Title</label>
                                        <input
                                            type="text"
                                            value={newPodcast.title}
                                            onChange={(e) => setNewPodcast(prev => ({ ...prev, title: e.target.value }))}
                                            className="w-full border border-gray-300 rounded-md px-3 py-2"
                                            placeholder="Enter podcast title"
                                            required
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium mb-1">Topic</label>
                                        <textarea
                                            value={newPodcast.topic}
                                            onChange={(e) => setNewPodcast(prev => ({ ...prev, topic: e.target.value }))}
                                            className="w-full border border-gray-300 rounded-md px-3 py-2"
                                            placeholder="Describe the topic for your podcast"
                                            rows={3}
                                            required
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium mb-1">Length (minutes)</label>
                                        <select
                                            value={newPodcast.length}
                                            onChange={(e) => setNewPodcast(prev => ({ ...prev, length: parseInt(e.target.value) }))}
                                            className="w-full border border-gray-300 rounded-md px-3 py-2"
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
                                    >
                                        Cancel
                                    </Button>
                                    <Button
                                        type="submit"
                                        disabled={creating}
                                        className="bg-purple-600 hover:bg-purple-700 text-white"
                                    >
                                        {creating ? (
                                            <>
                                                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                                Creating...
                                            </>
                                        ) : (
                                            'Create'
                                        )}
                                    </Button>
                                </div>
                            </form>
                        </div>
                    </div>
                )}

                {/* Generation Options Modal */}
                {showGenerationOptions && (
                    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                        <div className="bg-white rounded-lg p-6 w-full max-w-2xl mx-4 max-h-[90vh] overflow-y-auto">
                            <h2 className="text-xl font-bold mb-4">Generation Options</h2>

                            <div className="space-y-6">
                                {/* Voice Generation Toggle */}
                                <div className="border rounded-lg p-4">
                                    <div className="flex items-center justify-between mb-3">
                                        <label className="text-sm font-medium">Generate Voice Audio</label>
                                        <input
                                            type="checkbox"
                                            checked={generationOptions.generateVoice}
                                            onChange={(e) => setGenerationOptions(prev => ({
                                                ...prev,
                                                generateVoice: e.target.checked
                                            }))}
                                            className="rounded"
                                        />
                                    </div>
                                    <p className="text-xs text-gray-600">
                                        Enable to generate audio using Chatterbox TTS (local processing)
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
                                {generationOptions.generateVoice && (
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
                                )}
                            </div>

                            <div className="flex justify-end space-x-3 mt-6">
                                <Button
                                    variant="outline"
                                    onClick={() => setShowGenerationOptions(null)}
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
                                    className="bg-purple-600 hover:bg-purple-700 text-white"
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
            </div>
        </div>
    );
} 