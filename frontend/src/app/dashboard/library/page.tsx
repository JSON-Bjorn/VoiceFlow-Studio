'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import PodcastCard from '@/components/PodcastCard';
import { CreditBalance } from '@/components/ui/credit-balance';
import {
    Plus,
    Filter,
    Search,
    Loader2,
    RefreshCw,
    Mic,
    Volume2,
    Settings
} from 'lucide-react';
import {
    Podcast,
    PodcastListResponse,
    PodcastSummary,
    PodcastCreate,
    CreditSummary,
    api
} from '@/lib/api';

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

    // Generate podcast with enhanced AI pipeline
    const handleGenerateEnhanced = async (podcast: Podcast) => {
        try {
            setGenerating(podcast.id);

            const result = await api.generatePodcastWithAI(podcast.id, generationOptions);

            if (result.success) {
                // Update the podcast in the list
                setPodcasts(prev =>
                    prev.map(p => p.id === podcast.id ? { ...p, status: 'completed' } : p)
                );

                // Show success message with voice generation info
                const message = result.voice_generated
                    ? `Podcast generated successfully with voice! Duration: ${result.total_duration?.toFixed(1)}s, Cost: $${result.cost_estimate?.toFixed(4)}`
                    : 'Podcast script generated successfully!';

                alert(message);
            } else {
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
            alert(error.message || 'Failed to generate podcast');
        } finally {
            setGenerating(null);
            setShowGenerationOptions(null);
        }
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

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-50 to-indigo-100">
            <div className="container mx-auto px-4 py-8">
                {/* Header */}
                <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between mb-8">
                    <div>
                        <h1 className="text-3xl font-bold text-gray-900 mb-2">My Podcast Library</h1>
                        <p className="text-gray-600">Manage and listen to your AI-generated podcasts</p>
                    </div>
                    <div className="mt-4 lg:mt-0">
                        {creditSummary && (
                            <CreditBalance
                                credits={creditSummary.current_balance}
                                variant="header"
                            />
                        )}
                    </div>
                </div>

                {/* Summary Cards */}
                {summary && (
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
                        <Card>
                            <CardContent className="p-4">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-gray-600">Total Podcasts</p>
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
                                        <p className="text-sm text-gray-600">Completed</p>
                                        <p className="text-2xl font-bold text-green-600">{summary.completed_podcasts}</p>
                                    </div>
                                    <div className="h-8 w-8 rounded-full bg-green-100 flex items-center justify-center">
                                        <div className="h-3 w-3 rounded-full bg-green-600"></div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                        <Card>
                            <CardContent className="p-4">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-gray-600">Pending</p>
                                        <p className="text-2xl font-bold text-yellow-600">{summary.pending_podcasts}</p>
                                    </div>
                                    <div className="h-8 w-8 rounded-full bg-yellow-100 flex items-center justify-center">
                                        <div className="h-3 w-3 rounded-full bg-yellow-600"></div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                        <Card>
                            <CardContent className="p-4">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-gray-600">Failed</p>
                                        <p className="text-2xl font-bold text-red-600">{summary.failed_podcasts}</p>
                                    </div>
                                    <div className="h-8 w-8 rounded-full bg-red-100 flex items-center justify-center">
                                        <div className="h-3 w-3 rounded-full bg-red-600"></div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                )}

                {/* Controls */}
                <div className="flex flex-col lg:flex-row gap-4 mb-8">
                    <div className="flex-1">
                        <div className="relative">
                            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
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
                            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                        >
                            <option value="">All Status</option>
                            <option value="completed">Completed</option>
                            <option value="pending">Pending</option>
                            <option value="generating">Generating</option>
                            <option value="failed">Failed</option>
                        </select>

                        <Button
                            onClick={loadData}
                            variant="outline"
                            disabled={loading}
                        >
                            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                        </Button>

                        <Button
                            onClick={() => setShowCreateForm(true)}
                            className="bg-purple-600 hover:bg-purple-700"
                        >
                            <Plus className="mr-2 h-4 w-4" />
                            New Podcast
                        </Button>
                    </div>
                </div>

                {/* Create Podcast Form */}
                {showCreateForm && (
                    <Card className="mb-8">
                        <CardHeader>
                            <CardTitle>Create New Podcast</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <form onSubmit={handleCreatePodcast} className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">
                                        Podcast Title
                                    </label>
                                    <input
                                        type="text"
                                        value={newPodcast.title}
                                        onChange={(e) => setNewPodcast(prev => ({ ...prev, title: e.target.value }))}
                                        placeholder="Enter podcast title..."
                                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                        required
                                    />
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">
                                        Topic/Description
                                    </label>
                                    <textarea
                                        value={newPodcast.topic}
                                        onChange={(e) => setNewPodcast(prev => ({ ...prev, topic: e.target.value }))}
                                        placeholder="Describe what you want the podcast to be about..."
                                        rows={3}
                                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                        required
                                    />
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">
                                        Length (minutes)
                                    </label>
                                    <select
                                        value={newPodcast.length}
                                        onChange={(e) => setNewPodcast(prev => ({ ...prev, length: parseInt(e.target.value) }))}
                                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                    >
                                        <option value={5}>5 minutes</option>
                                        <option value={10}>10 minutes</option>
                                        <option value={15}>15 minutes</option>
                                        <option value={20}>20 minutes</option>
                                        <option value={30}>30 minutes</option>
                                        <option value={45}>45 minutes</option>
                                        <option value={60}>60 minutes</option>
                                    </select>
                                </div>

                                <div className="flex gap-2">
                                    <Button
                                        type="submit"
                                        disabled={creating}
                                        className="bg-purple-600 hover:bg-purple-700"
                                    >
                                        {creating ? (
                                            <>
                                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                                Creating...
                                            </>
                                        ) : (
                                            'Create Podcast'
                                        )}
                                    </Button>
                                    <Button
                                        type="button"
                                        variant="outline"
                                        onClick={() => setShowCreateForm(false)}
                                    >
                                        Cancel
                                    </Button>
                                </div>
                            </form>
                        </CardContent>
                    </Card>
                )}

                {/* Podcasts Grid */}
                {loading ? (
                    <div className="flex items-center justify-center py-12">
                        <Loader2 className="h-8 w-8 animate-spin text-purple-600" />
                    </div>
                ) : filteredPodcasts.length === 0 ? (
                    <Card>
                        <CardContent className="text-center py-12">
                            <Mic className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                            <h3 className="text-lg font-semibold text-gray-900 mb-2">No podcasts found</h3>
                            <p className="text-gray-600 mb-4">
                                {searchTerm || statusFilter
                                    ? 'Try adjusting your search or filter criteria.'
                                    : 'Create your first AI-generated podcast to get started!'
                                }
                            </p>
                            {!searchTerm && !statusFilter && (
                                <Button
                                    onClick={() => setShowCreateForm(true)}
                                    className="bg-purple-600 hover:bg-purple-700"
                                >
                                    <Plus className="mr-2 h-4 w-4" />
                                    Create Your First Podcast
                                </Button>
                            )}
                        </CardContent>
                    </Card>
                ) : (
                    <>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {filteredPodcasts.map(podcast => (
                                <PodcastCard
                                    key={podcast.id}
                                    podcast={podcast}
                                    onGenerate={showGenerationModal}
                                    onDelete={handleDelete}
                                    isGenerating={generating === podcast.id}
                                />
                            ))}
                        </div>

                        {/* Pagination */}
                        {totalPages > 1 && (
                            <div className="flex justify-center items-center space-x-2 mt-8">
                                <Button
                                    variant="outline"
                                    onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                                    disabled={currentPage === 1}
                                >
                                    Previous
                                </Button>
                                <span className="text-sm text-gray-600">
                                    Page {currentPage} of {totalPages}
                                </span>
                                <Button
                                    variant="outline"
                                    onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                                    disabled={currentPage === totalPages}
                                >
                                    Next
                                </Button>
                            </div>
                        )}
                    </>
                )}

                {/* Generation Options Modal */}
                {showGenerationOptions && (
                    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                        <Card className="w-full max-w-2xl mx-4 max-h-[90vh] overflow-y-auto">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Settings className="h-5 w-5" />
                                    Enhanced AI Generation Options
                                </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-6">
                                {/* Voice Generation Toggle */}
                                <div className="flex items-center justify-between p-4 border rounded-lg">
                                    <div className="flex items-center gap-3">
                                        <Volume2 className="h-5 w-5 text-purple-600" />
                                        <div>
                                            <h3 className="font-medium">Voice Generation</h3>
                                            <p className="text-sm text-gray-600">Generate AI voices for your podcast</p>
                                        </div>
                                    </div>
                                    <label className="relative inline-flex items-center cursor-pointer">
                                        <input
                                            type="checkbox"
                                            checked={generationOptions.generateVoice}
                                            onChange={(e) => setGenerationOptions(prev => ({
                                                ...prev,
                                                generateVoice: e.target.checked
                                            }))}
                                            className="sr-only peer"
                                        />
                                        <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
                                    </label>
                                </div>

                                {/* Host Personalities */}
                                <div className="space-y-4">
                                    <h3 className="font-medium">Host Personalities</h3>

                                    {/* Host 1 */}
                                    <div className="p-4 border rounded-lg space-y-3">
                                        <h4 className="font-medium text-sm">Host 1 (Primary Questioner)</h4>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                            <div>
                                                <label className="block text-xs font-medium text-gray-700 mb-1">Name</label>
                                                <input
                                                    type="text"
                                                    value={generationOptions.hostPersonalities.host_1.name}
                                                    onChange={(e) => setGenerationOptions(prev => ({
                                                        ...prev,
                                                        hostPersonalities: {
                                                            ...prev.hostPersonalities,
                                                            host_1: { ...prev.hostPersonalities.host_1, name: e.target.value }
                                                        }
                                                    }))}
                                                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-purple-500"
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-xs font-medium text-gray-700 mb-1">Personality</label>
                                                <input
                                                    type="text"
                                                    value={generationOptions.hostPersonalities.host_1.personality}
                                                    onChange={(e) => setGenerationOptions(prev => ({
                                                        ...prev,
                                                        hostPersonalities: {
                                                            ...prev.hostPersonalities,
                                                            host_1: { ...prev.hostPersonalities.host_1, personality: e.target.value }
                                                        }
                                                    }))}
                                                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-purple-500"
                                                />
                                            </div>
                                        </div>
                                    </div>

                                    {/* Host 2 */}
                                    <div className="p-4 border rounded-lg space-y-3">
                                        <h4 className="font-medium text-sm">Host 2 (Storyteller)</h4>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                            <div>
                                                <label className="block text-xs font-medium text-gray-700 mb-1">Name</label>
                                                <input
                                                    type="text"
                                                    value={generationOptions.hostPersonalities.host_2.name}
                                                    onChange={(e) => setGenerationOptions(prev => ({
                                                        ...prev,
                                                        hostPersonalities: {
                                                            ...prev.hostPersonalities,
                                                            host_2: { ...prev.hostPersonalities.host_2, name: e.target.value }
                                                        }
                                                    }))}
                                                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-purple-500"
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-xs font-medium text-gray-700 mb-1">Personality</label>
                                                <input
                                                    type="text"
                                                    value={generationOptions.hostPersonalities.host_2.personality}
                                                    onChange={(e) => setGenerationOptions(prev => ({
                                                        ...prev,
                                                        hostPersonalities: {
                                                            ...prev.hostPersonalities,
                                                            host_2: { ...prev.hostPersonalities.host_2, personality: e.target.value }
                                                        }
                                                    }))}
                                                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-purple-500"
                                                />
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Style Preferences */}
                                <div className="space-y-4">
                                    <h3 className="font-medium">Style Preferences</h3>
                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-xs font-medium text-gray-700 mb-1">Tone</label>
                                            <select
                                                value={generationOptions.stylePreferences.tone}
                                                onChange={(e) => setGenerationOptions(prev => ({
                                                    ...prev,
                                                    stylePreferences: { ...prev.stylePreferences, tone: e.target.value }
                                                }))}
                                                className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-purple-500"
                                            >
                                                <option value="conversational">Conversational</option>
                                                <option value="formal">Formal</option>
                                                <option value="casual">Casual</option>
                                                <option value="professional">Professional</option>
                                            </select>
                                        </div>
                                        <div>
                                            <label className="block text-xs font-medium text-gray-700 mb-1">Complexity</label>
                                            <select
                                                value={generationOptions.stylePreferences.complexity}
                                                onChange={(e) => setGenerationOptions(prev => ({
                                                    ...prev,
                                                    stylePreferences: { ...prev.stylePreferences, complexity: e.target.value }
                                                }))}
                                                className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-purple-500"
                                            >
                                                <option value="accessible">Accessible</option>
                                                <option value="intermediate">Intermediate</option>
                                                <option value="advanced">Advanced</option>
                                                <option value="expert">Expert</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                {/* Audio Options */}
                                <div className="space-y-4">
                                    <h3 className="font-medium">Audio Options</h3>
                                    <div className="flex items-center justify-between p-4 border rounded-lg">
                                        <div className="flex items-center gap-3">
                                            <Volume2 className="h-5 w-5 text-purple-600" />
                                            <div>
                                                <h4 className="font-medium">Intro Music</h4>
                                                <p className="text-sm text-gray-600">Add intro music to your podcast</p>
                                            </div>
                                        </div>
                                        <label className="relative inline-flex items-center cursor-pointer">
                                            <input
                                                type="checkbox"
                                                checked={generationOptions.audio_options.add_intro}
                                                onChange={(e) => setGenerationOptions(prev => ({
                                                    ...prev,
                                                    audio_options: { ...prev.audio_options, add_intro: e.target.checked }
                                                }))}
                                                className="sr-only peer"
                                            />
                                            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
                                        </label>
                                    </div>
                                    <div className="flex items-center justify-between p-4 border rounded-lg">
                                        <div className="flex items-center gap-3">
                                            <Volume2 className="h-5 w-5 text-purple-600" />
                                            <div>
                                                <h4 className="font-medium">Outro Music</h4>
                                                <p className="text-sm text-gray-600">Add outro music to your podcast</p>
                                            </div>
                                        </div>
                                        <label className="relative inline-flex items-center cursor-pointer">
                                            <input
                                                type="checkbox"
                                                checked={generationOptions.audio_options.add_outro}
                                                onChange={(e) => setGenerationOptions(prev => ({
                                                    ...prev,
                                                    audio_options: { ...prev.audio_options, add_outro: e.target.checked }
                                                }))}
                                                className="sr-only peer"
                                            />
                                            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
                                        </label>
                                    </div>
                                    <div className="flex items-center justify-between p-4 border rounded-lg">
                                        <div className="flex items-center gap-3">
                                            <Volume2 className="h-5 w-5 text-purple-600" />
                                            <div>
                                                <h4 className="font-medium">Background Music</h4>
                                                <p className="text-sm text-gray-600">Add background music to your podcast</p>
                                            </div>
                                        </div>
                                        <label className="relative inline-flex items-center cursor-pointer">
                                            <input
                                                type="checkbox"
                                                checked={generationOptions.audio_options.add_background_music}
                                                onChange={(e) => setGenerationOptions(prev => ({
                                                    ...prev,
                                                    audio_options: { ...prev.audio_options, add_background_music: e.target.checked }
                                                }))}
                                                className="sr-only peer"
                                            />
                                            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
                                        </label>
                                    </div>
                                </div>

                                {/* Action Buttons */}
                                <div className="flex gap-3 pt-4">
                                    <Button
                                        onClick={() => {
                                            const podcast = filteredPodcasts.find(p => p.id === showGenerationOptions);
                                            if (podcast) handleGenerateEnhanced(podcast);
                                        }}
                                        disabled={generating !== null}
                                        className="flex-1 bg-purple-600 hover:bg-purple-700"
                                    >
                                        {generating ? (
                                            <>
                                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                                Generating...
                                            </>
                                        ) : (
                                            <>
                                                <Volume2 className="mr-2 h-4 w-4" />
                                                Generate {generationOptions.generateVoice ? 'with Voice' : 'Script'}
                                            </>
                                        )}
                                    </Button>
                                    <Button
                                        variant="outline"
                                        onClick={() => setShowGenerationOptions(null)}
                                        disabled={generating !== null}
                                    >
                                        Cancel
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                )}
            </div>
        </div>
    );
} 