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
    Mic
} from 'lucide-react';
import {
    Podcast,
    PodcastListResponse,
    PodcastSummary,
    PodcastCreate,
    CreditSummary,
    apiClient
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
    const [newPodcast, setNewPodcast] = useState<PodcastCreate>({
        title: '',
        topic: '',
        length: 10
    });

    // Load podcasts and summary
    const loadData = async () => {
        try {
            setLoading(true);
            const [podcastsResponse, summaryResponse, creditResponse] = await Promise.all([
                apiClient.getPodcasts(currentPage, 12, statusFilter || undefined),
                apiClient.getPodcastSummary(),
                apiClient.getCreditSummary()
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
            const podcast = await apiClient.createPodcast(newPodcast);
            setPodcasts(prev => [podcast, ...prev]);
            setNewPodcast({ title: '', topic: '', length: 10 });
            setShowCreateForm(false);

            // Refresh summary
            const summaryResponse = await apiClient.getPodcastSummary();
            setSummary(summaryResponse);

            // Refresh credit summary
            const creditResponse = await apiClient.getCreditSummary();
            setCreditSummary(creditResponse);
        } catch (error: any) {
            alert(error.message || 'Failed to create podcast');
        } finally {
            setCreating(false);
        }
    };

    // Generate podcast
    const handleGenerate = async (podcast: Podcast) => {
        try {
            setGenerating(podcast.id);
            const updatedPodcast = await apiClient.simulatePodcastGeneration(podcast.id);
            setPodcasts(prev =>
                prev.map(p => p.id === podcast.id ? updatedPodcast : p)
            );

            // Refresh summary
            const summaryResponse = await apiClient.getPodcastSummary();
            setSummary(summaryResponse);

            // Refresh credit summary
            const creditResponse = await apiClient.getCreditSummary();
            setCreditSummary(creditResponse);
        } catch (error: any) {
            alert(error.message || 'Failed to generate podcast');
        } finally {
            setGenerating(null);
        }
    };

    // Delete podcast
    const handleDelete = async (podcast: Podcast) => {
        if (!confirm(`Are you sure you want to delete "${podcast.title}"?`)) return;

        try {
            await apiClient.deletePodcast(podcast.id);
            setPodcasts(prev => prev.filter(p => p.id !== podcast.id));

            // Refresh summary
            const summaryResponse = await apiClient.getPodcastSummary();
            setSummary(summaryResponse);

            // Refresh credit summary
            const creditResponse = await apiClient.getCreditSummary();
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
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {filteredPodcasts.map((podcast) => (
                            <PodcastCard
                                key={podcast.id}
                                podcast={podcast}
                                onGenerate={handleGenerate}
                                onDelete={handleDelete}
                                isGenerating={generating === podcast.id}
                            />
                        ))}
                    </div>
                )}

                {/* Pagination */}
                {totalPages > 1 && (
                    <div className="flex items-center justify-center gap-2 mt-8">
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
                )}
            </div>
        </div>
    );
} 