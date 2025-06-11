'use client';

import { Podcast } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { PodcastDownloadShare } from './PodcastDownloadShare';
import {
    Play,
    Download,
    Clock,
    Calendar,
    MoreVertical,
    Trash2,
    Eye,
    Loader2,
    Bug,
    Volume2,
    HardDrive,
    Radio
} from 'lucide-react';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { formatDistanceToNow } from 'date-fns';

interface PodcastCardProps {
    podcast: Podcast;
    onPlay?: (podcast: Podcast) => void;
    onDelete?: (podcast: Podcast) => void;
    onView?: (podcast: Podcast) => void;
    onGenerate?: (podcast: Podcast) => void;
    isGenerating?: boolean;
    generationProgress?: number;
    generationPhase?: string;
    onViewProgress?: () => void;
}

const getStatusColor = (status: Podcast['status']) => {
    switch (status) {
        case 'completed':
            return 'bg-green-500/20 text-green-400 border-green-500/30';
        case 'generating':
            return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
        case 'pending':
            return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
        case 'failed':
            return 'bg-red-500/20 text-red-400 border-red-500/30';
        default:
            return 'bg-slate-500/20 text-slate-400 border-slate-500/30';
    }
};

const getStatusText = (status: Podcast['status']) => {
    switch (status) {
        case 'completed':
            return 'Ready';
        case 'generating':
            return 'Generating...';
        case 'pending':
            return 'Pending';
        case 'failed':
            return 'Failed';
        default:
            return status;
    }
};

export default function PodcastCard({
    podcast,
    onPlay,
    onDelete,
    onView,
    onGenerate,
    isGenerating = false,
    generationProgress = 0,
    generationPhase,
    onViewProgress
}: PodcastCardProps) {
    const canPlay = podcast.status === 'completed' && podcast.audio_url;
    const canGenerate = podcast.status === 'pending' || podcast.status === 'failed';

    // Debug logging for download issue
    if (podcast.status === 'completed') {
        console.log('Podcast debug info:', {
            id: podcast.id,
            title: podcast.title,
            status: podcast.status,
            has_audio: podcast.has_audio,
            audio_url: podcast.audio_url,
            canPlay: canPlay
        });
    }

    // Debug function to check audio assembly status
    const handleDebugAudio = async () => {
        try {
            const token = localStorage.getItem('access_token');
            if (!token) {
                alert('Please login to debug');
                return;
            }

            const response = await fetch(`http://localhost:8000/api/podcasts/${podcast.id}/audio-debug`, {
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
            });

            if (!response.ok) {
                throw new Error('Debug request failed');
            }

            const debugInfo = await response.json();
            console.log('Audio Debug Info:', debugInfo);

            // Show debug info in alert (in production, use a proper modal)
            const summary = `
Audio Debug for "${podcast.title}":
- Status: ${debugInfo.podcast_status}
- Has Audio: ${debugInfo.has_audio}
- Audio Agent Available: ${debugInfo.audio_agent_available}
- Main Files: ${debugInfo.file_analysis.main_files.length}
- Segment Files: ${debugInfo.file_analysis.segment_files.length}
- Recommendations: ${debugInfo.recommendations.join(', ')}
            `.trim();

            alert(summary);
        } catch (error) {
            console.error('Debug failed:', error);
            alert('Debug request failed');
        }
    };

    return (
        <Card className="bg-slate-800/50 border-slate-700 hover:bg-slate-800/70 transition-all duration-200">
            <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                        <CardTitle className="text-lg font-semibold text-white truncate">
                            {podcast.title}
                        </CardTitle>
                        <p className="text-sm text-gray-300 mt-1 line-clamp-2">
                            {podcast.topic}
                        </p>
                    </div>

                    <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="sm" className="h-8 w-8 p-0 text-slate-300 hover:text-white hover:bg-slate-700">
                                <MoreVertical className="h-4 w-4" />
                            </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end" className="bg-slate-800 border-slate-700">
                            {onView && (
                                <DropdownMenuItem onClick={() => onView(podcast)}>
                                    <Eye className="h-4 w-4 mr-2" />
                                    View Details
                                </DropdownMenuItem>
                            )}

                            {podcast.status === 'completed' && (
                                <DropdownMenuItem onClick={handleDebugAudio}>
                                    <Bug className="h-4 w-4 mr-2" />
                                    Debug Audio
                                </DropdownMenuItem>
                            )}

                            {onDelete && (
                                <DropdownMenuItem
                                    onClick={() => onDelete(podcast)}
                                    className="text-red-400 hover:text-red-300"
                                >
                                    <Trash2 className="h-4 w-4 mr-2" />
                                    Delete
                                </DropdownMenuItem>
                            )}
                        </DropdownMenuContent>
                    </DropdownMenu>
                </div>
            </CardHeader>

            <CardContent>
                <div className="flex items-center justify-between mb-4">
                    <Badge className={getStatusColor(podcast.status)}>
                        {isGenerating && (
                            <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                        )}
                        {getStatusText(podcast.status)}
                    </Badge>
                    <div className="flex items-center text-sm text-gray-400">
                        <Clock className="mr-1 h-4 w-4" />
                        {podcast.length} min
                    </div>
                </div>

                {/* Generation Progress */}
                {isGenerating && (
                    <div className="mb-4 p-3 bg-purple-50 border border-purple-200 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium text-purple-700">
                                {generationPhase ? generationPhase.replace('_', ' ').toUpperCase() : 'Generating'}
                            </span>
                            <span className="text-sm text-purple-600">{generationProgress}%</span>
                        </div>
                        <Progress value={generationProgress} className="h-1.5 mb-2" />
                        <Button
                            size="sm"
                            variant="outline"
                            onClick={onViewProgress}
                            className="w-full text-xs border-purple-300 text-purple-600 hover:bg-purple-100"
                        >
                            View Progress
                        </Button>
                    </div>
                )}

                <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center text-xs text-gray-400">
                        <Calendar className="mr-1 h-3 w-3" />
                        {formatDistanceToNow(new Date(podcast.created_at), { addSuffix: true })}
                    </div>
                </div>

                {/* Audio Quality Indicators */}
                {canPlay && (
                    <div className="mb-4 p-2 bg-slate-700/30 rounded-lg border border-slate-600/50">
                        <div className="flex items-center justify-between text-xs">
                            <span className="text-gray-300 font-medium">Audio Quality</span>
                            <Badge variant="outline" className="text-green-400 border-green-500/30 bg-green-500/10">
                                MP3 • 128kbps
                            </Badge>
                        </div>
                        <div className="flex items-center justify-between mt-2 text-xs text-gray-400">
                            <div className="flex items-center">
                                <Radio className="mr-1 h-3 w-3" />
                                44.1kHz • Stereo
                            </div>
                            <div className="flex items-center">
                                <HardDrive className="mr-1 h-3 w-3" />
                                {podcast.length ? `~${Math.round(podcast.length * 0.96)}MB` : 'N/A'}
                            </div>
                        </div>
                    </div>
                )}

                <div className="flex items-center justify-between gap-2">
                    {/* Play/Generate Button */}
                    <div className="flex-1">
                        {canPlay && onPlay && (
                            <Button
                                onClick={() => onPlay(podcast)}
                                size="sm"
                                className="w-full bg-gradient-to-r from-purple-500 to-purple-600 hover:from-purple-600 hover:to-purple-700 text-white font-semibold"
                            >
                                <Play className="mr-2 h-4 w-4" />
                                Play
                            </Button>
                        )}

                        {canGenerate && onGenerate && (
                            <Button
                                onClick={() => onGenerate(podcast)}
                                size="sm"
                                variant="outline"
                                className="w-full border-purple-500 text-purple-500 hover:bg-purple-500 hover:text-slate-900"
                                disabled={isGenerating}
                            >
                                {isGenerating ? (
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                ) : (
                                    <Play className="mr-2 h-4 w-4" />
                                )}
                                {isGenerating ? 'Generating...' : 'Generate'}
                            </Button>
                        )}
                    </div>

                    {/* Download & Share Component */}
                    <PodcastDownloadShare
                        podcast={podcast}
                        onDownload={() => {
                            // Optional: refresh the podcast data or show success message
                            console.log('Download completed for:', podcast.title);
                        }}
                    />
                </div>
            </CardContent>
        </Card>
    );
} 