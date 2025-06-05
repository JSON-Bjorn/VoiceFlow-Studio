'use client';

import { Podcast } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { PodcastDownloadShare } from './PodcastDownloadShare';
import {
    Play,
    Download,
    Clock,
    Calendar,
    MoreVertical,
    Trash2,
    Eye,
    Loader2
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
}

const getStatusColor = (status: Podcast['status']) => {
    switch (status) {
        case 'completed':
            return 'bg-green-100 text-green-800 border-green-200';
        case 'generating':
            return 'bg-blue-100 text-blue-800 border-blue-200';
        case 'pending':
            return 'bg-yellow-100 text-yellow-800 border-yellow-200';
        case 'failed':
            return 'bg-red-100 text-red-800 border-red-200';
        default:
            return 'bg-gray-100 text-gray-800 border-gray-200';
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
    isGenerating = false
}: PodcastCardProps) {
    const canPlay = podcast.status === 'completed' && podcast.audio_url;
    const canGenerate = podcast.status === 'pending' || podcast.status === 'failed';

    return (
        <Card className="hover:shadow-lg transition-shadow duration-200">
            <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                        <CardTitle className="text-lg font-semibold text-gray-900 truncate">
                            {podcast.title}
                        </CardTitle>
                        <p className="text-sm text-gray-600 mt-1 line-clamp-2">
                            {podcast.topic}
                        </p>
                    </div>

                    <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                                <MoreVertical className="h-4 w-4" />
                            </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                            {onView && (
                                <DropdownMenuItem onClick={() => onView(podcast)}>
                                    <Eye className="mr-2 h-4 w-4" />
                                    View Details
                                </DropdownMenuItem>
                            )}
                            {onDelete && (
                                <DropdownMenuItem
                                    onClick={() => onDelete(podcast)}
                                    className="text-red-600"
                                >
                                    <Trash2 className="mr-2 h-4 w-4" />
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
                        {isGenerating && podcast.status === 'generating' && (
                            <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                        )}
                        {getStatusText(podcast.status)}
                    </Badge>
                    <div className="flex items-center text-sm text-gray-500">
                        <Clock className="mr-1 h-4 w-4" />
                        {podcast.length} min
                    </div>
                </div>

                <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center text-xs text-gray-500">
                        <Calendar className="mr-1 h-3 w-3" />
                        {formatDistanceToNow(new Date(podcast.created_at), { addSuffix: true })}
                    </div>
                </div>

                <div className="flex items-center justify-between gap-2">
                    {/* Play/Generate Button */}
                    <div className="flex-1">
                        {canPlay && onPlay && (
                            <Button
                                onClick={() => onPlay(podcast)}
                                size="sm"
                                className="w-full"
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
                                className="w-full"
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