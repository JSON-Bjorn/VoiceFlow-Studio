'use client'

import React, { useState } from 'react'
import { Button } from '@/components/ui/button'
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger
} from '@/components/ui/dialog'
import {
    Download,
    Share2,
    Copy,
    ExternalLink,
    Twitter,
    Facebook,
    Linkedin,
    MessageSquare,
    CheckCircle,
    AlertCircle
} from 'lucide-react'

interface Podcast {
    id: number
    title: string
    topic: string
    status: string
    has_audio?: boolean
    audio_url?: string
}

interface ShareInfo {
    podcast_id: number
    title: string
    topic: string
    duration: string
    share_url: string
    downloadable: boolean
    social_shares: {
        twitter: string
        facebook: string
        linkedin: string
        reddit: string
    }
}

interface PodcastDownloadShareProps {
    podcast: Podcast
    onDownload?: () => void
}

// Simple toast utility
const showToast = (message: string, type: 'success' | 'error' = 'success') => {
    // For now, use alert - in production, you'd want a proper toast library
    if (type === 'error') {
        alert(`Error: ${message}`);
    } else {
        alert(message);
    }
};

export function PodcastDownloadShare({ podcast, onDownload }: PodcastDownloadShareProps) {
    const [isShareDialogOpen, setIsShareDialogOpen] = useState(false)
    const [shareInfo, setShareInfo] = useState<ShareInfo | null>(null)
    const [isDownloading, setIsDownloading] = useState(false)
    const [isLoadingShareInfo, setIsLoadingShareInfo] = useState(false)

    const canDownload = podcast.status === 'completed' && (podcast.has_audio || podcast.audio_url || true) // Temporary: allow all completed podcasts to test download

    const handleDownload = async () => {
        if (!canDownload) {
            showToast('Podcast is not ready for download', 'error');
            return
        }

        setIsDownloading(true)

        try {
            const token = localStorage.getItem('access_token')
            if (!token) {
                showToast('Please login to download podcasts', 'error');
                return
            }

            const response = await fetch(`http://localhost:8000/api/podcasts/${podcast.id}/download`, {
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
            })

            if (!response.ok) {
                const errorData = await response.json()
                throw new Error(errorData.detail || 'Download failed')
            }

            // Get filename from response headers or create one
            const contentDisposition = response.headers.get('content-disposition')
            let filename = `${podcast.title}.mp3`

            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename="(.+)"/)
                if (filenameMatch) {
                    filename = filenameMatch[1]
                }
            }

            // Download the file
            const blob = await response.blob()
            const url = window.URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = filename
            document.body.appendChild(a)
            a.click()
            document.body.removeChild(a)
            window.URL.revokeObjectURL(url)

            showToast('Podcast downloaded successfully!');
            onDownload?.()

        } catch (error) {
            console.error('Download failed:', error)
            showToast(error instanceof Error ? error.message : 'Failed to download podcast', 'error');
        } finally {
            setIsDownloading(false)
        }
    }

    const handleShare = async () => {
        setIsLoadingShareInfo(true)

        try {
            const token = localStorage.getItem('access_token')
            if (!token) {
                showToast('Please login to share podcasts', 'error');
                return
            }

            const response = await fetch(`http://localhost:8000/api/podcasts/${podcast.id}/share-info`, {
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
            })

            if (!response.ok) {
                throw new Error('Failed to get share information')
            }

            const data = await response.json()
            setShareInfo(data)
            setIsShareDialogOpen(true)

        } catch (error) {
            console.error('Share failed:', error)
            showToast('Failed to get share information', 'error');
        } finally {
            setIsLoadingShareInfo(false)
        }
    }

    const copyToClipboard = async (text: string) => {
        try {
            await navigator.clipboard.writeText(text)
            showToast('Link copied to clipboard!');
        } catch (error) {
            console.error('Failed to copy:', error)
            showToast('Failed to copy link', 'error');
        }
    }

    const openInNewTab = (url: string) => {
        window.open(url, '_blank', 'noopener,noreferrer')
    }

    return (
        <div className="flex items-center space-x-2">
            {/* Download Button */}
            <Button
                size="sm"
                onClick={handleDownload}
                disabled={!canDownload || isDownloading}
                className={`flex items-center space-x-1 transition-all duration-200 ${canDownload
                    ? 'bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white shadow-lg hover:shadow-green-500/25 border-0'
                    : 'border-slate-600 text-slate-400 hover:bg-slate-700 hover:text-slate-300 bg-slate-800/50'
                    } ${(!canDownload || isDownloading) ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
                {isDownloading ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                ) : (
                    <Download className="h-4 w-4" />
                )}
                <span>{isDownloading ? 'Downloading...' : 'Download'}</span>
            </Button>

            {/* Share Button */}
            <Dialog open={isShareDialogOpen} onOpenChange={setIsShareDialogOpen}>
                <DialogTrigger asChild>
                    <Button
                        size="sm"
                        variant="outline"
                        onClick={handleShare}
                        disabled={isLoadingShareInfo}
                        className="flex items-center space-x-1 border-purple-500/50 text-purple-300 hover:bg-purple-500/20 hover:text-purple-200 hover:border-purple-400 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isLoadingShareInfo ? (
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current"></div>
                        ) : (
                            <Share2 className="h-4 w-4" />
                        )}
                        <span>Share</span>
                    </Button>
                </DialogTrigger>

                <DialogContent className="sm:max-w-md p-6">
                    <DialogHeader>
                        <DialogTitle className="flex items-center space-x-2">
                            <Share2 className="h-5 w-5" />
                            <span>Share Podcast</span>
                        </DialogTitle>
                        <DialogDescription>
                            Share this AI-generated podcast with others
                        </DialogDescription>
                    </DialogHeader>

                    {shareInfo && (
                        <div className="space-y-6 mt-4">
                            {/* Podcast Info */}
                            <div className="space-y-2">
                                <h3 className="font-semibold text-lg text-gray-900">{shareInfo.title}</h3>
                                <p className="text-sm text-gray-600">{shareInfo.topic}</p>
                                <p className="text-sm text-gray-500">Duration: {shareInfo.duration}</p>

                                {shareInfo.downloadable && (
                                    <div className="flex items-center space-x-1 text-green-600">
                                        <CheckCircle className="h-4 w-4" />
                                        <span className="text-sm">Ready for download</span>
                                    </div>
                                )}
                            </div>

                            {/* Share URL */}
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-gray-900">Share Link</label>
                                <div className="flex items-center space-x-2">
                                    <input
                                        type="text"
                                        value={shareInfo.share_url}
                                        readOnly
                                        className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm bg-gray-50"
                                    />
                                    <Button
                                        size="sm"
                                        variant="outline"
                                        onClick={() => copyToClipboard(shareInfo.share_url)}
                                    >
                                        <Copy className="h-4 w-4" />
                                    </Button>
                                </div>
                            </div>

                            {/* Social Media Sharing */}
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-gray-900">Share on Social Media</label>
                                <div className="grid grid-cols-2 gap-2">
                                    <Button
                                        size="sm"
                                        variant="twitter"
                                        onClick={() => openInNewTab(shareInfo.social_shares.twitter)}
                                        className="flex items-center space-x-2"
                                    >
                                        <Twitter className="h-4 w-4" />
                                        <span>Twitter</span>
                                        <ExternalLink className="h-3 w-3" />
                                    </Button>

                                    <Button
                                        size="sm"
                                        variant="facebook"
                                        onClick={() => openInNewTab(shareInfo.social_shares.facebook)}
                                        className="flex items-center space-x-2"
                                    >
                                        <Facebook className="h-4 w-4" />
                                        <span>Facebook</span>
                                        <ExternalLink className="h-3 w-3" />
                                    </Button>

                                    <Button
                                        size="sm"
                                        variant="linkedin"
                                        onClick={() => openInNewTab(shareInfo.social_shares.linkedin)}
                                        className="flex items-center space-x-2"
                                    >
                                        <Linkedin className="h-4 w-4" />
                                        <span>LinkedIn</span>
                                        <ExternalLink className="h-3 w-3" />
                                    </Button>

                                    <Button
                                        size="sm"
                                        variant="reddit"
                                        onClick={() => openInNewTab(shareInfo.social_shares.reddit)}
                                        className="flex items-center space-x-2"
                                    >
                                        <MessageSquare className="h-4 w-4" />
                                        <span>Reddit</span>
                                        <ExternalLink className="h-3 w-3" />
                                    </Button>
                                </div>
                            </div>

                            {/* Disclaimer */}
                            <div className="flex items-start space-x-2 p-3 bg-blue-50 rounded-md">
                                <AlertCircle className="h-4 w-4 text-blue-600 mt-0.5" />
                                <div className="text-sm text-blue-800">
                                    <p className="font-medium">Note about sharing:</p>
                                    <p>Shared podcasts will be accessible to anyone with the link. Only completed podcasts with audio can be downloaded by others.</p>
                                </div>
                            </div>
                        </div>
                    )}
                </DialogContent>
            </Dialog>
        </div>
    )
} 