import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Play, Mic, ArrowLeft, Clock, Tag } from 'lucide-react'

const examplePodcasts = [
    {
        id: 1,
        title: "The Future of Artificial Intelligence",
        description: "Exploring how AI will reshape industries, from healthcare to transportation, and what it means for the future of work.",
        topic: "Technology",
        duration: "12 min",
        featured: true,
        audioUrl: "/audio/ai-future.mp3"
    },
    {
        id: 2,
        title: "Healthy Living in the Digital Age",
        description: "Practical tips for maintaining physical and mental health while navigating our increasingly connected world.",
        topic: "Health & Wellness",
        duration: "15 min",
        featured: false,
        audioUrl: "/audio/healthy-living.mp3"
    },
    {
        id: 3,
        title: "Startup Success Stories",
        description: "Lessons learned from entrepreneurs who built successful companies from the ground up.",
        topic: "Business",
        duration: "18 min",
        featured: true,
        audioUrl: "/audio/startup-stories.mp3"
    },
    {
        id: 4,
        title: "Climate Change Solutions",
        description: "Innovative approaches to combating climate change and building a sustainable future.",
        topic: "Environment",
        duration: "14 min",
        featured: false,
        audioUrl: "/audio/climate-solutions.mp3"
    },
    {
        id: 5,
        title: "The Psychology of Productivity",
        description: "Understanding the mental frameworks that drive high performance and personal effectiveness.",
        topic: "Psychology",
        duration: "16 min",
        featured: false,
        audioUrl: "/audio/productivity-psychology.mp3"
    },
    {
        id: 6,
        title: "Space Exploration Frontiers",
        description: "The latest developments in space technology and humanity's journey to the stars.",
        topic: "Science",
        duration: "20 min",
        featured: true,
        audioUrl: "/audio/space-exploration.mp3"
    }
]

const topics = ["All", "Technology", "Health & Wellness", "Business", "Environment", "Psychology", "Science"]

export default function LibraryPage() {
    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
            {/* Navigation */}
            <nav className="container mx-auto px-4 py-6">
                <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                        <Link href="/" className="flex items-center space-x-2">
                            <ArrowLeft className="h-5 w-5 text-purple-400" />
                            <span className="text-purple-400 hover:text-purple-300">Back to Home</span>
                        </Link>
                        <div className="flex items-center space-x-2">
                            <Mic className="h-8 w-8 text-purple-400" />
                            <span className="text-2xl font-bold text-white">VoiceFlow Studio</span>
                        </div>
                    </div>
                    <div className="flex items-center space-x-4">
                        <Link href="/auth/login">
                            <Button variant="ghost" className="text-white hover:text-purple-300">
                                Sign In
                            </Button>
                        </Link>
                        <Link href="/auth/register">
                            <Button className="bg-purple-600 hover:bg-purple-700">
                                Get Started
                            </Button>
                        </Link>
                    </div>
                </div>
            </nav>

            {/* Header */}
            <section className="container mx-auto px-4 py-12">
                <div className="text-center mb-12">
                    <h1 className="text-4xl md:text-6xl font-bold text-white mb-4">
                        Example Podcast Library
                    </h1>
                    <p className="text-xl text-gray-300 max-w-2xl mx-auto">
                        Experience the quality and variety of AI-generated podcasts. Each episode features
                        natural conversations between AI hosts with distinct personalities.
                    </p>
                </div>

                {/* Topic Filter */}
                <div className="flex flex-wrap justify-center gap-2 mb-12">
                    {topics.map((topic) => (
                        <Button
                            key={topic}
                            variant={topic === "All" ? "default" : "outline"}
                            className={
                                topic === "All"
                                    ? "bg-purple-600 hover:bg-purple-700"
                                    : "border-purple-400 text-purple-400 hover:bg-purple-400 hover:text-white"
                            }
                        >
                            {topic}
                        </Button>
                    ))}
                </div>
            </section>

            {/* Featured Podcasts */}
            <section className="container mx-auto px-4 pb-12">
                <h2 className="text-2xl font-bold text-white mb-6">Featured Episodes</h2>
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
                    {examplePodcasts
                        .filter(podcast => podcast.featured)
                        .map((podcast) => (
                            <Card key={podcast.id} className="bg-slate-800/50 border-slate-700 hover:bg-slate-800/70 transition-all duration-300 group">
                                <CardHeader>
                                    <div className="flex items-start justify-between">
                                        <div className="flex-1">
                                            <div className="flex items-center space-x-2 mb-2">
                                                <Tag className="h-4 w-4 text-purple-400" />
                                                <span className="text-sm text-purple-400">{podcast.topic}</span>
                                            </div>
                                            <CardTitle className="text-white text-lg mb-2 group-hover:text-purple-300 transition-colors">
                                                {podcast.title}
                                            </CardTitle>
                                            <CardDescription className="text-gray-400 text-sm">
                                                {podcast.description}
                                            </CardDescription>
                                        </div>
                                    </div>
                                </CardHeader>
                                <CardContent>
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center space-x-2 text-gray-400">
                                            <Clock className="h-4 w-4" />
                                            <span className="text-sm">{podcast.duration}</span>
                                        </div>
                                        <Button
                                            size="sm"
                                            className="bg-purple-600 hover:bg-purple-700 group-hover:scale-105 transition-transform"
                                        >
                                            <Play className="h-4 w-4 mr-2" />
                                            Play
                                        </Button>
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                </div>
            </section>

            {/* All Podcasts */}
            <section className="container mx-auto px-4 pb-20">
                <h2 className="text-2xl font-bold text-white mb-6">All Episodes</h2>
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {examplePodcasts.map((podcast) => (
                        <Card key={podcast.id} className="bg-slate-800/50 border-slate-700 hover:bg-slate-800/70 transition-all duration-300 group">
                            <CardHeader>
                                <div className="flex items-start justify-between">
                                    <div className="flex-1">
                                        <div className="flex items-center space-x-2 mb-2">
                                            <Tag className="h-4 w-4 text-purple-400" />
                                            <span className="text-sm text-purple-400">{podcast.topic}</span>
                                            {podcast.featured && (
                                                <span className="bg-yellow-500 text-black text-xs px-2 py-1 rounded-full font-medium">
                                                    Featured
                                                </span>
                                            )}
                                        </div>
                                        <CardTitle className="text-white text-lg mb-2 group-hover:text-purple-300 transition-colors">
                                            {podcast.title}
                                        </CardTitle>
                                        <CardDescription className="text-gray-400 text-sm">
                                            {podcast.description}
                                        </CardDescription>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center space-x-2 text-gray-400">
                                        <Clock className="h-4 w-4" />
                                        <span className="text-sm">{podcast.duration}</span>
                                    </div>
                                    <Button
                                        size="sm"
                                        className="bg-purple-600 hover:bg-purple-700 group-hover:scale-105 transition-transform"
                                    >
                                        <Play className="h-4 w-4 mr-2" />
                                        Play
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>
                    ))}
                </div>
            </section>

            {/* CTA Section */}
            <section className="container mx-auto px-4 py-20 text-center">
                <div className="max-w-2xl mx-auto">
                    <h2 className="text-3xl font-bold text-white mb-4">
                        Ready to Create Your Own?
                    </h2>
                    <p className="text-xl text-gray-300 mb-8">
                        Join thousands of creators using AI to produce professional podcasts in minutes.
                    </p>
                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                        <Link href="/auth/register">
                            <Button size="lg" className="bg-purple-600 hover:bg-purple-700 text-lg px-8 py-4">
                                Start Creating Free
                            </Button>
                        </Link>
                        <Link href="/pricing">
                            <Button size="lg" variant="outline" className="text-lg px-8 py-4 border-purple-400 text-purple-400 hover:bg-purple-400 hover:text-white">
                                View Pricing
                            </Button>
                        </Link>
                    </div>
                </div>
            </section>

            {/* Footer */}
            <footer className="container mx-auto px-4 py-12 border-t border-slate-700">
                <div className="text-center text-gray-400">
                    <div className="flex items-center justify-center space-x-2 mb-4">
                        <Mic className="h-6 w-6 text-purple-400" />
                        <span className="text-xl font-bold text-white">VoiceFlow Studio</span>
                    </div>
                    <p>&copy; 2024 VoiceFlow Studio. All rights reserved.</p>
                </div>
            </footer>
        </div>
    )
} 