import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Play, Mic, Zap, Users, Star, ArrowRight } from 'lucide-react'

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Navigation */}
      <nav className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Mic className="h-8 w-8 text-purple-400" />
            <span className="text-2xl font-bold text-white">VoiceFlow Studio</span>
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

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-20 text-center">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6">
            Create Professional
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
              {" "}AI Podcasts
            </span>
          </h1>
          <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
            Transform any topic into engaging, professional-quality podcasts with our AI-powered
            multi-agent system. Just describe your topic and let our AI hosts create the conversation.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/auth/register">
              <Button size="lg" className="bg-purple-600 hover:bg-purple-700 text-lg px-8 py-4">
                Start Creating <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </Link>
            <Link href="/library">
              <Button size="lg" variant="outline" className="text-lg px-8 py-4 border-purple-400 text-purple-400 hover:bg-purple-400 hover:text-white">
                <Play className="mr-2 h-5 w-5" />
                Listen to Examples
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="container mx-auto px-4 py-20">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-white mb-4">
            Powered by Advanced AI Technology
          </h2>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Our multi-agent system creates natural, engaging conversations between AI hosts
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <Zap className="h-12 w-12 text-purple-400 mb-4" />
              <CardTitle className="text-white">Lightning Fast</CardTitle>
              <CardDescription className="text-gray-300">
                Generate professional podcasts in minutes, not hours
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <Users className="h-12 w-12 text-purple-400 mb-4" />
              <CardTitle className="text-white">Natural Conversations</CardTitle>
              <CardDescription className="text-gray-300">
                AI hosts with distinct personalities create engaging dialogue
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <Star className="h-12 w-12 text-purple-400 mb-4" />
              <CardTitle className="text-white">Professional Quality</CardTitle>
              <CardDescription className="text-gray-300">
                Studio-quality audio with music and sound effects
              </CardDescription>
            </CardHeader>
          </Card>
        </div>
      </section>

      {/* Example Podcasts Section */}
      <section className="container mx-auto px-4 py-20">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-white mb-4">
            Listen to AI-Generated Podcasts
          </h2>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Experience the quality and variety of our AI-generated content
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[
            { title: "The Future of AI", topic: "Technology", duration: "12 min" },
            { title: "Healthy Living Tips", topic: "Health & Wellness", duration: "15 min" },
            { title: "Startup Success Stories", topic: "Business", duration: "18 min" },
          ].map((podcast, index) => (
            <Card key={index} className="bg-slate-800/50 border-slate-700 hover:bg-slate-800/70 transition-colors">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Play className="h-8 w-8 text-purple-400" />
                    <div>
                      <CardTitle className="text-white text-lg">{podcast.title}</CardTitle>
                      <CardDescription className="text-gray-400">{podcast.topic}</CardDescription>
                    </div>
                  </div>
                  <span className="text-sm text-gray-400">{podcast.duration}</span>
                </div>
              </CardHeader>
            </Card>
          ))}
        </div>

        <div className="text-center mt-12">
          <Link href="/library">
            <Button size="lg" variant="outline" className="border-purple-400 text-purple-400 hover:bg-purple-400 hover:text-white">
              View Full Library
            </Button>
          </Link>
        </div>
      </section>

      {/* Pricing Section */}
      <section className="container mx-auto px-4 py-20">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-white mb-4">
            Simple, Transparent Pricing
          </h2>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Pay only for what you create. No subscriptions, no hidden fees.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-4xl mx-auto">
          {[
            { credits: 1, price: 5, popular: false },
            { credits: 5, price: 20, popular: true },
            { credits: 10, price: 35, popular: false },
          ].map((plan, index) => (
            <Card key={index} className={`bg-slate-800/50 border-slate-700 ${plan.popular ? 'ring-2 ring-purple-400' : ''}`}>
              <CardHeader className="text-center">
                {plan.popular && (
                  <div className="bg-purple-600 text-white text-sm font-medium px-3 py-1 rounded-full mb-4 inline-block">
                    Most Popular
                  </div>
                )}
                <CardTitle className="text-white text-2xl">{plan.credits} Credit{plan.credits > 1 ? 's' : ''}</CardTitle>
                <div className="text-4xl font-bold text-white">${plan.price}</div>
                <CardDescription className="text-gray-300">
                  ${plan.price / plan.credits} per podcast
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button className="w-full bg-purple-600 hover:bg-purple-700">
                  Get Started
                </Button>
              </CardContent>
            </Card>
          ))}
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
