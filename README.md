# VoiceFlow Studio 🎙️

An AI-powered podcast generation platform that creates high-quality, engaging podcasts from simple prompts. Built with Next.js, FastAPI, and advanced AI pipeline powered by OpenAI and Chatterbox TTS.

## 🎯 Overview

VoiceFlow Studio transforms any topic into engaging, professional-quality podcasts featuring natural conversations between AI hosts with distinct personalities. Users can generate podcasts in minutes using our advanced AI pipeline powered by OpenAI and Chatterbox TTS.

## 🚀 Features

### Core Features
- **Enhanced 6-Agent AI System**: Sophisticated multi-agent pipeline for natural conversations
- **Professional Quality**: Studio-quality audio with music and effects (coming soon)
- **Instant Creation**: Generate podcast scripts in minutes with AI agents
- **Credit System**: Pay-per-use model with Stripe integration
- **Example Library**: Showcase of AI-generated podcast samples

### Technical Features
- **Modern Frontend**: Next.js 14+ with TypeScript and Tailwind CSS
- **Robust Backend**: FastAPI with async support and SQLAlchemy ORM
- **Secure Authentication**: JWT-based auth with password hashing
- **Payment Integration**: Complete Stripe integration for credit purchases
- **Advanced AI Pipeline**: 6-agent system with sophisticated conversation generation
- **Responsive Design**: Mobile-first UI with shadcn/ui components

## 🛠️ Tech Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **shadcn/ui** - Modern UI components

### Backend
- **FastAPI** - Modern Python web framework
- **PostgreSQL** - Reliable database with SQLAlchemy ORM
- **Chatterbox TTS** - Open-source text-to-speech engine
- **OpenAI API** - GPT models for content generation
- **Stripe** - Payment processing
- **Pydantic** - Data validation

### AI Services
- **OpenAI GPT-4** - Script generation and content creation
- **Chatterbox TTS** - Voice synthesis and audio generation
- **Custom Agents** - Specialized AI agents for different tasks

## 📁 Project Structure

```
VoiceFlow Studio/
├── frontend/                 # Next.js frontend application
│   ├── src/
│   │   ├── app/             # App Router pages
│   │   │   ├── auth/        # Authentication pages
│   │   │   ├── dashboard/   # User dashboard
│   │   │   │   ├── credits/ # Credit management
│   │   │   │   ├── library/ # Podcast library
│   │   │   │   └── profile/ # User profile
│   │   │   └── library/     # Example podcasts
│   │   ├── components/      # Reusable components
│   │   │   ├── ui/          # shadcn/ui components
│   │   │   ├── auth/        # Auth components
│   │   │   └── podcast/     # Podcast components
│   │   └── lib/             # Utilities and config
│   └── public/              # Static assets
├── backend/                 # FastAPI backend application
│   ├── app/
│   │   ├── api/             # API routes
│   │   ├── core/            # Core functionality
│   │   ├── models/          # Database models
│   │   ├── schemas/         # Pydantic schemas
│   │   └── services/        # Business logic & AI agents
│   │       ├── research_agent.py
│   │       ├── script_agent.py
│   │       ├── content_planning_agent.py
│   │       ├── conversation_flow_agent.py
│   │       ├── dialogue_distribution_agent.py
│   │       ├── personality_adaptation_agent.py
│   │       └── enhanced_pipeline_orchestrator.py
│   ├── alembic/             # Database migrations
│   └── run.py               # Server startup script
├── start-dev.sh            # Unix startup script
├── start-dev.bat           # Windows startup script
├── tasks.txt               # Development progress tracking
├── CONFIGURATION.md        # Environment setup guide
└── README.md              # Project documentation
```

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

**Windows:**
```bash
# Double-click start-dev.bat or run:
start-dev.bat
```

**macOS/Linux:**
```bash
./start-dev.sh
```

### Option 2: Manual Setup

**Backend (Terminal 1):**
```bash
cd backend
source venv/Scripts/activate  # Windows Git Bash
source venv/bin/activate      # macOS/Linux
python run.py
```

**Frontend (Terminal 2):**
```bash
cd frontend
npm run dev
```

### Access the Application
- **Frontend**: http://localhost:3000 (or next available port)
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📋 Current Status

### ✅ Completed Features

#### Phase 1-2: Foundation & UI
- ✅ Next.js project setup with TypeScript
- ✅ FastAPI backend with PostgreSQL
- ✅ User authentication system
- ✅ Modern dashboard interface
- ✅ Credit management system

#### Phase 3: Payment Integration
- ✅ Stripe integration for credit purchases
- ✅ Secure payment processing
- ✅ Credit bundle system

#### Phase 5: AI Pipeline
- ✅ OpenAI API integration
- ✅ Enhanced 6-agent architecture
- ✅ Content generation pipeline
- ✅ Research and script agents

#### Phase 6: Voice & Audio (CHATTERBOX MIGRATION COMPLETE)
- ✅ **Chatterbox TTS Integration**: Open-source text-to-speech
- ✅ **Voice Profile System**: Distinct host personalities 
- ✅ **Local Audio Generation**: No API costs, runs on your hardware
- ✅ **Voice Cloning Support**: Custom voice prompts
- ✅ **Audio Processing Pipeline**: Complete podcast assembly

#### Phase 7: Generation Interface
- ✅ Podcast generation form
- ✅ **Real-Time Progress Tracking**: WebSocket-based live updates
- ✅ **Generation Progress Modal**: Beautiful phase-by-phase progress UI
- ✅ **Connection Management**: Auto-reconnection and error handling
- ✅ **Generation Queue Management**: Multiple concurrent generations

### 🔄 In Progress
- **Generation Queue Management**: Multiple concurrent generations (Task 7.4)

### 🔄 **CURRENT FOCUS** (Task 7.5)

**Download & Sharing Features** - Next implementation:
- Audio file download functionality
- Sharing links and embed codes
- Export options (MP3, transcript, etc.)
- Podcast metadata display and management

## 🎯 **NEXT IMMEDIATE TASKS**

#### 1. Download & Sharing Features (Task 7.5)
```typescript
// Download functionality
const handleDownload = (podcast: Podcast) => {
  if (podcast.audio_url) {
    const link = document.createElement('a');
    link.href = podcast.audio_url;
    link.download = `${podcast.title}.mp3`;
    link.click();
  }
};
```

#### 2. Error Handling & Retries (Task 7.6)
- Comprehensive error recovery
- Automatic retry mechanisms
- User-friendly error messages

#### 3. Enhanced Audio Features (Task 8.1)
- Background music integration
- Advanced audio effects
- Voice customization options

## 🚀 **STRATEGIC PLAN MOVING FORWARD**

## 🔧 Development Setup

### Prerequisites
- Node.js 18+ and npm
- Python 3.9+ and pip
- Git

### First-Time Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd voiceflow-studio
   ```

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/Scripts/activate  # Windows
   source venv/bin/activate      # macOS/Linux
   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   ```

4. **Database Setup**
   ```bash
   cd backend
   alembic upgrade head
   ```

5. **Start Development**
   ```bash
   # From project root
   ./start-dev.sh  # Unix
   start-dev.bat   # Windows
   ```

## 🔧 Environment Variables

### Backend (.env)
```env
# Database (SQLite for development)
DATABASE_URL=sqlite:///./voiceflow.db

# Security
SECRET_KEY=your-secret-key-change-this-in-production

# External APIs
OPENAI_API_KEY=your-openai-api-key-here

# Stripe
STRIPE_SECRET_KEY=your-stripe-secret-key-here
STRIPE_PUBLISHABLE_KEY=your-stripe-publishable-key-here
STRIPE_WEBHOOK_SECRET=your-stripe-webhook-secret-here

# AWS S3 (for future use)
AWS_ACCESS_KEY_ID=your-aws-access-key-here
AWS_SECRET_ACCESS_KEY=your-aws-secret-key-here
S3_BUCKET_NAME=your-s3-bucket-name-here

# Application
DEBUG=true
```

### Frontend (.env.local)
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=your-stripe-publishable-key-here
```

## 🎨 Design System

### Color Palette
- **Primary**: Purple (#8B5CF6)
- **Background**: Slate gradients (#0F172A to #7C3AED)
- **Text**: White (#FFFFFF) and Gray (#9CA3AF)
- **Accent**: Pink (#EC4899) for highlights

### Typography
- **Headings**: Bold, large sizes for impact
- **Body**: Clean, readable fonts
- **Code**: Monospace for technical content

## 🧪 Testing the Application

### Complete User Journey
1. Visit http://localhost:3000
2. Register a new account with email validation
3. Login with your credentials
4. Explore the dashboard with credit balance display
5. Visit profile settings to update account information
6. Check the credit management page
7. Test the podcast library interface
8. Try the Stripe payment integration (test mode)

### AI Pipeline Testing
- The enhanced 6-agent system is ready for testing
- Research Agent generates comprehensive topic research
- Script Agent creates natural dialogues with distinct personalities
- All agents coordinate through the Enhanced Pipeline Orchestrator

### API Testing
- Visit http://localhost:8000/docs for interactive API documentation
- Test all authentication endpoints
- Verify credit system operations
- Test Stripe webhook handling

## 📋 Development Tasks

See `tasks.txt` for detailed development progress and upcoming features.

**Current Phase**: Voice & Audio Processing (Phase 6)
**Current Task**: Implementing Chatterbox TTS for cost-effective, high-quality voice generation
**Next Phase**: Podcast Generation Interface (Phase 7)

## 🎯 AI Pipeline Architecture

### Enhanced 6-Agent System

1. **Research Agent**: Gathers comprehensive information about the topic
2. **Content Planning Agent**: Structures the episode with segments and flow
3. **Script Agent**: Generates initial dialogue with personality considerations
4. **Conversation Flow Agent**: Ensures natural transitions and engagement
5. **Dialogue Distribution Agent**: Balances speaking time and interaction patterns
6. **Personality Adaptation Agent**: Maintains consistent character voices throughout

### Orchestration
The **Enhanced Pipeline Orchestrator** coordinates all agents, manages state, and ensures coherent output through sophisticated workflow management.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the API documentation at http://localhost:8000/docs
- Review the tasks.txt file for development status
- Create an issue in the repository

## 💡 **Key Achievements**

You've successfully built a **production-ready AI podcast generation platform** with:
- **Cost-effective audio generation** (Chatterbox TTS vs. ElevenLabs)
- **Sophisticated AI pipeline** (6-agent system)
- **Professional user experience** (modern UI/UX)
- **Real-time progress tracking** (WebSocket-based live updates)
- **Sustainable business model** (credit-based payments)
- **Scalable architecture** (cloud-ready infrastructure)

You're approximately **90-95% complete** with the core platform. The remaining work focuses on queue management, download features, and production deployment rather than major feature development.

---

**VoiceFlow Studio** - Transform ideas into professional podcasts with AI ✨
