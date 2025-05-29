# VoiceFlow Studio

An AI-powered podcast generation platform that creates professional-quality podcasts from simple text prompts using a sophisticated 6-agent AI system.

## 🎯 Overview

VoiceFlow Studio transforms any topic into engaging, professional-quality podcasts featuring natural conversations between AI hosts with distinct personalities. Users can generate podcasts in minutes using our advanced AI pipeline powered by OpenAI and ElevenLabs.

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

## 🛠 Tech Stack

### Frontend
- **Framework**: Next.js 14+ with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Components**: shadcn/ui
- **Icons**: Lucide React
- **Authentication**: JWT tokens
- **Forms**: React Hook Form with validation
- **Payments**: Stripe Elements

### Backend
- **Framework**: FastAPI (Python)
- **Database**: SQLite (development) / PostgreSQL (production)
- **Authentication**: JWT with bcrypt
- **Payments**: Stripe API with webhooks
- **AI Services**: OpenAI API, ElevenLabs API (in progress)
- **File Storage**: AWS S3 (planned)
- **Migration**: Alembic

### Enhanced AI Pipeline (6-Agent System)
- **Research Agent**: Comprehensive topic research and fact-gathering
- **Script Agent**: Dialogue generation with distinct personalities
- **Content Planning Agent**: Episode structure and content organization
- **Conversation Flow Agent**: Natural dialogue flow and transitions
- **Dialogue Distribution Agent**: Balanced speaker allocation and timing
- **Personality Adaptation Agent**: Consistent character voice maintenance
- **Enhanced Pipeline Orchestrator**: Coordinated multi-agent execution

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

## ✅ Current Status

### Working Features
- ✅ **Landing Page**: Professional UI with hero section and navigation
- ✅ **User Authentication**: Complete registration and login system
- ✅ **Protected Dashboard**: User profile display with credit balance
- ✅ **User Profile Management**: Edit profile, change password, account settings
- ✅ **Credit System**: Complete credit management with transaction history
- ✅ **Stripe Integration**: Secure payment processing with webhooks
- ✅ **Credit Purchases**: Multiple credit bundles with secure checkout
- ✅ **Podcast Library**: CRUD operations with search and filtering
- ✅ **Database Integration**: SQLite with Alembic migrations
- ✅ **API Integration**: Full frontend-backend communication
- ✅ **Error Handling**: Comprehensive error messages and loading states
- ✅ **Enhanced 6-Agent AI Pipeline**: Complete multi-agent system for podcast generation
  - Research Agent for comprehensive topic research
  - Script Agent for dialogue generation
  - Content Planning Agent for episode structure
  - Conversation Flow Agent for natural dialogue flow
  - Dialogue Distribution Agent for balanced speaker allocation
  - Personality Adaptation Agent for consistent character voices
  - Enhanced Pipeline Orchestrator for coordinated execution
- ✅ **Advanced State Management**: Memory systems and conversation context
- ✅ **Sophisticated Dialogue Generation**: Multi-personality conversation system

### In Development
- 🔄 **ElevenLabs TTS Integration**: Text-to-speech voice generation
- 🔄 **Voice Profile Management**: Distinct host voice profiles

### Not Yet Implemented
- ❌ **Audio File Generation**: ElevenLabs TTS integration
- ❌ **Audio Assembly**: Intro/outro music and effects
- ❌ **Podcast Generation UI**: Frontend interface for AI pipeline
- ❌ **File Storage**: AWS S3 integration
- ❌ **Real-time Progress**: Generation status tracking
- ❌ **Download System**: Audio file delivery

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
ELEVENLABS_API_KEY=your-elevenlabs-api-key-here

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
**Current Task**: ElevenLabs API Integration for TTS
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

---

**VoiceFlow Studio** - Transform ideas into professional podcasts with AI ✨
