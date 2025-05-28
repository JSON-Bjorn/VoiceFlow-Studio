# VoiceFlow Studio

An AI-powered podcast generation platform that creates professional-quality podcasts from simple text prompts using a multi-agent system.

## 🎯 Overview

VoiceFlow Studio transforms any topic into engaging, professional-quality podcasts featuring natural conversations between AI hosts with distinct personalities. Users can generate podcasts in minutes using our advanced AI pipeline powered by OpenAI and ElevenLabs.

## 🚀 Features

### Core Features
- **AI-Powered Generation**: Multi-agent system creates natural conversations
- **Professional Quality**: Studio-quality audio with music and effects
- **Instant Creation**: Generate podcasts in minutes, not hours
- **Credit System**: Pay-per-use model with transparent pricing
- **Example Library**: Showcase of AI-generated podcast samples

### Technical Features
- **Modern Frontend**: Next.js 14+ with TypeScript and Tailwind CSS
- **Robust Backend**: FastAPI with async support and SQLAlchemy ORM
- **Secure Authentication**: JWT-based auth with password hashing
- **Payment Integration**: Stripe for secure credit purchases
- **Cloud Storage**: AWS S3 for audio file storage
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

### Backend
- **Framework**: FastAPI (Python)
- **Database**: SQLite (development) / PostgreSQL (production)
- **Authentication**: JWT with bcrypt
- **Payments**: Stripe API
- **AI Services**: OpenAI API, ElevenLabs API
- **File Storage**: AWS S3
- **Migration**: Alembic

### AI Pipeline
- **Research Agent**: OpenAI GPT for topic research
- **Script Agent**: Dialogue generation with distinct personalities
- **Voice Agent**: ElevenLabs TTS with multiple voices
- **Audio Agent**: Audio assembly with intro/outro music
- **Orchestrator**: Manages the entire generation workflow

## 📁 Project Structure

```
VoiceFlow Studio/
├── frontend/                 # Next.js frontend application
│   ├── src/
│   │   ├── app/             # App Router pages
│   │   │   ├── auth/        # Authentication pages
│   │   │   ├── dashboard/   # User dashboard
│   │   │   ├── generate/    # Podcast generation
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
│   │   ├── services/        # Business logic
│   │   └── agents/          # AI agent system
│   ├── alembic/             # Database migrations
│   └── run.py               # Server startup script
├── start-dev.sh            # Unix startup script
├── start-dev.bat           # Windows startup script
├── tasks.txt               # Development progress tracking
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
- ✅ **User Registration**: Email validation, password confirmation, terms acceptance
- ✅ **User Login**: JWT authentication with secure token storage
- ✅ **Protected Dashboard**: User profile display with credit balance
- ✅ **Database Integration**: SQLite with Alembic migrations
- ✅ **API Integration**: Full frontend-backend communication
- ✅ **Error Handling**: Proper error messages and loading states
- ✅ **Logout Functionality**: Secure token cleanup

### In Development
- 🔄 **User Profile Management**: Edit profile, change password
- 🔄 **Credit System Integration**: Purchase and track credits

### Not Yet Implemented
- ❌ **Podcast Generation Pipeline**: AI agent system
- ❌ **Payment Processing**: Stripe integration
- ❌ **Audio Playback**: Example podcast audio files
- ❌ **File Storage**: AWS S3 integration
- ❌ **AI Services**: OpenAI/ElevenLabs connection

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

# External APIs (add your keys when ready)
OPENAI_API_KEY=your-openai-api-key-here
ELEVENLABS_API_KEY=your-elevenlabs-api-key-here

# Stripe (add your keys when ready)
STRIPE_SECRET_KEY=your-stripe-secret-key-here
STRIPE_WEBHOOK_SECRET=your-stripe-webhook-secret-here

# AWS S3 (add your keys when ready)
AWS_ACCESS_KEY_ID=your-aws-access-key-here
AWS_SECRET_ACCESS_KEY=your-aws-secret-key-here
S3_BUCKET_NAME=your-s3-bucket-name-here

# Application
DEBUG=true
```

### Frontend (.env.local)
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
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

### User Registration & Login Flow
1. Visit http://localhost:3000
2. Click "Get Started" → "Sign up"
3. Register with email and password
4. Login with your credentials
5. Access the dashboard with your profile

### API Testing
- Visit http://localhost:8000/docs for interactive API documentation
- Test authentication endpoints
- Verify user creation and login

## 📋 Development Tasks

See `tasks.txt` for detailed development progress and upcoming features.

**Current Phase**: User Management & UI (Phase 2)
**Next Phase**: Payment Integration or AI Pipeline Development

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
