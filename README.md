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
- **Authentication**: NextAuth.js
- **Forms**: React Hook Form with Zod validation

### Backend
- **Framework**: FastAPI (Python)
- **Database**: PostgreSQL with SQLAlchemy ORM
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
│   └── alembic/             # Database migrations
├── tasks.txt               # Development progress tracking
└── README.md              # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ and npm
- Python 3.9+ and pip
- PostgreSQL database
- OpenAI API key
- ElevenLabs API key
- Stripe account (for payments)
- AWS S3 bucket (for file storage)

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - Features: Landing page, auth pages, example library

### Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   source venv/bin/activate      # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database URL
   ```

5. **Run database migrations**
   ```bash
   alembic upgrade head
   ```

6. **Start the server**
   ```bash
   python app/main.py
   ```

7. **Access the API**
   - Backend: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## 🔧 Environment Variables

### Backend (.env)
```env
# Database
DATABASE_URL=postgresql://user:password@localhost/voiceflow_db

# Security
SECRET_KEY=your-secret-key-here

# External APIs
OPENAI_API_KEY=your-openai-api-key
ELEVENLABS_API_KEY=your-elevenlabs-api-key

# Stripe
STRIPE_SECRET_KEY=your-stripe-secret-key
STRIPE_WEBHOOK_SECRET=your-stripe-webhook-secret

# AWS S3
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
S3_BUCKET_NAME=your-s3-bucket-name

# Application
DEBUG=false
```

### Frontend (.env.local)
```env
NEXTAUTH_SECRET=your-nextauth-secret
NEXTAUTH_URL=http://localhost:3000
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=your-stripe-publishable-key
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

### Components
- **Cards**: Glassmorphism effect with slate backgrounds
- **Buttons**: Purple primary with hover effects
- **Forms**: Dark theme with purple focus states
- **Navigation**: Clean, minimal design

## 📊 Development Progress

Track development progress in `tasks.txt`:
- ✅ Phase 1: Foundation & Setup (Partially Complete)
- 🔄 Current: Database setup and authentication
- 📋 Next: Payment integration and AI pipeline

### Completed Features
- [x] Modern landing page with hero section
- [x] Authentication pages (login/register)
- [x] Example podcast library
- [x] FastAPI backend structure
- [x] Database models (User, Podcast)
- [x] shadcn/ui component system

### In Progress
- [ ] PostgreSQL database connection
- [ ] Authentication middleware
- [ ] Credit system implementation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT API
- **ElevenLabs** for text-to-speech
- **Vercel** for Next.js framework
- **FastAPI** for the backend framework
- **shadcn/ui** for beautiful components
- **Tailwind CSS** for styling system

## 📞 Support

For support, email support@voiceflowstudio.com or join our Discord community.

---

**VoiceFlow Studio** - Transform ideas into professional podcasts with AI ✨
