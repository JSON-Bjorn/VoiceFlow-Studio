# VoiceFlow Studio Configuration Guide

## Required Environment Variables

To run VoiceFlow Studio with full functionality, you need to set up the following environment variables in a `.env` file in the backend directory:

### Database Configuration
```
DATABASE_URL=sqlite:///./voiceflow.db
```

### Security
```
SECRET_KEY=your-secret-key-here-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### OpenAI Configuration (Required for AI Pipeline)
```
OPENAI_API_KEY=sk-your-openai-api-key-here
```
**Note:** This is required for the enhanced 6-agent podcast generation pipeline to work. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys).

### ElevenLabs Configuration (Required for Voice Generation - Phase 6)
```
ELEVENLABS_API_KEY=your-elevenlabs-api-key-here
```
**Note:** This will be needed for voice synthesis in Phase 6. Get your API key from [ElevenLabs](https://elevenlabs.io/).

### Stripe Configuration (Required for Payments) ✅
```
STRIPE_SECRET_KEY=sk_test_your-stripe-secret-key-here
STRIPE_PUBLISHABLE_KEY=pk_test_your-stripe-publishable-key-here
STRIPE_WEBHOOK_SECRET=whsec_your-webhook-secret-here
```
**Note:** Required for credit purchases. Get your keys from [Stripe Dashboard](https://dashboard.stripe.com/apikeys).

### AWS S3 Configuration (Optional - for file storage)
```
AWS_ACCESS_KEY_ID=your-aws-access-key-id
AWS_SECRET_ACCESS_KEY=your-aws-secret-access-key
S3_BUCKET_NAME=your-s3-bucket-name
```

### Application Configuration
```
APP_NAME=VoiceFlow Studio
DEBUG=true
```

## Frontend Environment Variables

Create a `.env.local` file in the frontend directory:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_test_your-stripe-publishable-key-here
```

## Setup Instructions

1. **Create backend .env file:**
   ```bash
   cd backend
   cp .env.example .env  # If .env.example exists
   # OR create .env manually with the variables above
   ```

2. **Create frontend .env.local file:**
   ```bash
   cd frontend
   # Create .env.local with the frontend variables above
   ```

3. **Add your API keys:**
   - Replace `your-openai-api-key-here` with your actual OpenAI API key
   - Replace `your-stripe-secret-key-here` with your actual Stripe secret key
   - Replace other placeholder values with your actual credentials

4. **Test the configuration:**
   - Start the backend server
   - Visit `http://localhost:8000/docs` to check API documentation
   - Test the AI pipeline through the API endpoints

## Enhanced 6-Agent AI Pipeline Features ✅

With proper configuration, the following AI features are available:

### Phase 5 - Enhanced AI Pipeline Foundation ✅
- **OpenAI Service:** Core API integration with error handling and rate limiting
- **Research Agent:** Generates comprehensive topic research with subtopics, facts, and discussion angles
- **Script Agent:** Converts research into natural podcast dialogue with host personalities
- **Content Planning Agent:** Structures episodes with segments, timing, and content organization
- **Conversation Flow Agent:** Ensures natural dialogue transitions and engagement patterns
- **Dialogue Distribution Agent:** Balances speaking time and manages interaction patterns
- **Personality Adaptation Agent:** Maintains consistent character voices and personalities
- **Enhanced Pipeline Orchestrator:** Coordinates all agents with sophisticated workflow management
- **Advanced Memory/State Store:** Tracks generation progress and maintains conversation context

### Available API Endpoints

#### Testing & Configuration
- `GET /api/ai/config` - Get pipeline configuration status
- `POST /api/ai/test` - Test all pipeline components

#### Enhanced Pipeline Operations
- `POST /api/ai/research` - Generate comprehensive research for any topic
  ```json
  {
    "topic": "The future of renewable energy",
    "target_length": 10,
    "depth": "standard"
  }
  ```

- `POST /api/ai/script/generate` - Generate script from research data
- `POST /api/ai/generate/podcast` - Generate complete podcast using all 6 agents
- `GET /api/ai/generation/status` - Get generation progress and status

#### Agent-Specific Endpoints
- `POST /api/ai/content-planning` - Content Planning Agent operations
- `POST /api/ai/conversation-flow` - Conversation Flow Agent operations
- `POST /api/ai/dialogue-distribution` - Dialogue Distribution Agent operations
- `POST /api/ai/personality-adaptation` - Personality Adaptation Agent operations

## Current Implementation Status

### ✅ Completed Features
- **User Authentication System:** Complete registration, login, and profile management
- **Credit Management System:** Stripe integration with secure payment processing
- **Podcast Library Interface:** CRUD operations with search and filtering
- **Enhanced 6-Agent AI Pipeline:** Sophisticated multi-agent conversation generation
- **Database Integration:** SQLite with Alembic migrations
- **API Documentation:** Complete FastAPI documentation at `/docs`

### 🔄 In Progress
- **ElevenLabs TTS Integration:** Text-to-speech voice generation (Phase 6)
- **Voice Profile Management:** Distinct host voice profiles

### ❌ Not Yet Implemented
- **Audio File Generation:** ElevenLabs TTS integration
- **Audio Assembly:** Intro/outro music and effects
- **Podcast Generation UI:** Frontend interface for AI pipeline
- **File Storage:** AWS S3 integration
- **Real-time Progress:** Generation status tracking UI

## Development Notes

1. **OpenAI API Costs:** The pipeline uses GPT-4o-mini by default for cost efficiency
2. **Rate Limits:** OpenAI has rate limits - the service includes comprehensive error handling
3. **Quality Control:** Built-in validation ensures generated content meets quality thresholds
4. **Extensibility:** The 6-agent system is designed to be easily extended with new capabilities
5. **State Management:** Advanced memory systems maintain context across all agent operations
6. **Coordination:** Enhanced Pipeline Orchestrator ensures all agents work together seamlessly

## Troubleshooting

### "OpenAI API key not configured"
- Ensure `OPENAI_API_KEY` is set in your backend `.env` file
- Verify the API key is valid and has sufficient credits
- Check that the backend server has been restarted after adding the key

### "Stripe integration not working"
- Verify all three Stripe keys are set in both backend and frontend environments
- Check that webhook endpoints are properly configured
- Ensure you're using test keys during development

### "Pipeline test failed"
- Check your internet connection
- Verify all required environment variables are set
- Check the backend logs for specific error messages
- Ensure OpenAI API key has sufficient credits

### "Enhanced pipeline coordination issues"
- This usually indicates an agent communication problem
- Check the Enhanced Pipeline Orchestrator logs
- Verify all 6 agents are properly initialized
- Check memory/state store functionality

## Security Notes

- Never commit your `.env` files to version control
- Use test API keys during development
- For production, use strong `SECRET_KEY` and set `DEBUG=false`
- Regularly rotate your API keys
- Keep Stripe webhook secrets secure
- Monitor OpenAI API usage and costs

## Next Steps

1. **Phase 6:** Integrate ElevenLabs API for text-to-speech
2. **Phase 7:** Build podcast generation UI interface
3. **Phase 8:** Comprehensive testing and quality assurance
4. **Phase 9:** Production deployment and monitoring 