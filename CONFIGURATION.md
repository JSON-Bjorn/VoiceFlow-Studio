# VoiceFlow Studio Configuration Guide

## Required Environment Variables

To run VoiceFlow Studio with full AI pipeline functionality, you need to set up the following environment variables in a `.env` file in the project root:

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
**Note:** This is required for the podcast generation pipeline to work. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys).

### ElevenLabs Configuration (Required for Voice Generation - Phase 6)
```
ELEVENLABS_API_KEY=your-elevenlabs-api-key-here
```
**Note:** This will be needed for voice synthesis in Phase 6. Get your API key from [ElevenLabs](https://elevenlabs.io/).

### Stripe Configuration (Required for Payments)
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

## Setup Instructions

1. **Create .env file:**
   ```bash
   cp .env.example .env  # If .env.example exists
   # OR create .env manually with the variables above
   ```

2. **Add your API keys:**
   - Replace `your-openai-api-key-here` with your actual OpenAI API key
   - Replace `your-stripe-secret-key-here` with your actual Stripe secret key
   - Replace other placeholder values with your actual credentials

3. **Test the configuration:**
   - Start the backend server
   - Visit `http://localhost:8000/api/ai/config` to check AI pipeline status
   - Visit `http://localhost:8000/api/ai/test` to test the OpenAI connection

## AI Pipeline Features

With proper configuration, the following AI features are available:

### Phase 5 - AI Pipeline Foundation ✅
- **OpenAI Service:** Core API integration with error handling
- **Research Agent:** Generates comprehensive topic research with subtopics, facts, and discussion angles
- **Script Agent:** Converts research into natural podcast dialogue with host personalities
- **Conversation Orchestrator:** Coordinates the entire generation pipeline with progress tracking
- **Memory/State Store:** Tracks generation progress and maintains state

### Available API Endpoints

#### Testing
- `GET /api/ai/test` - Test all pipeline components
- `POST /api/ai/test/simple` - Test with a simple topic
- `GET /api/ai/config` - Get pipeline configuration status

#### Research Generation
- `POST /api/ai/research` - Generate research for any topic
  ```json
  {
    "topic": "The future of renewable energy",
    "target_length": 10,
    "depth": "standard"
  }
  ```

#### Script Generation
- `POST /api/ai/script/generate` - Generate script from research data
- `POST /api/ai/generate/podcast` - Generate complete podcast (research + script)

#### Status Tracking
- `GET /api/ai/generation/status` - Get generation progress

## Development Notes

1. **OpenAI API Costs:** The pipeline uses GPT-4o-mini by default for cost efficiency
2. **Rate Limits:** OpenAI has rate limits - the service includes error handling for this
3. **Quality Control:** Built-in validation ensures generated content meets quality thresholds
4. **Extensibility:** The agent system is designed to be easily extended with new capabilities

## Troubleshooting

### "OpenAI API key not configured"
- Ensure `OPENAI_API_KEY` is set in your `.env` file
- Verify the API key is valid and has sufficient credits

### "Pipeline test failed"
- Check your internet connection
- Verify all required environment variables are set
- Check the backend logs for specific error messages

### "Research generation failed"
- This usually indicates an OpenAI API issue
- Check your API key and account status
- Verify you have sufficient OpenAI credits

## Security Notes

- Never commit your `.env` file to version control
- Use test API keys during development
- For production, use strong `SECRET_KEY` and set `DEBUG=false`
- Regularly rotate your API keys 