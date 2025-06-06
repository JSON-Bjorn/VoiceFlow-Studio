# VoiceFlow Studio Configuration Guide

This guide covers all configuration requirements for VoiceFlow Studio.

## Recent Updates - Deprecation Warnings Fixed

### Version 2.1.0 Updates
**Resolved Deprecation Warnings:**
- ✅ Fixed `LoRACompatibleLinear` deprecation by adding PEFT backend support
- ✅ Updated PyTorch attention mechanisms (replaced `torch.backends.cuda.sdp_kernel`)
- ✅ Added modern `torch.nn.attention.sdpa_kernel` support
- ✅ Upgraded dependencies to latest compatible versions

**To apply these fixes:**
```bash
cd backend
python upgrade_dependencies.py
```

## Environment Variables

### Database Configuration (Required)
```env
DATABASE_URL=postgresql://user:password@localhost/voiceflow_studio
```

### OpenAI Configuration (Required for Content Generation)
```env
OPENAI_API_KEY=your-openai-api-key-here
```
Get your API key from [OpenAI](https://platform.openai.com/api-keys).

### Chatterbox TTS Configuration (Local - No API Key Required)
```env
# Optional: Force specific device (auto-detected if not specified)
CHATTERBOX_DEVICE=cuda  # or "cpu"

# Optional: Custom model cache directory
CHATTERBOX_MODEL_CACHE=./storage/models
```

**Note:** Chatterbox TTS runs locally and doesn't require API keys. It will automatically detect if CUDA is available for GPU acceleration. For optimal performance, ensure you have:
- CUDA-compatible GPU (recommended)
- Sufficient disk space for model downloads (~1-2GB)
- Adequate RAM (4GB+ recommended)

### Stripe Configuration (Required for Payments)
```env
STRIPE_API_KEY=your-stripe-secret-key-here
STRIPE_PUBLISHABLE_KEY=your-stripe-publishable-key-here
STRIPE_WEBHOOK_SECRET=your-stripe-webhook-secret-here
```

### Security Configuration
```env
JWT_SECRET_KEY=your-super-secret-key-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
```

### Storage Configuration
```env
STORAGE_TYPE=local  # or "s3", "gcs"
STORAGE_BUCKET=your-bucket-name  # if using cloud storage
STORAGE_REGION=us-east-1  # if using cloud storage
```

### Audio Processing Configuration
```env
AUDIO_SAMPLE_RATE=22050
AUDIO_FORMAT=wav
```

## Hardware Requirements

### Minimum Requirements
- **CPU**: Modern multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space for models and audio cache
- **Network**: Stable internet for model downloads and OpenAI API

### Recommended for Optimal Performance
- **GPU**: CUDA-compatible GPU with 4GB+ VRAM (RTX 3060 or better)
- **RAM**: 16GB+ system RAM
- **Storage**: SSD with 20GB+ free space
- **CPU**: Intel i7/AMD Ryzen 7 or better

### GPU Support
Chatterbox TTS supports CUDA acceleration for significantly faster audio generation:
- **With CUDA GPU**: 2-5x faster generation, smooth real-time processing
- **CPU Only**: Slower but still functional, suitable for development/testing

## Installation Steps

### 1. Python Dependencies
```bash
cd backend
pip install -r requirements.txt
```

**Key Dependencies:**
- `chatterbox-tts` - Open-source TTS engine
- `torch` - PyTorch for ML operations
- `torchaudio` - Audio processing
- `transformers` - Model loading and inference

### 2. CUDA Setup (Optional but Recommended)
If you have a CUDA-compatible GPU:

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If CUDA is not detected, install PyTorch with CUDA support:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Model Download
On first run, Chatterbox will automatically download required models (~1-2GB). This is a one-time process.

### 4. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## Development vs Production

### Development Configuration
- Use local storage (`STORAGE_TYPE=local`)
- SQLite database for quick setup
- CPU-only Chatterbox for basic testing

### Production Configuration
- Cloud storage (S3/GCS) recommended
- PostgreSQL database
- GPU-enabled server for optimal TTS performance
- Proper SSL certificates
- Environment-specific secrets

## Troubleshooting

### Chatterbox TTS Issues

**Model Loading Errors:**
```bash
# Clear model cache and re-download
rm -rf ./storage/models/*
# Restart application to trigger fresh download
```

**CUDA Memory Issues:**
```env
# Force CPU mode if GPU memory is insufficient
CHATTERBOX_DEVICE=cpu
```

**Slow Generation:**
- Verify GPU is being used: Check logs for "device: cuda" message
- Monitor GPU usage during generation
- Consider upgrading to a more powerful GPU

### Audio Quality Issues
- Ensure audio sample rate matches your requirements (22050 Hz default)
- Check available disk space for audio cache
- Verify FFmpeg is installed for audio processing

### Performance Optimization
- Use SSD storage for faster model loading
- Ensure adequate cooling for sustained GPU usage
- Monitor system resources during podcast generation

## Migration from ElevenLabs

If migrating from a previous ElevenLabs setup:

1. **Remove ElevenLabs Configuration:**
   ```bash
   # Remove from .env
   # ELEVENLABS_API_KEY=...
   ```

2. **Update Dependencies:**
   ```bash
   pip uninstall elevenlabs
   pip install chatterbox-tts torch torchaudio
   ```

3. **Benefits of Migration:**
   - **Cost Savings**: No API costs, runs locally
   - **Privacy**: Audio never leaves your servers
   - **Customization**: Voice cloning with custom prompts
   - **Scalability**: No rate limits or usage restrictions

## Feature Status

### ✅ Implemented
- **Chatterbox TTS Integration:** Local text-to-speech generation
- **Voice Profile System:** Multiple distinct podcast host voices
- **GPU Acceleration:** CUDA support for fast generation
- **Voice Cloning:** Custom voice prompts for personalized audio
- **Audio Processing:** Complete podcast assembly pipeline

### 🔄 Current Development
- Generation progress tracking and real-time updates
- Advanced voice customization options
- Batch audio processing optimization

## Support

For configuration issues:
1. Check the logs for specific error messages
2. Verify all environment variables are set correctly
3. Ensure hardware requirements are met
4. Consult the project documentation

**Note**: Chatterbox TTS migration provides significant cost savings and improved performance compared to ElevenLabs API-based solutions. 