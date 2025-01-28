# MI Session Analyzer
*Advancing Motivational Interviewing through AI-powered feedback*

We're tackling one of healthcare's most pressing challenges: the global shortage of mental health support. While addiction affects 39.5 million individuals worldwide, only 20% access treatment. Our mission is to democratize access to evidence-based therapy by empowering practitioners with real-time, AI-driven insights.

## The Challenge
Motivational Interviewing (MI) is a proven intervention technique, but mastering it requires extensive practice and expert feedback. Traditional supervision is:
- Resource-intensive
- Geographically limited
- Time-consuming
- Expensive to scale

## Our Solution
We've developed an AI-powered platform that provides instant, structured feedback on MI sessions using the gold-standard MITI (Motivational Interviewing Treatment Integrity) coding system.

### Core Technology Stack
```
Frontend
└── Streamlit
    └── Interactive UI for session analysis

Speech Processing
├── Whisper
│   └── Advanced speech-to-text
└── PyAnnote
    └── Speaker diarization

AI/ML
└── Google Gemini
    └── Natural language understanding
    └── MITI scoring algorithms

Data Pipeline
├── Pandas
├── NumPy
└── FFmpeg
```

### Key Features

#### Automated Analysis
- Real-time transcription with speaker identification
- MITI scoring across multiple dimensions
- Behavioral pattern recognition
- Trend analysis over multiple sessions

#### Comprehensive Metrics
```python
Global Ratings
├── Technical Components
│   ├── Cultivating Change Talk
│   └── Softening Sustain Talk
└── Relational Components
    ├── Partnership
    └── Empathy

Behavior Counts
├── Core MI Skills
│   ├── Reflections (Simple/Complex)
│   ├── Questions
│   └── Information Giving
└── MI-Adherent Behaviors
    ├── Affirmations
    ├── Seeking Collaboration
    └── Emphasizing Autonomy
```

## Getting Started

### Prerequisites
```bash
# Required API keys
GEMINI_API_KEY="from-google-cloud-console"
HF_AUTH_TOKEN="from-huggingface"
```

### Quick Start
```bash
# Clone repository
git clone https://github.com/ChiaPatricia/tranquil-turing.git

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app.py
```

## Built for MEXA
This project was developed during the MEXA hackathon, focusing on leveraging cutting-edge AI to solve critical Mental Health challenges.

## Research & Impact
- Potential to increase MI training capacity by 10x
- Reduce feedback cycle time from weeks to minutes
- Enable continuous improvement through data-driven insights
- Scale evidence-based addiction treatment globally

## Future Development
- [ ] Multi-language support
- [ ] Integration with telehealth platforms
- [ ] Advanced visualization of therapy dynamics
- [ ] Personalized improvement recommendations
- [ ] Secure cloud-based session management

## Contributing
We welcome contributions that align with our mission of improving mental healthcare access.

## License
This project is licensed under the [Apache License](http://www.apache.org/licenses/LICENSE-2.0
).

---

*Developed with ❤️ for the global mental health community*

*Part of the MEXA Hackathon 2025 - Transforming Healthcare Through AI*

## References:
Moyers, T.B., Manuel, J.K., & Ernst, D. (2015). Motivational Interviewing Treatment Integrity Coding Manual 4.2.1 (Unpublished manual) 
