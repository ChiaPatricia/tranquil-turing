import streamlit as st
import pandas as pd
import google.generativeai as genai
import whisper
import torch
import re
import numpy as np
import tempfile
import os
import json
from pathlib import Path
from moviepy import VideoFileClip
from pyannote.audio import Pipeline
import time
import ffmpeg

# MediaProcessor class handles media processing (transcription and diarization)
class MediaProcessor:
    def __init__(self, auth_token: str):
        """
        Initialize with HuggingFace auth token for speaker diarization
        """
        # Load Whisper model
        self.whisper_model = whisper.load_model("medium")
        # Initialize PyAnnote speaker diarization pipeline
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token=auth_token
        )
        self.supported_formats = {
            'audio': ['.mp3', '.wav', '.m4a', '.ogg', '.flac'],
            'video': ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        }

    def process_media(self, file, progress_bar=None) -> pd.DataFrame:
        """Process audio or video file and return transcript DataFrame"""
        file_ext = Path(file.name).suffix.lower()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / file.name

            # Save uploaded file
            with open(temp_path, 'wb') as f:
                f.write(file.getvalue())

            # Convert video to audio if necessary
            if file_ext in self.supported_formats['video']:
                audio_path = self._extract_audio_from_video(temp_path)
            else:
                audio_path = temp_path

            # Process audio
            return self._process_audio_file(audio_path, progress_bar)

    def _extract_audio_from_video(self, video_path: Path) -> Path:
        """Extract audio from video file"""
        audio_path = video_path.with_suffix('.wav')
        video = VideoFileClip(str(video_path))
        video.audio.write_audiofile(str(audio_path))
        video.close()
        return audio_path

    def _process_audio_file(self, audio_path: Path, progress_bar) -> pd.DataFrame:
        """
        Process audio file with transcription and diarization
        Returns DataFrame with speaker-separated transcript
        """
        if progress_bar:
            progress_bar.progress(0.1)
            progress_bar.text("Transcribing audio...")

        # Transcribe audio using Whisper
        transcription = self.whisper_model.transcribe(str(audio_path))

        if progress_bar:
            progress_bar.progress(0.5)
            progress_bar.text("Performing speaker diarization...")

        # Perform speaker diarization
        diarization = self.diarization_pipeline(str(audio_path))

        if progress_bar:
            progress_bar.progress(0.8)
            progress_bar.text("Aligning transcription with speakers...")

        # Align transcription with speaker segments
        transcript_data = self._align_transcript_with_speakers(
            transcription, diarization
        )

        if progress_bar:
            progress_bar.progress(1.0)
            progress_bar.text("Processing complete!")

        return pd.DataFrame(transcript_data)

    def _align_transcript_with_speakers(self, transcription, diarization):
        """
        Align transcription with speaker segments
        Returns list of dicts with aligned data
        """
        # Prepare a list to hold the aligned segments
        segments = []
        # Iterate over diarization segments
        for segment in diarization.itersegments():
            speaker = diarization[segment]
            # Find corresponding text from transcription
            text = self._find_text_in_timerange(
                transcription['segments'],
                segment.start,
                segment.end
            )
            if text:
                segments.append({
                    'P or C': 'P' if speaker == 'SPEAKER_00' else 'C',
                    'Content of Utterance': text,
                    'Start Time': segment.start,
                    'End Time': segment.end,
                    'Speaker': speaker
                })
        return segments

    @staticmethod
    def _find_text_in_timerange(segments, start_time, end_time):
        """Find transcribed text within a time range"""
        relevant_segments = [
            seg['text'] for seg in segments
            if (seg['start'] >= start_time and seg['end'] <= end_time)
        ]
        return ' '.join(relevant_segments).strip()

# MITIAnalyzer class handles analysis and scoring using Google Gemini API
class MITIAnalyzer:
    def __init__(self, api_key):
        # Set the API key for Google Gemini
        genai.configure(api_key=api_key)
        self.global_scores = {
            "cultivating_change": None,
            "softening_sustain-talk": None,
            "partnership": None,
            "empathy": None
        }
        self.behavior_counts = {
            "gi": 0,  # Giving Information
            "persuade": 0,
            "persuade_with": 0,  # Persuade with Permission
            "question": 0,
            "sr": 0,  # Simple Reflection
            "cr": 0,  # Complex Reflection
            "affirm": 0,
            "seek": 0,  # Seeking Collaboration
            "emphasize": 0,  # Emphasizing Autonomy
            "confront": 0
        }

    def extract_score(self, response_text):
        """Extract numerical score from Gemini API response"""
        # Look for patterns like "Score: X" or "I would rate this as X"
        score_patterns = [
            r"score.*?([1-5])",
            r"rate.*?([1-5])",
            r"([1-5]).*?out of 5"
        ]

        for pattern in score_patterns:
            match = re.search(pattern, response_text.lower())
            if match:
                return int(match.group(1))
        return None

    def analyze_transcript(self, transcript_df):
        """Analyze transcript and generate all MITI scores"""
        # Analyze global scores
        model = genai.GenerativeModel('gemini-1.5-flash')
        generation_config = genai.GenerationConfig(max_output_tokens=2048)
        for dimension in self.global_scores.keys():
            prompt = self.load_prompt(f"prompts/prompts/0{list(self.global_scores.keys()).index(dimension)+1}-MITI-global-{dimension.replace('_', '-')}.md")
        
            full_prompt = f"{prompt}\n\n<transcript>\n{transcript_df.to_csv(index=False)}\n</transcript>"
            
            response = model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            score = self.extract_score(response.text)
            self.global_scores[dimension] = score

        # Analyze behavior counts
        self.count_behaviors(transcript_df)

    def count_behaviors(self, transcript_df):
        """Count specific behaviors in transcript"""
        model = genai.GenerativeModel('gemini-1.5-flash')
        generation_config = genai.GenerationConfig(max_output_tokens=2048)
        # Create behavior detection prompt
        behavior_prompt = """
You are an expert in Motivational Interviewing. Analyze the following therapist utterance and identify any of these behaviors:
- Giving Information (GI)
- Persuade
- Persuade with Permission
- Question (Q)
- Simple Reflection (SR)
- Complex Reflection (CR)
- Affirm (AF)
- Seeking Collaboration (Seek)
- Emphasizing Autonomy (Emphasize)
- Confront

Return results in JSON format, e.g., {"GI":1, "Persuade":0, ...}
"""

        for _, row in transcript_df.iterrows():
            if row['P or C'] == 'P':  # Provider/Therapist utterance
                
                behavior_full_prompt = f"{behavior_prompt}\n\nUtterance: {row['Content of Utterance']}"
                response = model.generate_content(
                    behavior_full_prompt,
                    generation_config=generation_config
                )
                try:
                    # Extract JSON from response
                    behaviors = json.loads(response.text)
                    for behavior, count in behaviors.items():
                        key = behavior.lower().replace(" ", "_")
                        if key in self.behavior_counts:
                            self.behavior_counts[key] += count
                except Exception as e:
                    st.warning(f"Could not parse behaviors for utterance: {row['Content of Utterance']}\nError: {e}")

    def calculate_summary_scores(self):
        """Calculate MITI summary scores"""
        summary = {}

        # Technical Global
        if all(self.global_scores[s] is not None for s in ['cultivating_change', 'softening_sustain-talk']):
            summary['technical'] = (self.global_scores['cultivating_change'] +
                                    self.global_scores['softening_sustain-talk']) / 2

        # Relational Global
        if all(self.global_scores[s] is not None for s in ['partnership', 'empathy']):
            summary['relational'] = (self.global_scores['partnership'] +
                                     self.global_scores['empathy']) / 2

        # % Complex Reflections
        total_reflections = self.behavior_counts['sr'] + self.behavior_counts['cr']
        if total_reflections > 0:
            summary['pct_cr'] = (self.behavior_counts['cr'] / total_reflections) * 100

        # Reflection-to-Question Ratio
        if self.behavior_counts['question'] > 0:
            summary['r_to_q'] = total_reflections / self.behavior_counts['question']

        # Total MI-Adherent
        summary['total_mia'] = (self.behavior_counts['seek'] +
                                self.behavior_counts['affirm'] +
                                self.behavior_counts['emphasize'])

        # Total MI Non-Adherent
        summary['total_mina'] = (self.behavior_counts['confront'] +
                                 self.behavior_counts['persuade'])

        return summary

    @staticmethod
    def load_prompt(filename):
        """Load prompt from file"""
        try:
            with open(filename, 'r') as f:
                return f.read()
        except Exception as e:
            st.error(f"Could not load prompt file: {filename}\nError: {e}")
            return ""

def render_miti_results(analyzer):
    """Render MITI results in Streamlit"""
    st.header("MITI Evaluation Results")

    # Global Scores
    st.subheader("Global Scores")
    global_scores_df = pd.DataFrame(analyzer.global_scores.items(), columns=['Dimension', 'Score'])
    st.table(global_scores_df)

    # Behavior Counts
    st.subheader("Behavior Counts")
    counts_df = pd.DataFrame(analyzer.behavior_counts.items(), columns=['Behavior', 'Count'])
    st.table(counts_df)

    # Summary Scores
    st.subheader("Summary Scores")
    summary = analyzer.calculate_summary_scores()
    summary_items = summary.items()
    if summary_items:
        summary_df = pd.DataFrame(summary_items, columns=['Metric', 'Value'])
        st.table(summary_df)
    else:
        st.write("No summary scores available.")

def export_results(analyzer, export_format):
    """Export results in specified format"""
    results = {
        'global_scores': analyzer.global_scores,
        'behavior_counts': analyzer.behavior_counts,
        'summary_scores': analyzer.calculate_summary_scores()
    }
    if export_format == "JSON":
        return json.dumps(results, indent=2)
    elif export_format == "CSV":
        # Convert results to CSV format
        all_results = {**analyzer.global_scores, **analyzer.behavior_counts, **analyzer.calculate_summary_scores()}
        df = pd.DataFrame(list(all_results.items()), columns=['Metric', 'Value'])
        return df.to_csv(index=False)
    elif export_format == "TXT":
        # Plain text format
        output = ""
        output += "Global Scores:\n"
        for k, v in analyzer.global_scores.items():
            output += f"{k}: {v}\n"
        output += "\nBehavior Counts:\n"
        for k, v in analyzer.behavior_counts.items():
            output += f"{k}: {v}\n"
        output += "\nSummary Scores:\n"
        for k, v in analyzer.calculate_summary_scores().items():
            output += f"{k}: {v}\n"
        return output

def main():
    st.title("MITI Session Analyzer")

    # Hide Streamlit's default hamburger menu
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # Initialize processors
    if 'media_processor' not in st.session_state:
        if "HF_AUTH_TOKEN" not in st.secrets:
            st.error("Hugging Face Auth Token not found. Please add it to Streamlit secrets.")
            return
        st.session_state.media_processor = MediaProcessor(
            auth_token=st.secrets["HF_AUTH_TOKEN"]
        )
    if 'miti_analyzer' not in st.session_state:
        if "GEMINI_API_KEY" not in st.secrets:
            st.error("Gemini API Key not found. Please add it to Streamlit secrets.")
            return
        st.session_state.miti_analyzer = MITIAnalyzer(
            api_key=st.secrets["GEMINI_API_KEY"]
        )

    # File upload section
    st.subheader("Upload Session Recording or Transcript")

    file_type = st.radio(
        "Select input type:",
        ["Audio/Video Recording", "Text Transcript"]
    )

    if file_type == "Audio/Video Recording":
        supported_formats = (
            st.session_state.media_processor.supported_formats['audio'] +
            st.session_state.media_processor.supported_formats['video']
        )

        uploaded_file = st.file_uploader(
            "Upload recording",
            type=[fmt[1:] for fmt in supported_formats]
        )

        if uploaded_file:
            progress_bar = st.progress(0)
            with st.spinner("Processing media file..."):
                try:
                    transcript_df = st.session_state.media_processor.process_media(
                        uploaded_file,
                        progress_bar
                    )
                    st.session_state.transcript_df = transcript_df

                    # Display transcript
                    st.subheader("Generated Transcript")
                    st.dataframe(transcript_df)

                    # Allow transcript editing
                    if st.checkbox("Edit Transcript"):
                        st.session_state.transcript_df = st.data_editor(
                            transcript_df,
                            num_rows="dynamic"
                        )

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    else:  # Text Transcript
        uploaded_file = st.file_uploader(
            "Upload transcript (CSV format)",
            type=['csv']
        )

        if uploaded_file:
            try:
                transcript_df = pd.read_csv(uploaded_file)
                st.session_state.transcript_df = transcript_df
                st.subheader("Transcript")
                st.dataframe(transcript_df)
                # Allow transcript editing
                if st.checkbox("Edit Transcript"):
                    st.session_state.transcript_df = st.data_editor(
                        transcript_df,
                        num_rows="dynamic"
                    )

            except Exception as e:
                st.error(f"Error reading transcript: {str(e)}")

    # Analysis section
    if 'transcript_df' in st.session_state:
        st.subheader("MITI Analysis")

        if st.button("Generate MITI Ratings"):
            with st.spinner("Analyzing session..."):
                st.session_state.miti_analyzer.analyze_transcript(
                    st.session_state.transcript_df
                )
                render_miti_results(st.session_state.miti_analyzer)

                # Save results
                st.session_state.last_results = st.session_state.miti_analyzer

    # Export options
    if 'last_results' in st.session_state:
        st.subheader("Export Analysis Report")
        export_format = st.selectbox(
            "Select export format",
            ["JSON", "CSV", "TXT"]
        )

        if st.button("Download Report"):
            report_data = export_results(
                st.session_state.last_results,
                export_format
            )
            file_extension = export_format.lower()
            st.download_button(
                label="Download Report",
                data=report_data,
                file_name=f"miti_analysis.{file_extension}",
                mime=f"text/{file_extension}" if export_format != 'JSON' else 'application/json'
            )

if __name__ == "__main__":
    main()
