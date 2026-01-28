"""
Clinical Report Generator
=========================
Generates comprehensive clinical reports for stutter analysis.

This module combines:
- WavLM stutter detection (WHAT type of stutter, WHERE it occurs)
- Whisper transcription (WHAT words were spoken, WHEN)
- Clinical metrics and severity scoring

Output includes:
- Patient summary
- Stutter frequency analysis
- Type breakdown with percentages
- Word-level stutter mapping
- Severity score (Mild/Moderate/Severe)
- Clinical recommendations
- Data for visualization (charts)
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import our inference modules
from inference.WaveLM_inference import analyze_audio, load_wavlm_model, STUTTER_LABELS
from inference.Whisper_inference import transcribe_audio, load_whisper_model, get_words_with_timestamps


# ============================================================================
# SEVERITY CLASSIFICATION
# ============================================================================
# Based on Stuttering Severity Instrument (SSI-4) guidelines
# These thresholds are approximate and should be calibrated with clinical data

SEVERITY_THRESHOLDS = {
    'very_mild': 5,      # < 5% words stuttered
    'mild': 10,          # 5-10% words stuttered
    'moderate': 20,      # 10-20% words stuttered
    'severe': 30,        # 20-30% words stuttered
    'very_severe': 100   # > 30% words stuttered
}


def calculate_severity_score(stutter_percentage: float) -> Tuple[str, int]:
    """
    Calculate severity level based on stutter percentage (Words Stuttered).
    
    Uses clinical guidelines adjusted for % Words Stuttered (%WS).
    
    Args:
        stutter_percentage: Percentage of words with detected stutters
    
    Returns:
        Tuple of (severity_label, severity_score 1-5)
    """
    if stutter_percentage < SEVERITY_THRESHOLDS['very_mild']:
        return 'Very Mild', 1
    elif stutter_percentage < SEVERITY_THRESHOLDS['mild']:
        return 'Mild', 2
    elif stutter_percentage < SEVERITY_THRESHOLDS['moderate']:
        return 'Moderate', 3
    elif stutter_percentage < SEVERITY_THRESHOLDS['severe']:
        return 'Severe', 4
    else:
        return 'Very Severe', 5


# ============================================================================
# WORD-STUTTER MAPPING
# ============================================================================

def map_stutters_to_words(wavlm_results: Dict, whisper_words: List[Dict]) -> List[Dict]:
    """
    Map detected stutters to specific words based on timestamps.
    
    This is the KEY function that combines WavLM + Whisper!
    
    Logic:
    - For each word from Whisper (with start/end time)
    - Find which chunk(s) it overlaps with
    - If that chunk has detected stutters, associate them with the word
    
    Args:
        wavlm_results: Output from analyze_audio() - chunks with stutter predictions
        whisper_words: Output from get_words_with_timestamps() - words with times
    
    Returns:
        List of words with their associated stutters:
        [{'word': 'hello', 'start': 0.5, 'end': 1.0, 'stutters': ['Prolongation']}, ...]
    """
    word_stutters = []
    
    for word_info in whisper_words:
        word_start = word_info['start']
        word_end = word_info['end']
        word_text = word_info['word']
        
        # Find overlapping chunks
        associated_stutters = set()
        
        for chunk in wavlm_results['chunks']:
            chunk_start = chunk['start_time']
            chunk_end = chunk['end_time']
            
            # Check if word overlaps with this chunk
            # Overlap exists if: word_start < chunk_end AND word_end > chunk_start
            if word_start < chunk_end and word_end > chunk_start:
                # Add any detected stutters from this chunk
                for stutter_type in chunk['prediction']['detected']:
                    associated_stutters.add(stutter_type)
        
        word_stutters.append({
            'word': word_text,
            'start': word_start,
            'end': word_end,
            'stutters': list(associated_stutters),
            'has_stutter': len(associated_stutters) > 0
        })
    
    return word_stutters


# ============================================================================
# CLINICAL METRICS CALCULATION
# ============================================================================

def calculate_clinical_metrics(wavlm_results: Dict, word_stutters: List[Dict]) -> Dict:
    """
    Calculate detailed clinical metrics for the report.
    
    Metrics include:
    - Stuttering frequency (% of words/chunks affected)
    - Type distribution (which stutters are most common)
    - Duration statistics
    - Speaking rate estimation
    
    Args:
        wavlm_results: Output from WavLM analysis
        word_stutters: Output from map_stutters_to_words()
    
    Returns:
        Dictionary of clinical metrics
    """
    summary = wavlm_results['summary']
    
    # Basic counts
    total_chunks = summary['total_chunks']
    total_words = len(word_stutters)
    words_with_stutter = sum(1 for w in word_stutters if w['has_stutter'])
    
    # Calculate percentages
    chunk_stutter_rate = summary['stutter_percentage']
    word_stutter_rate = (words_with_stutter / total_words * 100) if total_words > 0 else 0
    
    # Type distribution (for pie chart)
    type_counts = summary['stutter_counts']
    total_stutter_instances = sum(type_counts.values())
    
    type_distribution = {}
    for stutter_type, count in type_counts.items():
        percentage = (count / total_stutter_instances * 100) if total_stutter_instances > 0 else 0
        type_distribution[stutter_type] = {
            'count': count,
            'percentage': round(percentage, 1)
        }
    
    # Audio duration
    if wavlm_results['chunks']:
        last_chunk = wavlm_results['chunks'][-1]
        total_duration = last_chunk['end_time']
    else:
        total_duration = 0
    
    # Speaking rate (words per minute)
    speaking_rate = (total_words / total_duration * 60) if total_duration > 0 else 0
    
    # Severity calculation
    # Changed from chunk_stutter_rate to word_stutter_rate for better clinical accuracy
    # % Words Stuttered (%WS) is essentially the standard clinical metric (proxy for %SS)
    severity_label, severity_score = calculate_severity_score(word_stutter_rate)
    
    return {
        'total_duration_sec': round(total_duration, 2),
        'total_chunks': total_chunks,
        'total_words': total_words,
        'words_with_stutter': words_with_stutter,
        'chunk_stutter_rate': round(chunk_stutter_rate, 1),
        'word_stutter_rate': round(word_stutter_rate, 1),
        'type_distribution': type_distribution,
        'speaking_rate_wpm': round(speaking_rate, 1),
        'severity_label': severity_label,
        'severity_score': severity_score,
        'total_stutter_instances': total_stutter_instances
    }


# ============================================================================
# RECOMMENDATIONS GENERATOR
# ============================================================================

def generate_recommendations(metrics: Dict) -> List[str]:
    """
    Generate clinical recommendations based on analysis results.
    
    These are general suggestions - actual clinical advice should come
    from qualified speech-language pathologists.
    
    Args:
        metrics: Output from calculate_clinical_metrics()
    
    Returns:
        List of recommendation strings
    """
    recommendations = []
    severity = metrics['severity_label']
    type_dist = metrics['type_distribution']
    
    # Severity-based recommendations
    if severity in ['Very Mild', 'Mild']:
        recommendations.append(
            "Stuttering frequency is within mild range. Regular monitoring recommended."
        )
    elif severity == 'Moderate':
        recommendations.append(
            "Moderate stuttering detected. Consider speech therapy consultation for "
            "fluency-enhancing techniques."
        )
    else:  # Severe or Very Severe
        recommendations.append(
            "Significant stuttering detected. Professional speech-language pathology "
            "evaluation strongly recommended."
        )
    
    # Type-specific recommendations
    # Find the most common stutter type
    if type_dist:
        most_common = max(type_dist.items(), key=lambda x: x[1]['count'])
        stutter_type, data = most_common
        
        if data['count'] > 0:
            if stutter_type == 'Prolongation':
                recommendations.append(
                    f"Prolongations are most frequent ({data['percentage']}%). "
                    "Techniques like easy onset and light contact may help."
                )
            elif stutter_type == 'Block':
                recommendations.append(
                    f"Blocks are most frequent ({data['percentage']}%). "
                    "Breathing techniques and voluntary stuttering practice may be beneficial."
                )
            elif stutter_type == 'SoundRep':
                recommendations.append(
                    f"Sound repetitions are most frequent ({data['percentage']}%). "
                    "Slow speech techniques and pull-outs may help."
                )
            elif stutter_type == 'WordRep':
                recommendations.append(
                    f"Word repetitions are most frequent ({data['percentage']}%). "
                    "Pacing strategies and pausing techniques recommended."
                )
            elif stutter_type == 'Interjection':
                recommendations.append(
                    f"Interjections are most frequent ({data['percentage']}%). "
                    "May indicate word-finding difficulties or anxiety. "
                    "Relaxation techniques may help."
                )
    
    # Speaking rate recommendation
    if metrics['speaking_rate_wpm'] > 180:
        recommendations.append(
            f"Speaking rate is high ({metrics['speaking_rate_wpm']} WPM). "
            "Slower, more deliberate speech may reduce stuttering."
        )
    
    return recommendations


# ============================================================================
# TIMELINE DATA GENERATOR (FOR VISUALIZATION)
# ============================================================================

def generate_timeline_data(wavlm_results: Dict) -> List[Dict]:
    """
    Generate timeline data for visualization (chart showing stutters over time).
    
    Args:
        wavlm_results: Output from WavLM analysis
    
    Returns:
        List of data points for timeline chart:
        [{'time': 1.5, 'stutters': 2, 'types': ['Block', 'SoundRep']}, ...]
    """
    timeline = []
    
    for chunk in wavlm_results['chunks']:
        midpoint = (chunk['start_time'] + chunk['end_time']) / 2
        detected = chunk['prediction']['detected']
        
        timeline.append({
            'time': round(midpoint, 2),
            'start': chunk['start_time'],
            'end': chunk['end_time'],
            'stutter_count': len(detected),
            'types': detected,
            'probabilities': chunk['prediction']['probabilities']
        })
    
    return timeline


# ============================================================================
# MAIN REPORT GENERATOR
# ============================================================================

def generate_clinical_report(
    audio_path: str,
    patient_name: str = "Anonymous",
    patient_id: str = None,
    clinician_name: str = None,
    wavlm_model=None,
    wavlm_device=None,
    whisper_model=None,
    threshold: float = 0.4
) -> Dict:
    """
    Generate a comprehensive clinical report for stutter analysis.
    
    This is the MAIN function that orchestrates everything:
    1. Run WavLM inference (stutter detection)
    2. Run Whisper inference (transcription)
    3. Map stutters to words
    4. Calculate clinical metrics
    5. Generate recommendations
    6. Package everything into a report
    
    Args:
        audio_path: Path to the audio file to analyze
        patient_name: Name for the report (default: Anonymous)
        patient_id: Optional patient ID
        clinician_name: Optional clinician name
        wavlm_model: Pre-loaded WavLM model (optional)
        wavlm_device: Device for WavLM
        whisper_model: Pre-loaded Whisper model (optional)
        threshold: Detection threshold (default 0.4 for clinical sensitivity)
    
    Returns:
        Complete clinical report as a dictionary
    """
    print("=" * 60)
    print("GENERATING CLINICAL REPORT")
    print("=" * 60)
    
    # Generate unique report ID
    report_id = f"RPT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # -------------------------------------------------------------------------
    # Step 1: Load models if not provided
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading models...")
    
    if wavlm_model is None:
        wavlm_model, wavlm_device = load_wavlm_model()
    
    if whisper_model is None:
        whisper_model = load_whisper_model("base")
    
    # -------------------------------------------------------------------------
    # Step 2: Run WavLM stutter detection
    # -------------------------------------------------------------------------
    print("\n[2/5] Analyzing stutters with WavLM...")
    
    wavlm_results = analyze_audio(
        audio_path,
        model=wavlm_model,
        device=wavlm_device,
        threshold=threshold
    )
    
    # -------------------------------------------------------------------------
    # Step 3: Run Whisper transcription
    # -------------------------------------------------------------------------
    print("\n[3/5] Transcribing with Whisper...")
    
    whisper_result = transcribe_audio(audio_path, model=whisper_model)
    transcription = whisper_result['text']
    whisper_words = get_words_with_timestamps(whisper_result)
    
    # -------------------------------------------------------------------------
    # Step 4: Map stutters to words
    # -------------------------------------------------------------------------
    print("\n[4/5] Mapping stutters to words...")
    
    word_stutters = map_stutters_to_words(wavlm_results, whisper_words)
    
    # -------------------------------------------------------------------------
    # Step 5: Calculate metrics and generate report
    # -------------------------------------------------------------------------
    print("\n[5/5] Generating report...")
    
    metrics = calculate_clinical_metrics(wavlm_results, word_stutters)
    recommendations = generate_recommendations(metrics)
    timeline_data = generate_timeline_data(wavlm_results)
    
    # -------------------------------------------------------------------------
    # Compile final report
    # -------------------------------------------------------------------------
    report = {
        # Header information
        'report_id': report_id,
        'generated_at': datetime.now().isoformat(),
        'patient_info': {
            'name': patient_name,
            'id': patient_id or 'N/A'
        },
        'clinician': clinician_name or 'N/A',
        'audio_file': os.path.basename(audio_path),
        
        # Transcription
        'transcription': {
            'full_text': transcription,
            'word_count': len(whisper_words),
            'words_with_timestamps': whisper_words
        },
        
        # Stutter analysis
        'stutter_analysis': {
            'word_level': word_stutters,
            'chunk_level': wavlm_results['chunks'],
            'timeline': timeline_data
        },
        
        # Clinical metrics
        'metrics': metrics,
        
        # Severity assessment
        'severity': {
            'label': metrics['severity_label'],
            'score': metrics['severity_score'],
            'description': f"Based on {metrics['word_stutter_rate']}% word stutter rate"
        },
        
        # Recommendations
        'recommendations': recommendations,

        # Stutter Definitions (Clinical Glossary)
        'stutter_definitions': {
            'Prolongation': 'Sound or airflow continues for an unusual length of time (e.g., "Ssssssnake"). Often accompanied by a feeling of something being "stuck" moving forward.',
            'Block': 'Airflow and sound are completely stopped during speech production. May involve visible tension in face or neck.',
            'SoundRep': 'Repetition of a sound or syllable (e.g., "B-b-b-ball"). Also known as part-word repetition.',
            'WordRep': 'Repetition of a whole word (e.g., "I-I-I want"). Often used as a starter or to hold the floor.',
            'Interjection': 'Extra words or sounds (e.g., "um", "uh", "like") inserted into the speech stream. Often used as fillers or to delay a feared word.'
        },
        
        # Chart data (for frontend visualization)
        'chart_data': {
            'type_distribution': [
                {'name': k, 'value': v['count'], 'percentage': v['percentage']}
                for k, v in metrics['type_distribution'].items()
            ],
            'timeline': timeline_data,
            'severity_gauge': {
                'value': metrics['severity_score'],
                'max': 5,
                'label': metrics['severity_label']
            }
        },
        
        # Analysis settings
        'settings': {
            'threshold': threshold,
            'chunk_duration': wavlm_results['summary']['chunk_duration_sec']
        }
    }
    
    print("\n" + "=" * 60)
    print(f"Report generated: {report_id}")
    print(f"Severity: {metrics['severity_label']} ({metrics['severity_score']}/5)")
    print(f"Stutter rate: {metrics['chunk_stutter_rate']}%")
    print("=" * 60)
    
    return report


def save_report_json(report: Dict, output_path: str = None) -> str:
    """
    Save report as JSON file.
    
    Args:
        report: Report dictionary from generate_clinical_report()
        output_path: Optional custom path. If None, saves to reports/ folder
    
    Returns:
        Path to saved file
    """
    if output_path is None:
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join(project_root, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        output_path = os.path.join(reports_dir, f"{report['report_id']}.json")
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to: {output_path}")
    return output_path


# ============================================================================
# ENTRY POINT FOR TESTING
# ============================================================================

if __name__ == "__main__":
    # Test with a sample audio file
    test_audio = "data/clips/HVSA/0/HVSA_0_4.wav"
    
    # Generate report
    report = generate_clinical_report(
        audio_path=test_audio,
        patient_name="Test Patient",
        patient_id="TEST-001",
        clinician_name="Dr. Test"
    )
    
    # Save report
    save_report_json(report)
    
    # Print summary
    print("\n" + "=" * 60)
    print("REPORT SUMMARY")
    print("=" * 60)
    print(f"\nTranscription: {report['transcription']['full_text']}")
    print(f"\nSeverity: {report['severity']['label']}")
    print(f"Stutter Rate: {report['metrics']['chunk_stutter_rate']}%")
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
