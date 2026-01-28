"""
Example usage scripts for Resume Skill Recognition System
"""

from pathlib import Path
from matching_engine import ResumeJDMatcher
from skill_extraction import SkillExtractor
from text_extraction import TextExtractor
from preprocessing import TextPreprocessor


def example_1_basic_skill_extraction():
    """Example 1: Basic skill extraction from text."""
    print("="*80)
    print("EXAMPLE 1: Basic Skill Extraction")
    print("="*80 + "\n")
    
    # Sample resume text
    resume_text = """
    John Doe
    Senior Software Engineer
    
    SKILLS:
    - Programming Languages: Python, Java, JavaScript, TypeScript
    - Machine Learning: TensorFlow, PyTorch, Scikit-learn
    - Tools: Docker, Kubernetes, Git, Jenkins
    - Frameworks: Django, Flask, React.js, Node.js
    - Soft Skills: Team Leadership, Problem Solving, Communication
    
    EXPERIENCE:
    Developed ML models using Python and TensorFlow for natural language processing.
    Built scalable web applications with Django and React.js.
    """
    
    # Extract skills
    extractor = SkillExtractor()
    skills = extractor.extract(resume_text)
    
    # Display results
    print("Extracted Skills:")
    for category, skill_list in skills.items():
        print(f"\n{category.upper()}:")
        for skill in skill_list:
            print(f"  • {skill}")
    
    print("\n" + "="*80 + "\n")


def example_2_resume_file_processing():
    """Example 2: Process a resume file."""
    print("="*80)
    print("EXAMPLE 2: Resume File Processing")
    print("="*80 + "\n")
    
    # Note: This requires an actual resume file
    resume_path = "data/resumes/sample_resume.pdf"
    
    if not Path(resume_path).exists():
        print(f"⚠️  Sample resume not found at: {resume_path}")
        print("   Please add a sample resume to test this example.")
        print("\n" + "="*80 + "\n")
        return
    
    # Extract text
    extractor = TextExtractor()
    result = extractor.extract(resume_path)
    
    if result['success']:
        print(f"✓ Text extraction successful!")
        print(f"  Method: {result['method']}")
        print(f"  Length: {len(result['text'])} characters")
        print(f"\nFirst 200 characters:")
        print(result['text'][:200] + "...")
    else:
        print(f"✗ Extraction failed: {result['error']}")
    
    print("\n" + "="*80 + "\n")


def example_3_text_preprocessing():
    """Example 3: Text preprocessing pipeline."""
    print("="*80)
    print("EXAMPLE 3: Text Preprocessing")
    print("="*80 + "\n")
    
    # Sample text
    raw_text = """
    Looking for a Python Developer with 5+ years of experience!!!
    Must have: Machine Learning, Deep Learning, TensorFlow, PyTorch.
    Strong communication skills and team leadership abilities.
    """
    
    # Preprocess
    preprocessor = TextPreprocessor(download_nltk_data=False)
    processed_text = preprocessor.preprocess(raw_text)
    
    # Get statistics
    stats = preprocessor.get_stats(raw_text)
    
    print(f"Original Text:\n{raw_text}\n")
    print(f"Processed Text:\n{processed_text}\n")
    print(f"Statistics:")
    print(f"  Original tokens: {stats['original_tokens']}")
    print(f"  Processed tokens: {stats['preprocessed_tokens']}")
    print(f"  Reduction: {stats['reduction_ratio']*100:.1f}%")
    
    print("\n" + "="*80 + "\n")


def example_4_resume_jd_matching():
    """Example 4: Match resume to job description."""
    print("="*80)
    print("EXAMPLE 4: Resume-JD Matching")
    print("="*80 + "\n")
    
    # Sample JD
    job_description = """
    Senior Machine Learning Engineer
    
    Requirements:
    - 5+ years of experience in Python programming
    - Strong expertise in Machine Learning and Deep Learning
    - Experience with TensorFlow, PyTorch, or Scikit-learn
    - Knowledge of MLOps and model deployment
    - Proficiency with Docker and Kubernetes
    - Experience with cloud platforms (AWS, Azure, or GCP)
    - Strong communication and team collaboration skills
    
    Nice to have:
    - Experience with NLP
    - Knowledge of React.js or similar frontend frameworks
    - Familiarity with CI/CD pipelines
    """
    
    # Sample resume (as text)
    resume_text = """
    Jane Smith
    Machine Learning Engineer
    
    SKILLS:
    Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, Scikit-learn
    Docker, Kubernetes, AWS, Git, Jenkins
    NLP, Computer Vision, Data Analysis
    Team Leadership, Communication
    
    EXPERIENCE:
    - Developed ML models using TensorFlow and PyTorch
    - Deployed models to production using Docker and Kubernetes on AWS
    - Led team of 5 ML engineers
    - Built NLP pipelines for text classification
    """
    
    # For this example, we'll extract skills from both
    extractor = SkillExtractor()
    
    jd_skills = extractor.extract(job_description)
    resume_skills = extractor.extract(resume_text)
    
    print("Job Description Skills:")
    all_jd_skills = extractor.get_all_skills_flat(jd_skills)
    print(f"  Found {len(all_jd_skills)} skills: {', '.join(all_jd_skills[:10])}")
    
    print("\nResume Skills:")
    all_resume_skills = extractor.get_all_skills_flat(resume_skills)
    print(f"  Found {len(all_resume_skills)} skills: {', '.join(all_resume_skills[:10])}")
    
    # Find matches
    matched = set(s.lower() for s in all_jd_skills) & set(s.lower() for s in all_resume_skills)
    missing = set(s.lower() for s in all_jd_skills) - set(s.lower() for s in all_resume_skills)
    
    match_percentage = len(matched) / len(all_jd_skills) * 100 if all_jd_skills else 0
    
    print(f"\nMatch Analysis:")
    print(f"  Match Score: {match_percentage:.1f}%")
    print(f"  Matched Skills ({len(matched)}): {', '.join(list(matched)[:10])}")
    print(f"  Missing Skills ({len(missing)}): {', '.join(list(missing)[:10])}")
    
    print("\n" + "="*80 + "\n")


def example_5_batch_processing():
    """Example 5: Batch process multiple resumes."""
    print("="*80)
    print("EXAMPLE 5: Batch Resume Processing")
    print("="*80 + "\n")
    
    # Check for resume files
    resume_dir = Path("data/resumes")
    resume_files = list(resume_dir.glob("*.pdf")) + list(resume_dir.glob("*.docx"))
    
    if not resume_files:
        print("⚠️  No resume files found in data/resumes/")
        print("   Please add some sample resumes to test this example.")
        print("\n" + "="*80 + "\n")
        return
    
    print(f"Found {len(resume_files)} resume files")
    
    # Sample JD
    job_description = """
    We are looking for a Python developer with experience in
    Machine Learning, TensorFlow, and Django. Knowledge of Docker
    and AWS is a plus.
    """
    
    # Initialize matcher
    matcher = ResumeJDMatcher()
    
    print("Processing resumes...\n")
    
    # Match resumes
    results_df = matcher.match_resumes_to_jd(
        [str(f) for f in resume_files],
        job_description
    )
    
    # Display results
    print("\nResults:")
    print(results_df[['rank', 'resume_file', 'match_percentage', 
                     'matched_skills_count']].to_string(index=False))
    
    print("\n" + "="*80 + "\n")


def main():
    """Run all examples."""
    print("\n🚀 Resume Skill Recognition System - Usage Examples\n")
    
    # Run examples
    example_1_basic_skill_extraction()
    
    example_2_resume_file_processing()
    
    example_3_text_preprocessing()
    
    example_4_resume_jd_matching()
    
    example_5_batch_processing()
    
    print("✓ All examples completed!\n")


if __name__ == "__main__":
    main()
