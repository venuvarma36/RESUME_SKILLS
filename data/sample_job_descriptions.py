"""
Sample job descriptions for testing the Resume Skill Recognition System
"""

SAMPLE_JD_ML_ENGINEER = """
Senior Machine Learning Engineer

Company: TechCorp AI
Location: Remote
Type: Full-time

About the Role:
We are seeking an experienced Machine Learning Engineer to join our AI research team. 
You will be responsible for developing, deploying, and optimizing ML models at scale.

Required Skills:
• 5+ years of experience in Python programming
• Strong expertise in Machine Learning and Deep Learning
• Proficiency with TensorFlow, PyTorch, or Scikit-learn
• Experience with neural networks (CNN, RNN, LSTM, Transformers)
• Knowledge of MLOps and model deployment
• Experience with Docker and Kubernetes
• Proficiency with cloud platforms (AWS, Azure, or GCP)
• Strong understanding of data structures and algorithms
• Experience with Git version control

Preferred Skills:
• Experience with Natural Language Processing (NLP)
• Knowledge of Computer Vision
• Familiarity with Spark and distributed computing
• Experience with CI/CD pipelines (Jenkins, GitHub Actions)
• Frontend development experience (React.js)
• Publications in ML conferences (NeurIPS, ICML, CVPR)

Soft Skills:
• Strong communication and presentation skills
• Team collaboration and leadership abilities
• Problem-solving and analytical thinking
• Self-motivated and proactive
• Ability to work in a fast-paced environment

What We Offer:
• Competitive salary and equity
• Flexible work arrangements
• Professional development opportunities
• Cutting-edge technology stack
"""

SAMPLE_JD_FULL_STACK = """
Full Stack Developer

Company: WebTech Solutions
Location: San Francisco, CA / Hybrid
Type: Full-time

Job Description:
We're looking for a talented Full Stack Developer to join our dynamic engineering team.
You'll work on building scalable web applications using modern technologies.

Technical Requirements:
• 3+ years of professional software development experience
• Strong proficiency in JavaScript/TypeScript
• Frontend: React.js, Vue.js, or Angular
• Backend: Node.js, Express.js, or Django
• Database: PostgreSQL, MongoDB, or MySQL
• RESTful API development
• Experience with GraphQL (plus)
• Git version control
• Understanding of web security best practices

Additional Skills:
• Cloud platforms: AWS, Azure, or GCP
• Docker and containerization
• CI/CD pipelines
• Testing frameworks (Jest, Mocha, Pytest)
• Agile/Scrum methodology
• Responsive design and CSS frameworks (Bootstrap, Tailwind CSS)

Nice to Have:
• Mobile development (React Native)
• Redis or other caching solutions
• Microservices architecture
• WebSockets and real-time applications
• Performance optimization

Personal Qualities:
• Excellent problem-solving skills
• Strong communication abilities
• Team player with leadership potential
• Attention to detail
• Continuous learner
• Adaptable to changing requirements

Perks:
• Competitive salary
• Health insurance
• Stock options
• Unlimited PTO
• Learning budget
"""

SAMPLE_JD_DATA_SCIENTIST = """
Data Scientist

Company: DataDriven Analytics
Location: Boston, MA / Remote
Type: Full-time

Overview:
Join our data science team to build predictive models and derive actionable insights
from large-scale datasets. You'll work closely with business stakeholders and engineers.

Required Qualifications:
• Master's or PhD in Computer Science, Statistics, or related field
• 4+ years of experience in data science or analytics
• Expert-level Python programming (Pandas, NumPy, SciPy)
• Strong knowledge of statistical analysis and hypothesis testing
• Machine Learning expertise (Scikit-learn, XGBoost, LightGBM)
• SQL and database querying
• Data visualization (Matplotlib, Seaborn, Plotly, Tableau)
• Experience with A/B testing and experimentation
• Jupyter notebooks and reproducible research

Preferred Qualifications:
• Deep Learning experience (TensorFlow, PyTorch)
• Big Data technologies (Spark, Hadoop, Hive)
• Natural Language Processing or Computer Vision
• Cloud platforms (AWS, GCP, Azure)
• MLOps and model deployment
• Feature engineering and selection
• Time series analysis
• R programming language

Soft Skills:
• Strong analytical and critical thinking
• Excellent communication and storytelling with data
• Collaboration with cross-functional teams
• Business acumen
• Project management
• Mentoring and knowledge sharing

Tools & Technologies:
• Python, R, SQL
• Git, GitHub
• Docker, Kubernetes
• Airflow or similar workflow orchestration
• Power BI or Looker
• Jira, Confluence

Benefits:
• Competitive compensation
• Remote flexibility
• Professional development
• Conference attendance
• Latest hardware and software
"""

# Export all sample JDs
SAMPLE_JDS = {
    'ml_engineer': SAMPLE_JD_ML_ENGINEER,
    'full_stack': SAMPLE_JD_FULL_STACK,
    'data_scientist': SAMPLE_JD_DATA_SCIENTIST
}


def get_sample_jd(role: str = 'ml_engineer') -> str:
    """
    Get a sample job description.
    
    Args:
        role: Role type ('ml_engineer', 'full_stack', 'data_scientist')
        
    Returns:
        Job description text
    """
    return SAMPLE_JDS.get(role, SAMPLE_JD_ML_ENGINEER)


if __name__ == "__main__":
    # Print all sample JDs
    for role, jd in SAMPLE_JDS.items():
        print(f"\n{'='*80}")
        print(f"SAMPLE JD: {role.upper().replace('_', ' ')}")
        print(f"{'='*80}\n")
        print(jd)
