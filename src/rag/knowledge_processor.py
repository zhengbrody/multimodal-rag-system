"""
Knowledge Base Processor for Personal RAG System

Converts structured personal data into searchable documents
"""

import json
from pathlib import Path
from typing import List, Dict, Any


class Document:
    """Simple document class for RAG"""
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(content={self.content[:50]}..., metadata={self.metadata})"


def load_knowledge_base(path: str) -> Dict[str, Any]:
    """Load knowledge base from JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_personal_info(data: Dict[str, Any]) -> List[Document]:
    """Process personal information into documents"""
    info = data.get('personal_info', {})

    content = f"""
Personal Information:
Name: {info.get('name', '')}
Title: {info.get('title', '')}
Email: {info.get('email', '')}
Location: {info.get('location', '')}
Bio: {info.get('bio', '')}
"""

    return [Document(
        content=content.strip(),
        metadata={
            'type': 'personal_info',
            'category': 'about'
        }
    )]


def process_skills(data: Dict[str, Any]) -> List[Document]:
    """Process skills into documents"""
    skills = data.get('skills', [])
    documents = []

    # Create one comprehensive skills document
    skills_text = "Skills and Expertise:\n"
    for skill_group in skills:
        category = skill_group.get('category', '')
        items = ', '.join(skill_group.get('items', []))
        skills_text += f"- {category}: {items}\n"

    documents.append(Document(
        content=skills_text.strip(),
        metadata={
            'type': 'skills',
            'category': 'skills'
        }
    ))

    return documents


def process_projects(data: Dict[str, Any]) -> List[Document]:
    """Process projects into individual documents"""
    projects = data.get('projects', [])
    documents = []

    for project in projects:
        # Support both old and new field names
        highlights = project.get('highlights', project.get('achievements', []))
        year = project.get('year', project.get('period', ''))
        link = project.get('link', project.get('github', ''))

        content = f"""
Project: {project.get('name', '')}
Period: {year}
Status: {project.get('status', '')}
Category: {project.get('category', '')}
Description: {project.get('description', '')}
Problem: {project.get('problem_statement', '')}
Solution: {project.get('solution', '')}
Tech Stack: {', '.join(project.get('tech_stack', []))}
Key Achievements: {', '.join(highlights)}
Key Learnings: {project.get('key_learnings', '')}
Project Link: {link}
"""
        documents.append(Document(
            content=content.strip(),
            metadata={
                'type': 'project',
                'id': project.get('id', ''),
                'name': project.get('name', ''),
                'period': year,
                'status': project.get('status', ''),
                'category': 'projects'
            }
        ))

    return documents


def process_blog_posts(data: Dict[str, Any]) -> List[Document]:
    """Process blog posts into documents"""
    posts = data.get('blog_posts', [])
    documents = []

    for post in posts:
        content = f"""
Blog Post: {post.get('title', '')}
Published Date: {post.get('date', '')}
Summary: {post.get('summary', '')}
Tags: {', '.join(post.get('tags', []))}
Key Points: {post.get('content', '')}
"""
        documents.append(Document(
            content=content.strip(),
            metadata={
                'type': 'blog_post',
                'id': post.get('id', ''),
                'title': post.get('title', ''),
                'date': post.get('date', ''),
                'tags': post.get('tags', []),
                'category': 'blog'
            }
        ))

    return documents


def process_experience(data: Dict[str, Any]) -> List[Document]:
    """Process work experience into documents"""
    experiences = data.get('experience', [])
    documents = []

    for exp in experiences:
        achievements = '\n'.join([f"  - {a}" for a in exp.get('achievements', [])])
        tech_stack = ', '.join(exp.get('tech_stack', []))

        content = f"""
Work Experience:
Company: {exp.get('company', '')}
Role: {exp.get('role', '')}
Location: {exp.get('location', '')}
Period: {exp.get('period', '')}
Type: {exp.get('type', '')}
Description: {exp.get('description', '')}
Key Achievements:
{achievements}
Tech Stack: {tech_stack}
Key Learnings: {exp.get('key_learnings', '')}
"""
        documents.append(Document(
            content=content.strip(),
            metadata={
                'type': 'experience',
                'company': exp.get('company', ''),
                'role': exp.get('role', ''),
                'period': exp.get('period', ''),
                'category': 'experience'
            }
        ))

    return documents


def process_education(data: Dict[str, Any]) -> List[Document]:
    """Process education into documents"""
    education = data.get('education', [])
    documents = []

    edu_text = "Education Background:\n"
    for edu in education:
        edu_text += f"""
Degree: {edu.get('degree', '')}
School: {edu.get('school', '')}
Period: {edu.get('period', '')}
Focus: {edu.get('focus', '')}
"""
        if 'thesis' in edu:
            edu_text += f"Thesis: {edu.get('thesis', '')}\n"

    documents.append(Document(
        content=edu_text.strip(),
        metadata={
            'type': 'education',
            'category': 'education'
        }
    ))

    return documents


def process_faq(data: Dict[str, Any]) -> List[Document]:
    """Process FAQ into documents"""
    faqs = data.get('faq', [])
    documents = []

    for faq in faqs:
        content = f"""
Frequently Asked Question: {faq.get('question', '')}
Answer: {faq.get('answer', '')}
"""
        documents.append(Document(
            content=content.strip(),
            metadata={
                'type': 'faq',
                'question': faq.get('question', ''),
                'category': 'faq'
            }
        ))

    return documents


def process_interests_and_contact(data: Dict[str, Any]) -> List[Document]:
    """Process interests and contact info"""
    documents = []

    # Interests (support both old and new field names)
    interests = data.get('interests', data.get('interests_and_activities', []))
    if interests:
        interests_text = "Interests and Activities:\n" + '\n'.join([f"- {i}" for i in interests])
        documents.append(Document(
            content=interests_text,
            metadata={
                'type': 'interests',
                'category': 'about'
            }
        ))

    # Contact
    contact = data.get('contact', {})
    if contact:
        contact_text = f"""
Contact Information:
Email: {contact.get('email', '')}
Phone: {contact.get('phone', '')}
GitHub: {contact.get('github', '')}
LinkedIn: {contact.get('linkedin', '')}
Location: {contact.get('location', '')}
Availability: {contact.get('availability', '')}
Visa Status: {contact.get('visa_status', '')}
"""
        documents.append(Document(
            content=contact_text.strip(),
            metadata={
                'type': 'contact',
                'category': 'contact'
            }
        ))

    return documents


def process_comprehensive_qa(data: Dict[str, Any]) -> List[Document]:
    """Process comprehensive Q&A pairs into documents with smart categorization"""
    qa_data = data.get('comprehensive_qa', {})
    documents = []

    # Map QA categories to document types and categories
    category_mapping = {
        'projects_detailed': ('project', 'projects'),
        'experience_deep_dive': ('experience', 'experience'),
        'technical_skills': ('skills', 'skills'),
        'mlops_and_production': ('skills', 'skills'),
        'education': ('education', 'education'),
        'basic_info': ('personal_info', 'about'),
        'behavioral': ('faq', 'faq'),
        'company_and_role_fit': ('career_goals', 'about'),
        'system_design': ('skills', 'skills'),
        'technical_depth': ('skills', 'skills'),
        'tools_and_technologies': ('skills', 'skills'),
        'metrics_and_impact': ('achievement', 'about'),
        'miscellaneous': ('faq', 'faq')
    }

    for qa_category, qa_list in qa_data.items():
        doc_type, doc_category = category_mapping.get(qa_category, ('qa', 'faq'))

        for qa in qa_list:
            question = qa.get('q', '')
            answer = qa.get('a', '')
            keywords = qa.get('keywords', [])
            
            # Smart categorization: detect work experience/internship questions
            question_lower = question.lower()
            answer_lower = answer.lower()
            if any(kw in question_lower or kw in answer_lower for kw in ['work experience', 'internship', 'internships', 'employment', 'allianz', 'penn state', 'qingdao']):
                if 'experience' in keywords or 'internship' in keywords or 'employment' in keywords:
                    doc_category = 'experience'
                    doc_type = 'experience'

            content = f"""
Question: {question}
Answer: {answer}
Keywords: {', '.join(keywords)}
"""
            documents.append(Document(
                content=content.strip(),
                metadata={
                    'type': doc_type,
                    'category': doc_category,
                    'qa_category': qa_category,
                    'keywords': keywords,
                    'question': question
                }
            ))

    return documents


def process_keyword_mappings(data: Dict[str, Any]) -> List[Document]:
    """Process keyword mappings for quick lookups with smart categorization"""
    mappings = data.get('keyword_mappings', {})
    documents = []

    # Categorize keywords by content
    skill_keywords = {'python', 'pytorch', 'tensorflow', 'langchain', 'llm', 'rag', 'mlops',
                      'aws', 'docker', 'kubernetes', 'spark', 'kafka', 'redis', 'postgresql',
                      'fastapi', 'react', 'skills', 'languages'}
    project_keywords = {'project', 'rag project', 'recommendation', 'llm project', 'kaggle', 'medal', 'competition'}
    experience_keywords = {'allianz', 'penn state', 'research'}
    contact_keywords = {'github', 'linkedin', 'email', 'phone'}
    availability_keywords = {'internship', '2026', 'summer', 'availability', 'start date', 'duration', 'visa', 'location', 'graduation'}
    about_keywords = {'strength', 'achievement', 'metric', 'interest', 'goal', 'prefer'}

    for keyword, answer in mappings.items():
        # Determine category based on keyword
        keyword_lower = keyword.lower()
        if keyword_lower in skill_keywords:
            doc_type, doc_category = 'skills', 'skills'
        elif keyword_lower in project_keywords:
            doc_type, doc_category = 'project', 'projects'
        elif keyword_lower in experience_keywords:
            doc_type, doc_category = 'experience', 'experience'
        elif keyword_lower in contact_keywords:
            doc_type, doc_category = 'contact', 'contact'
        elif keyword_lower in availability_keywords:
            doc_type, doc_category = 'availability', 'about'
        elif keyword_lower in about_keywords:
            doc_type, doc_category = 'about', 'about'
        else:
            doc_type, doc_category = 'keyword_mapping', 'faq'

        content = f"""
Keyword: {keyword}
Quick Answer: {answer}
"""
        documents.append(Document(
            content=content.strip(),
            metadata={
                'type': doc_type,
                'category': doc_category,
                'keyword': keyword
            }
        ))

    return documents


def process_technical_achievements(data: Dict[str, Any]) -> List[Document]:
    """Process technical achievements"""
    achievements = data.get('technical_achievements', [])
    documents = []

    for area_data in achievements:
        area = area_data.get('area', '')
        achievement_list = area_data.get('achievements', [])

        content = f"""
Technical Achievements in {area}:
{chr(10).join([f"- {a}" for a in achievement_list])}
"""
        documents.append(Document(
            content=content.strip(),
            metadata={
                'type': 'achievement',
                'category': 'about',
                'area': area
            }
        ))

    return documents


def process_career_goals(data: Dict[str, Any]) -> List[Document]:
    """Process career goals and interests"""
    goals = data.get('career_goals', {})
    if not goals:
        return []

    content = f"""
Career Goals and Interests:
Short-term Goal: {goals.get('short_term', '')}
Long-term Goal: {goals.get('long_term', '')}
Ideal Roles: {', '.join(goals.get('ideal_roles', []))}
Company Interests: {', '.join(goals.get('company_interests', []))}
"""
    return [Document(
        content=content.strip(),
        metadata={
            'type': 'career_goals',
            'category': 'about'
        }
    )]


def process_strengths(data: Dict[str, Any]) -> List[Document]:
    """Process strengths"""
    strengths = data.get('strengths', [])
    if not strengths:
        return []

    content = "Key Strengths:\n" + '\n'.join([f"- {s}" for s in strengths])
    return [Document(
        content=content,
        metadata={
            'type': 'strengths',
            'category': 'about'
        }
    )]


def process_research_interests(data: Dict[str, Any]) -> List[Document]:
    """Process research interests"""
    interests = data.get('research_interests', [])
    if not interests:
        return []

    content = "Research Interests:\n" + '\n'.join([f"- {i}" for i in interests])
    return [Document(
        content=content,
        metadata={
            'type': 'research_interests',
            'category': 'about'
        }
    )]


def process_blog_topics(data: Dict[str, Any]) -> List[Document]:
    """Process blog topics (alternative to blog_posts)"""
    topics = data.get('blog_topics', [])
    documents = []

    for topic in topics:
        content = f"""
Blog Topic: {topic.get('title', '')}
Summary: {topic.get('summary', '')}
Tags: {', '.join(topic.get('tags', []))}
"""
        documents.append(Document(
            content=content.strip(),
            metadata={
                'type': 'blog_topic',
                'category': 'blog',
                'title': topic.get('title', ''),
                'tags': topic.get('tags', [])
            }
        ))

    return documents


def process_availability(data: Dict[str, Any]) -> List[Document]:
    """Process availability information"""
    avail = data.get('availability', {})
    if not avail:
        return []

    content = f"""
Availability for Internship:
Period: {avail.get('internship_period', '')}
Start Date: {avail.get('start_date', '')}
Duration: {avail.get('duration', '')}
Location Preference: {avail.get('location_preference', '')}
Work Authorization: {avail.get('work_authorization', '')}
Current Status: {avail.get('currently', '')}
"""
    return [Document(
        content=content.strip(),
        metadata={
            'type': 'availability',
            'category': 'about'
        }
    )]


def process_detailed_info(data: Dict[str, Any]) -> List[Document]:
    """Process detailed_info section with education, skills, and career info"""
    detailed = data.get('detailed_info', {})
    documents = []

    # Process education_extended
    edu_extended = detailed.get('education_extended', {})
    for school_key, school_data in edu_extended.items():
        coursework = ', '.join(school_data.get('relevant_coursework', []))
        focus = ', '.join(school_data.get('focus_areas', []))
        achievements = school_data.get('achievements', [])
        achievements_text = '\n'.join([f"  - {a}" for a in achievements]) if achievements else ''
        research = ', '.join(school_data.get('research_interests', []))

        content = f"""
Education at {school_key.upper()}:
Program: {school_data.get('program', '')}
Graduation: {school_data.get('expected_graduation', school_data.get('graduation', ''))}
Status: {school_data.get('status', '')}
Focus Areas: {focus}
Relevant Coursework: {coursework}
{f'Research Interests: {research}' if research else ''}
{f'Achievements:\\n{achievements_text}' if achievements_text else ''}
{f'GPA Note: {school_data.get("gpa_note", "")}' if school_data.get("gpa_note") else ''}
"""
        documents.append(Document(
            content=content.strip(),
            metadata={
                'type': 'education',
                'category': 'education',
                'school': school_key
            }
        ))

    # Process technical_proficiency as structured skills
    tech_prof = detailed.get('technical_proficiency', {})

    # Python proficiency
    if 'python' in tech_prof:
        py = tech_prof['python']
        content = f"""
Python Proficiency:
Level: {py.get('level', '')}
Years of Experience: {py.get('years', '')}
Libraries: {', '.join(py.get('libraries', []))}
Use Cases: {', '.join(py.get('use_cases', []))}
Notes: {py.get('projects', '')}
"""
        documents.append(Document(
            content=content.strip(),
            metadata={'type': 'skills', 'category': 'skills', 'skill': 'python'}
        ))

    # ML frameworks
    ml_info = tech_prof.get('machine_learning', {})
    frameworks = ml_info.get('frameworks', {})
    for fw_name, fw_data in frameworks.items():
        if isinstance(fw_data, dict):
            experience = ', '.join(fw_data.get('experience', []))
            content = f"""
{fw_name.upper()} Framework:
Level: {fw_data.get('level', '')}
Experience: {experience}
{f'Note: {fw_data.get("note", "")}' if fw_data.get("note") else ''}
"""
            documents.append(Document(
                content=content.strip(),
                metadata={'type': 'skills', 'category': 'skills', 'skill': fw_name}
            ))

    # NLP Tools
    nlp_tools = ml_info.get('nlp_tools', {})
    if nlp_tools:
        nlp_text = "NLP Tools Proficiency:\n"
        for tool, desc in nlp_tools.items():
            nlp_text += f"- {tool}: {desc}\n"
        documents.append(Document(
            content=nlp_text.strip(),
            metadata={'type': 'skills', 'category': 'skills', 'skill': 'nlp'}
        ))

    # MLOps skills
    mlops = tech_prof.get('mlops', {})
    if mlops:
        content = f"""
MLOps Skills:
Experiment Tracking: {', '.join(mlops.get('experiment_tracking', []))}
Model Versioning: {', '.join(mlops.get('model_versioning', []))}
Deployment Tools: {', '.join(mlops.get('deployment', []))}
Monitoring: {', '.join(mlops.get('monitoring', []))}
CI/CD: {', '.join(mlops.get('ci_cd', []))}
Experience Level: {mlops.get('experience_level', '')}
"""
        documents.append(Document(
            content=content.strip(),
            metadata={'type': 'skills', 'category': 'skills', 'skill': 'mlops'}
        ))

    # Big Data tools
    big_data = tech_prof.get('big_data', {})
    for tool_name, tool_data in big_data.items():
        if isinstance(tool_data, dict):
            # Handle both list and string formats for components/experience
            comp_data = tool_data.get('components', tool_data.get('experience', []))
            if isinstance(comp_data, list):
                components = ', '.join(comp_data)
            else:
                components = str(comp_data)
            content = f"""
{tool_name.upper()} Big Data Tool:
Level: {tool_data.get('level', '')}
Components/Experience: {components}
{f'Scale: {tool_data.get("scale", "")}' if tool_data.get("scale") else ''}
{f'Use Cases: {", ".join(tool_data.get("use_cases", []))}' if tool_data.get("use_cases") else ''}
"""
            documents.append(Document(
                content=content.strip(),
                metadata={'type': 'skills', 'category': 'skills', 'skill': tool_name}
            ))

    # Databases
    dbs = tech_prof.get('databases', {})
    for db_name, db_data in dbs.items():
        if isinstance(db_data, dict) and 'level' in db_data:
            use_cases = ', '.join(db_data.get('use_cases', db_data.get('experience', [])))
            content = f"""
{db_name.upper()} Database:
Level: {db_data.get('level', '')}
Use Cases: {use_cases}
{f'Achievement: {db_data.get("achievement", "")}' if db_data.get("achievement") else ''}
"""
            documents.append(Document(
                content=content.strip(),
                metadata={'type': 'skills', 'category': 'skills', 'skill': db_name}
            ))
        elif isinstance(db_data, dict):
            # Vector databases
            vdb_text = f"{db_name.upper()} Databases:\n"
            for vdb, desc in db_data.items():
                vdb_text += f"- {vdb}: {desc}\n"
            documents.append(Document(
                content=vdb_text.strip(),
                metadata={'type': 'skills', 'category': 'skills', 'skill': db_name}
            ))

    # Backend frameworks
    backend = tech_prof.get('backend', {})
    for fw_name, fw_data in backend.items():
        if isinstance(fw_data, dict):
            features = ', '.join(fw_data.get('features', fw_data.get('use_cases', [])))
            projects = ', '.join(fw_data.get('projects', []))
            content = f"""
{fw_name.upper()} Backend Framework:
Level: {fw_data.get('level', '')}
Features/Use Cases: {features}
{f'Projects: {projects}' if projects else ''}
{f'Scale: {fw_data.get("scale", "")}' if fw_data.get("scale") else ''}
"""
            documents.append(Document(
                content=content.strip(),
                metadata={'type': 'skills', 'category': 'skills', 'skill': fw_name}
            ))

    # Cloud (AWS)
    cloud = tech_prof.get('cloud', {})
    aws = cloud.get('aws', {})
    if aws:
        services = aws.get('services', {})
        services_text = ""
        for category, service_list in services.items():
            services_text += f"\n  {category}: {', '.join(service_list)}"
        content = f"""
AWS Cloud Experience:
Level: {aws.get('level', '')}
Services: {services_text}
Projects: {aws.get('projects', '')}
Certifications: {aws.get('certifications', '')}
"""
        documents.append(Document(
            content=content.strip(),
            metadata={'type': 'skills', 'category': 'skills', 'skill': 'aws'}
        ))

    # Soft skills
    soft_skills = detailed.get('soft_skills', {})
    if soft_skills:
        soft_text = "Soft Skills:\n"
        for skill, desc in soft_skills.items():
            soft_text += f"- {skill.replace('_', ' ').title()}: {desc}\n"
        documents.append(Document(
            content=soft_text.strip(),
            metadata={'type': 'soft_skills', 'category': 'about'}
        ))

    # Career timeline
    timeline = detailed.get('career_timeline', {})
    if timeline:
        timeline_text = "Career Timeline:\n"
        for period, desc in timeline.items():
            timeline_text += f"- {period}: {desc}\n"
        documents.append(Document(
            content=timeline_text.strip(),
            metadata={'type': 'career_timeline', 'category': 'about'}
        ))

    # Ideal work environment
    ideal_env = detailed.get('ideal_work_environment', {})
    if ideal_env:
        env_text = "Ideal Work Environment:\n"
        for aspect, desc in ideal_env.items():
            env_text += f"- {aspect.replace('_', ' ').title()}: {desc}\n"
        documents.append(Document(
            content=env_text.strip(),
            metadata={'type': 'preferences', 'category': 'about'}
        ))

    return documents


def build_knowledge_base(knowledge_path: str) -> List[Document]:
    """
    Build complete knowledge base from JSON file

    Returns list of Document objects ready for embedding
    """
    data = load_knowledge_base(knowledge_path)

    documents = []

    # Process all sections (core)
    documents.extend(process_personal_info(data))
    documents.extend(process_skills(data))
    documents.extend(process_projects(data))
    documents.extend(process_blog_posts(data))
    documents.extend(process_experience(data))
    documents.extend(process_education(data))
    documents.extend(process_faq(data))
    documents.extend(process_interests_and_contact(data))

    # Process new sections (extended)
    documents.extend(process_comprehensive_qa(data))
    documents.extend(process_keyword_mappings(data))
    documents.extend(process_technical_achievements(data))
    documents.extend(process_career_goals(data))
    documents.extend(process_strengths(data))
    documents.extend(process_research_interests(data))
    documents.extend(process_blog_topics(data))
    documents.extend(process_availability(data))
    documents.extend(process_detailed_info(data))

    print(f"Built knowledge base with {len(documents)} documents")
    return documents


if __name__ == "__main__":
    # Test the processor
    kb_path = Path(__file__).parent.parent.parent / "data" / "raw" / "knowledge_base.json"
    documents = build_knowledge_base(str(kb_path))

    print("\nSample documents:")
    for i, doc in enumerate(documents[:3]):
        print(f"\n--- Document {i+1} ---")
        print(f"Type: {doc.metadata.get('type')}")
        print(f"Content preview: {doc.content[:200]}...")
