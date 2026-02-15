"""
Structured interview question generator.
Generates questions by job role, experience level, skills, and round type.
Runs locally; no cloud APIs.
"""

import json
import random
from typing import List, Literal

Difficulty = Literal["Easy", "Medium", "Hard"]
Category = Literal["Technical", "HR", "Aptitude"]
AnswerType = Literal["Conceptual", "Code", "Behavioral"]


# Round type → category and answer type mapping
ROUND_CONFIG = {
    "Technical": {"category": "Technical", "answer_types": ["Conceptual", "Code"]},
    "HR": {"category": "HR", "answer_types": ["Behavioral"]},
    "Aptitude": {"category": "Aptitude", "answer_types": ["Conceptual"]},
}

# Experience level → difficulty bias (more Hard for senior)
LEVEL_DIFFICULTY_BIAS = {
    "Fresher": [0.5, 0.35, 0.15],   # Easy, Medium, Hard
    "Junior": [0.3, 0.5, 0.2],
    "Mid": [0.2, 0.5, 0.3],
    "Senior": [0.1, 0.4, 0.5],
    "Lead": [0.05, 0.35, 0.6],
}


def _technical_questions(job_role: str, skills: List[str]) -> List[dict]:
    skill = skills[0] if skills else job_role
    return [
        {"q": f"Explain a core concept in {skill} that you use often in your day-to-day work as a {job_role}.", "d": "Easy", "a": "Conceptual"},
        {"q": f"How would you debug a production issue where {skill} is involved and logs are limited?", "d": "Medium", "a": "Conceptual"},
        {"q": f"Describe a scenario where you had to optimize performance in a system using {skill}. What was the bottleneck and your approach?", "d": "Medium", "a": "Conceptual"},
        {"q": f"Design an approach (high-level steps or pseudocode) for a real-world problem in {job_role}: scaling a critical service that uses {skill}.", "d": "Hard", "a": "Code"},
        {"q": f"You are given a failing integration between two systems; one uses {skill}. How do you systematically identify the root cause and fix it?", "d": "Hard", "a": "Conceptual"},
        {"q": f"What is the difference between two key concepts or tools in {skill}? When would you choose one over the other?", "d": "Easy", "a": "Conceptual"},
        {"q": f"Write or outline code (in your preferred language) to solve a typical {job_role} task involving {skill}. Explain your approach.", "d": "Medium", "a": "Code"},
        {"q": f"How would you ensure reliability and maintainability when building a new feature with {skill} for a {job_role} role?", "d": "Hard", "a": "Conceptual"},
    ]


def _hr_questions(job_role: str, _skills: List[str]) -> List[dict]:
    return [
        {"q": f"Why do you want to work as a {job_role} at this stage of your career?", "d": "Easy", "a": "Behavioral"},
        {"q": "Tell me about a time you disagreed with a teammate or manager. How did you handle it and what was the outcome?", "d": "Medium", "a": "Behavioral"},
        {"q": "Describe a situation where you had to meet a tight deadline. How did you prioritize and what did you sacrifice, if anything?", "d": "Medium", "a": "Behavioral"},
        {"q": "Give an example of when you failed at a task. What did you learn and how did you apply it later?", "d": "Hard", "a": "Behavioral"},
        {"q": "How do you stay updated with changes in your field, and how do you balance learning with delivery?", "d": "Easy", "a": "Behavioral"},
        {"q": "Describe a time you had to convince others to adopt your technical or process idea. What was the resistance and how did you overcome it?", "d": "Hard", "a": "Behavioral"},
        {"q": f"What motivates you in a {job_role} role, and what kind of work environment do you perform best in?", "d": "Easy", "a": "Behavioral"},
        {"q": "Tell me about a situation where you had to work with someone difficult. How did you build a working relationship?", "d": "Medium", "a": "Behavioral"},
    ]


def _aptitude_questions(job_role: str, _skills: List[str]) -> List[dict]:
    return [
        {"q": "A sequence follows a pattern: 2, 6, 12, 20, 30, … What is the next number and what is the rule?", "d": "Easy", "a": "Conceptual"},
        {"q": "If 3 workers complete a task in 8 days, how would you reason about the time needed if 6 workers are assigned (assuming same productivity)? Explain your logic.", "d": "Easy", "a": "Conceptual"},
        {"q": "You have 8 identical-looking balls; one is heavier. Using a balance scale only twice, how do you identify the heavier ball? Describe your approach.", "d": "Medium", "a": "Conceptual"},
        {"q": "A project has 4 tasks with dependencies: A and B can start immediately; C needs A; D needs B and C. What is the minimum time to finish if A=2, B=3, C=1, D=2 days? Explain your reasoning.", "d": "Medium", "a": "Conceptual"},
        {"q": "Design a logical approach to allocate limited resources across 3 projects with different priorities and deadlines. What criteria would you use and in what order?", "d": "Hard", "a": "Conceptual"},
        {"q": "In a group of 50 people, each person shakes hands with a subset of others. What can you conclude about the number of people who shook an odd number of hands? Explain.", "d": "Hard", "a": "Conceptual"},
        {"q": "A pipe fills a tank in 6 hours; a leak empties it in 10 hours. If both run together, how long to fill the tank? Walk through your reasoning.", "d": "Medium", "a": "Conceptual"},
        {"q": "You have 24 hours to complete 5 tasks with different durations and deadlines. How do you decide the order and what to drop if needed? Describe your decision framework.", "d": "Hard", "a": "Conceptual"},
    ]


def _get_pool(round_type: str, job_role: str, skills: List[str]) -> List[dict]:
    round_type_norm = round_type.strip().lower().replace(" ", "")
    if round_type_norm in ("technical", "tech"):
        return _technical_questions(job_role, skills)
    if round_type_norm in ("hr", "behavioral", "humanresources"):
        return _hr_questions(job_role, skills)
    if round_type_norm in ("aptitude", "logical", "reasoning"):
        return _aptitude_questions(job_role, skills)
    return _technical_questions(job_role, skills)


def _parse_skills(skills: str | List[str]) -> List[str]:
    if isinstance(skills, list):
        return [str(s).strip() for s in skills if s]
    if isinstance(skills, str):
        return [s.strip() for s in skills.replace(",", " ").split() if s.strip()]
    return []


def _progressive_difficulties(count: int, experience_level: str) -> List[str]:
    level = experience_level.strip() or "Mid"
    level_key = next((k for k in LEVEL_DIFFICULTY_BIAS if k.lower() == level.lower()), "Mid")
    weights = LEVEL_DIFFICULTY_BIAS[level_key]
    order = ["Easy", "Medium", "Hard"]
    result = []
    for i in range(count):
        r = random.random()
        if r < weights[0]:
            result.append("Easy")
        elif r < weights[0] + weights[1]:
            result.append("Medium")
        else:
            result.append("Hard")
    result.sort(key=lambda x: order.index(x))
    return result


def generate_questions(
    job_role: str,
    experience_level: str,
    skills: str | List[str],
    round_type: str,
    count: int = 5,
    seed: int | None = None,
) -> dict:
    """
    Generate structured interview questions.

    Args:
        job_role: e.g. "Python Developer", "Data Engineer".
        experience_level: Fresher, Junior, Mid, Senior, Lead.
        skills: Comma-separated string or list, e.g. "Python, SQL, AWS".
        round_type: Technical, HR, or Aptitude.
        count: Number of questions (default 5).
        seed: Optional RNG seed for reproducibility.

    Returns:
        Dict with "questions" list; each item has question, difficulty, category, expected_answer_type.
    """
    if seed is not None:
        random.seed(seed)
    job_role = (job_role or "Software Engineer").strip()
    experience_level = (experience_level or "Mid").strip()
    skills_list = _parse_skills(skills)
    round_type = round_type or "Technical"

    pool = _get_pool(round_type, job_role, skills_list)
    config = ROUND_CONFIG.get(
        "Technical" if "technical" in round_type.lower() else "HR" if "hr" in round_type.lower() else "Aptitude",
        ROUND_CONFIG["Technical"],
    )
    category = config["category"]
    answer_types = config["answer_types"]

    difficulties = _progressive_difficulties(count, experience_level)
    chosen = []
    used = set()
    for d in difficulties:
        candidates = [i for i, p in enumerate(pool) if p["d"] == d and i not in used]
        if not candidates:
            candidates = [i for i in range(len(pool)) if i not in used]
        if not candidates:
            break
        idx = random.choice(candidates)
        used.add(idx)
        entry = pool[idx]
        q_text = entry["q"].format(job_role=job_role, skill=skills_list[0] if skills_list else job_role) if "{job_role}" in entry["q"] or "{skill}" in entry["q"] else entry["q"]
        at = entry["a"] if entry["a"] in answer_types else answer_types[0]
        chosen.append({
            "question": q_text,
            "difficulty": entry["d"],
            "category": category,
            "expected_answer_type": at,
        })

    while len(chosen) < count and len(pool) > len(chosen):
        idx = random.randint(0, len(pool) - 1)
        entry = pool[idx]
        q_text = entry["q"]
        if "{job_role}" in q_text:
            q_text = q_text.format(job_role=job_role, skill=skills_list[0] if skills_list else job_role)
        if not any(c["question"] == q_text for c in chosen):
            chosen.append({
                "question": q_text,
                "difficulty": entry["d"],
                "category": category,
                "expected_answer_type": entry["a"],
            })

    return {"questions": chosen}


def generate_questions_json(
    job_role: str,
    experience_level: str,
    skills: str | List[str],
    round_type: str,
    count: int = 5,
    seed: int | None = None,
    indent: int = 2,
) -> str:
    """Same as generate_questions but returns a JSON string."""
    return json.dumps(generate_questions(job_role, experience_level, skills, round_type, count, seed), indent=indent)


def generate_questions_openai(
    role: str,
    experience: str,
    skills: str,
    round_type: str,
    *,
    api_key: str | None = None,
    model: str = "gpt-4",
    temperature: float = 0.7,
) -> dict:
    """
    Generate 5 interview questions using OpenAI Chat Completions API.

    Requires: pip install openai and OPENAI_API_KEY set (or pass api_key).

    Returns:
        {"questions": [{"question", "difficulty", "category", "expected_answer_type?"}, ...]}
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package is required for generate_questions_openai. Install with: pip install openai")

    client = OpenAI(api_key=api_key)  # uses os.environ["OPENAI_API_KEY"] if api_key is None

    prompt = f"""
You are a professional technical interviewer.

Generate exactly 5 interview questions.

Details:
- Job Role: {role}
- Experience Level: {experience}
- Skills: {skills}
- Round Type: {round_type}

Rules:
1. Return ONLY valid JSON.
2. Do NOT include markdown.
3. Do NOT include explanations.
4. Do NOT include text before or after JSON.
5. Follow this exact format:

{{
  "questions": [
    {{
      "question": "text",
      "difficulty": "Easy/Medium/Hard",
      "category": "Technical/HR/Aptitude",
      "expected_answer_type": "Conceptual/Code/Behavioral"
    }}
  ]
}}

Ensure JSON is properly formatted.
"""

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content or ""

    # Clean markdown if present
    content = content.replace("```json", "").replace("```", "").strip()

    data = json.loads(content)
    questions = data.get("questions", [])

    # Ensure each item has expected_answer_type for consistency with local generator
    category_to_answer_type = {"Technical": "Conceptual", "HR": "Behavioral", "Aptitude": "Conceptual"}
    for q in questions:
        if "expected_answer_type" not in q:
            q["expected_answer_type"] = category_to_answer_type.get(
                q.get("category", "Technical"), "Conceptual"
            )

    return {"questions": questions}


if __name__ == "__main__":
    import sys
    job_role = sys.argv[1] if len(sys.argv) > 1 else "Software Engineer"
    experience_level = sys.argv[2] if len(sys.argv) > 2 else "Mid"
    skills = sys.argv[3] if len(sys.argv) > 3 else "Python, SQL"
    round_type = sys.argv[4] if len(sys.argv) > 4 else "Technical"
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    out = generate_questions_json(job_role, experience_level, skills, round_type, count=n, seed=42)
    print(out)
