"""
Quiz Manager Module for StudyMate.
Handles automated Multiple-Choice Question (MCQ) generation from PDF study materials,
strict JSON schema parsing, user answer evaluation, grade calculation, and feedback.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from core import prompts
from core import gemini_client
import config


class QuizParsingError(Exception):
    """Raised when generated quiz JSON cannot be parsed or validated."""
    pass


def clean_json_string(raw_str: str) -> str:
    """
    Cleans raw LLM response by removing markdown fences, leading/trailing notes,
    and ensuring a valid JSON block.
    """
    s = raw_str.strip()
    
    # Strip markdown code blocks like ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if match:
        s = match.group(1).strip()
    else:
        # If text begins before '[' or '{', strip leading characters
        start_bracket = s.find("[")
        end_bracket = s.rfind("]")
        if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
            s = s[start_bracket : end_bracket + 1]

    # Fix potential trailing commas before closing braces
    s = re.sub(r",\s*([\]}])", r"\1", s)
    return s


def validate_and_normalize_quiz(data: Any, expected_count: int) -> List[Dict[str, Any]]:
    """
    Validates the structure of the parsed JSON quiz objects and normalizes formats.
    """
    if not isinstance(data, list):
        if isinstance(data, dict) and "questions" in data and isinstance(data["questions"], list):
            data = data["questions"]
        else:
            raise QuizParsingError(f"Expected a JSON list of questions, but received {type(data).__name__}.")

    if len(data) == 0:
        raise QuizParsingError("Generated quiz contains 0 questions.")

    validated = []
    for idx, item in enumerate(data, 1):
        if not isinstance(item, dict):
            continue

        question_text = item.get("question", "").strip()
        if not question_text:
            continue

        raw_options = item.get("options", {})
        norm_options = {}

        if isinstance(raw_options, dict):
            for opt_key in ["A", "B", "C", "D"]:
                val = str(raw_options.get(opt_key, "")).strip()
                if not val:
                    # Check lowercase
                    val = str(raw_options.get(opt_key.lower(), "")).strip()
                norm_options[opt_key] = val or f"Option {opt_key}"
        elif isinstance(raw_options, list):
            letters = ["A", "B", "C", "D"]
            for l_idx, opt_val in enumerate(raw_options[:4]):
                norm_options[letters[l_idx]] = str(opt_val).strip()

        # Ensure all 4 options exist
        for opt_key in ["A", "B", "C", "D"]:
            if opt_key not in norm_options or not norm_options[opt_key]:
                norm_options[opt_key] = f"Option {opt_key}"

        correct_ans = str(item.get("correct_answer", "A")).strip().upper()
        # If correct_answer was given as full option text instead of letter
        if correct_ans not in ["A", "B", "C", "D"]:
            matched = False
            for k, v in norm_options.items():
                if correct_ans.lower() == v.lower():
                    correct_ans = k
                    matched = True
                    break
            if not matched:
                # Default to A if unclear
                correct_ans = "A"

        explanation = item.get("explanation", "").strip() or "No additional explanation provided."
        topic = item.get("reference_topic", "").strip() or "Course Material"

        validated.append({
            "id": idx,
            "question": question_text,
            "options": norm_options,
            "correct_answer": correct_ans,
            "explanation": explanation,
            "reference_topic": topic
        })

    if not validated:
        raise QuizParsingError("None of the generated questions met the required schema.")

    return validated


def generate_quiz(
    pages: List[Dict[str, Any]],
    num_questions: int = 5,
    difficulty: str = "Medium",
    topic: str = "All Chapters",
    api_key: Optional[str] = None,
    model: str = config.DEFAULT_LLM_MODEL
) -> List[Dict[str, Any]]:
    """
    Generates a high-quality Multiple-Choice Quiz from PDF text.
    
    Args:
        pages: List of extracted page dicts.
        num_questions: Number of questions to create (default 5).
        difficulty: 'Easy', 'Medium', or 'Hard'.
        topic: Focus topic or 'All Chapters'.
        api_key: Gemini API key.
        model: Model name.
        
    Returns:
        List of validated question dicts.
    """
    if not pages:
        raise ValueError("Cannot generate quiz from an empty document. Please upload a PDF first.")

    # Assemble document sample
    full_text = "\n\n".join([f"[Page {p['page_number']}]\n{p['text']}" for p in pages if p.get("text")])
    if len(full_text) > 30000:
        full_text = full_text[:30000] + "\n...[Content truncated for quiz context]..."

    prompt = prompts.QUIZ_GENERATION_PROMPT_TEMPLATE.format(
        num_questions=num_questions,
        difficulty=difficulty,
        topic=topic,
        document_text=full_text
    )

    raw_response = gemini_client.generate_text(
        prompt=prompt,
        api_key=api_key,
        model=model,
        temperature=config.TEMPERATURE_QUIZ,
        response_mime_type="application/json"
    )

    clean_json = clean_json_string(raw_response)
    try:
        parsed_data = json.loads(clean_json)
    except json.JSONDecodeError as e:
        raise QuizParsingError(f"Failed to parse LLM response into JSON: {str(e)}\nRaw Response Snippet: {clean_json[:200]}")

    quiz_questions = validate_and_normalize_quiz(parsed_data, expected_count=num_questions)
    return quiz_questions


def evaluate_quiz(
    questions: List[Dict[str, Any]], 
    user_answers: Dict[int, str]
) -> Dict[str, Any]:
    """
    Evaluates submitted user answers against the question answer key.
    
    Args:
        questions: List of question dictionaries.
        user_answers: Dictionary mapping question_id (int) -> selected option letter ('A', 'B', 'C', 'D').
        
    Returns:
        Comprehensive assessment dictionary with score, grade, feedback, and question breakdown.
    """
    total = len(questions)
    if total == 0:
        return {"total_questions": 0, "score_percentage": 0.0}

    correct_count = 0
    incorrect_count = 0
    unanswered_count = 0
    breakdown = []

    for q in questions:
        q_id = q["id"]
        correct_opt = q["correct_answer"].strip().upper()
        user_opt = user_answers.get(q_id, "").strip().upper()

        if not user_opt:
            unanswered_count += 1
            is_correct = False
            status = "unanswered"
        elif user_opt == correct_opt:
            correct_count += 1
            is_correct = True
            status = "correct"
        else:
            incorrect_count += 1
            is_correct = False
            status = "incorrect"

        breakdown.append({
            "id": q_id,
            "question": q["question"],
            "options": q["options"],
            "user_answer": user_opt,
            "correct_answer": correct_opt,
            "is_correct": is_correct,
            "status": status,
            "explanation": q["explanation"],
            "reference_topic": q.get("reference_topic", "General")
        })

    percentage = round((correct_count / total) * 100, 1)

    # Assign Letter Grade
    if percentage >= 90:
        grade = "A+"
        feedback = "Outstanding Mastery! 🌟 You demonstrated exceptional understanding of the material."
    elif percentage >= 80:
        grade = "A"
        feedback = "Excellent Performance! 🎯 You have a strong grasp of the core concepts."
    elif percentage >= 70:
        grade = "B"
        feedback = "Good Job! 👍 You understand the main principles well. Review the missed questions below."
    elif percentage >= 60:
        grade = "C"
        feedback = "Satisfactory! 📘 You passed, but reinforcing key topics will boost your confidence."
    elif percentage >= 50:
        grade = "D"
        feedback = "Needs Improvement! ⚠️ Consider re-reading the summary and chatting with AI on tricky topics."
    else:
        grade = "F"
        feedback = "Keep Practicing! 💡 Review the study notes and retake the quiz after revision."

    return {
        "total_questions": total,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "unanswered_count": unanswered_count,
        "score_percentage": percentage,
        "grade": grade,
        "passed": percentage >= 60.0,
        "feedback": feedback,
        "breakdown": breakdown
    }
