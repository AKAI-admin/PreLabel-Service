"""
Video analysis prompt template for GPT-4 vision API.
This prompt is used to analyze video keyframes and generate structured data for video labeling.
"""

VIDEO_ANALYSIS_PROMPT = """
You are an AI assistant that is to assist in data labeling of videos. Please analyze these video keyframes and identify the following aspects:
1. Location (Where?)
2. Number of participants (Who?)
3. Event description (What?)
4. Timing (When?)
5. Objects present
6. Actions/Activities
7. Interactions
8. Scene type
9. Lighting conditions
10. Weather (if applicable)
11. Time of day
12. Camera perspective
13. Emotions/Expressions
14. Text in scene
15. Anomalies/Events
16. Background and Foreground
17. Occlusion
18. Scale and Size
19. Colors
20. Additional notes: Provide any other observations or details that might be relevant.

Please provide detailed descriptions for each aspect based on the keyframes.
For all of these aspect:
- Generate 5 relevant questions that include a combination of the aspect and answers
- Flag any uncertain information

Additional requirements:
- Generate relevant keywords describing the main elements and themes
- Create a detailed description of the video in as many words as possible that includes all aspects of the video
- Determine map placement from these options: Town, Village, Water body, Mountains, Snow, Road
- Use all images for creating question-answer pairs; do not favor any single image disproportionately

Example:
For these keyframes:
- Frame 1: People sitting in a classroom setting
- Frame 2: A teacher writing on a blackboard

Instructions:
- Output ONLY raw JSON (no markdown, no ```json blocks, no extra text)
- The response must begin with `{` and end with `}`
- Do not add any explanation, introduction, or comments

Please provide output in exactly this format do not deviate from this json format(donot use any work like keyframs:

{
    "questions": [
        {"q": "...?", "a": "..."},
        {"q": "...?", "a": "..."},
        {"q": "...?", "a": "..."},
        {"q": "...?", "a": "..."},
        {"q": "...?", "a": "..."}
    ],
    "keywords": ["..", "..", "..", "..", "..", ".."],
    "map_placement": {
        "value": "..."
    },
    "summary": "..."
}

Please analyze the provided all keyframes and provide only one output combining all information in the same JSON structure
"""
