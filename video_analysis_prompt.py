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

Example:
For these keyframes:
- Frame 1: People sitting in a classroom setting
- Frame 2: A teacher writing on a blackboard

Instructions:
- Output ONLY raw JSON (no markdown, no ```json blocks, no extra text)
- The response must begin with `{` and end with `}`
- Do not add any explanation, introduction, or comments

Please provide output in exactly this format do not deviate from this json format:

{
    "questions": [
        {"q": "What is the setting and time of day for this educational activity?", "a": "An indoor classroom during daytime hours with natural lighting"},
        {"q": "How many participants are present and what are their roles in this learning environment?", "a": "One teacher and approximately 20 students engaged in a lecture format"},
        {"q": "What objects and equipment are visible in the classroom and how are they being used?", "a": "A whiteboard for instruction, desks and chairs arranged in rows, and a projector for visual aids"},
        {"q": "What is the camera perspective and how does it capture the classroom dynamics?", "a": "Static camera positioned at the front of the class, providing a clear view of both teacher and student interactions"},
        {"q": "What teaching methods and student interactions are visible in this educational setting?", "a": "Teacher-led instruction using board work, followed by student group discussions in a collaborative learning environment"}
    ],
    "keywords": ["education", "classroom", "teaching", "students", "group work", "learning"],
    "map_placement": {
        "value": "Town"
    },
    "summary": "The video depicts a daytime classroom scene with a teacher instructing approximately 20 students, followed by students engaging in group discussions. The classroom is equipped with desks, chairs, a whiteboard, and a projector. The lighting is well-lit with natural light, and the camera is static from the front of the class."
}

Please analyze the provided all keyframes and provide only one output combining all information in the same JSON structure
"""
