import os
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

PROMPT_TEMPLATE = """
You are a creative podcast researcher for a show hosted by Björn (the curious, enthusiastic questioner) and Felix (the knowledgeable, witty explainer).

For the main topic: "{topic}", do the following:

1. List 5 subtopics that would make for a lively, engaging, and varied podcast episode. 
   - At least one subtopic should be controversial or debated in the field.
   - At least one subtopic should be humorous, quirky, or unexpected.

2. For each subtopic, provide a 2-3 sentence summary or fun fact, written in a conversational and engaging style suitable for a podcast script.

3. Output in the following structured format (repeat for each subtopic):

Subtopic: <subtopic title>
Summary: <engaging, conversational summary or fun fact>

Make sure the subtopics are diverse, surprising, and would spark interesting conversation between the hosts.
"""


def fetch_subtopics_and_summaries(topic):
    prompt = PROMPT_TEMPLATE.format(topic=topic)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.9,
    )
    content = response["choices"][0]["message"]["content"]
    # Parse the response into subtopics and summaries
    subtopics = []
    summaries = {}
    current_subtopic = None
    for line in content.splitlines():
        if line.startswith("Subtopic:"):
            current_subtopic = line.replace("Subtopic:", "").strip()
            subtopics.append(current_subtopic)
        elif line.startswith("Summary:") and current_subtopic:
            summaries[current_subtopic] = line.replace("Summary:", "").strip()
    return subtopics, summaries


# NOTE: OpenAI utility functions have been moved to app/services/openai_utils.py
