import random
import wikipediaapi
from openai_utils import fetch_subtopics_and_summaries


class StateStore:
    def __init__(self):
        self.covered_subtopics = []
        self.used_facts = set()
        self.host_memories = {"Björn": [], "Felix": []}
        self.running_jokes = []

    def add_subtopic(self, subtopic):
        self.covered_subtopics.append(subtopic)

    def add_fact(self, fact):
        self.used_facts.add(fact)

    def add_memory(self, host, memory):
        self.host_memories[host].append(memory)

    def add_joke(self, joke):
        self.running_jokes.append(joke)

    def has_covered(self, subtopic):
        return subtopic in self.covered_subtopics

    def has_used_fact(self, fact):
        return fact in self.used_facts


class ConversationOrchestrator:
    def __init__(self, main_topic, subtopics, summaries):
        self.state = StateStore()
        self.main_topic = main_topic
        self.subtopics = subtopics
        self.summaries = summaries  # dict: subtopic -> summary
        self.current_segment = 0
        self.hosts = ["Björn", "Felix"]

    def next_segment(self):
        if self.current_segment >= len(self.subtopics):
            return None
        subtopic = self.subtopics[self.current_segment]
        summary = self.summaries.get(subtopic, "")
        self.state.add_subtopic(subtopic)
        host_intro = self.hosts[self.current_segment % 2]
        host_reply = self.hosts[(self.current_segment + 1) % 2]
        # Simple dynamic prompt for now
        intro = f'{host_intro}: "I was reading about {subtopic}. What do you think, {host_reply}?"'
        reply = f'{host_reply}: "Great question! {summary}"'
        self.current_segment += 1
        return intro + "\n" + reply

    def generate_full_script(self):
        script = []
        while True:
            segment = self.next_segment()
            if not segment:
                break
            script.append(segment)
        return "\n".join(script)


class ResearchAgent:
    def __init__(self):
        pass

    def get_subtopics_and_summaries(self, main_topic, max_subtopics=5):
        subtopics, summaries = fetch_subtopics_and_summaries(main_topic)
        # Optionally limit to max_subtopics
        subtopics = subtopics[:max_subtopics]
        summaries = {k: summaries[k] for k in subtopics if k in summaries}
        return subtopics, summaries


# NOTE: Agent classes have been moved to app/agents/
