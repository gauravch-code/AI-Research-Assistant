import dspy
import os
from dotenv import load_dotenv

load_dotenv()
dspy.settings.configure(openai_api_key=os.getenv("OPENAI_API_KEY"))

class PaperSummarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarize = dspy.Predict("context -> summary")

    def forward(self, context):
        return self.summarize(context=context).summary
