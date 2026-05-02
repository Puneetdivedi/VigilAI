"""
Prompt templates for the LLM agents.
"""
from langchain_core.prompts import PromptTemplate

DIAGNOSIS_PROMPT = PromptTemplate.from_template(
    """You are an expert industrial equipment diagnostic AI.
You have detected a potential issue with a machine. 
Based on the ML model's prediction and the provided maintenance manual guidelines, generate a diagnostic report.

Machine ID: {machine_id}
Predicted Fault Type: {fault_label}
Anomaly Score: {anomaly_score} (0 is normal, 1 is highly anomalous)

Maintenance Manual Guidelines:
{context}

Generate a clear, plain-English report with the following structure:
1. Fault Summary: A brief explanation of the issue.
2. Severity: Assign one of (Low, Medium, High).
3. Recommended Actions: Provide exactly 3 actionable bullet points.
4. Estimated Downtime Risk: Describe the risk if left unaddressed.

Format your response strictly using these 4 headings. Do not include extra conversational text.
"""
)
