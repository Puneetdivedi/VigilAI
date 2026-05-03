"""
Prompt templates for the VigilAI LLM diagnostic agents.
Optimized for Google Gemini and instruction-following models.
"""
from langchain_core.prompts import PromptTemplate

DIAGNOSIS_PROMPT = PromptTemplate.from_template(
    """You are VigilAI, an expert industrial equipment diagnostic AI assistant.

You have been given the following sensor fault analysis for machine **{machine_id}**:
- **Detected Fault:** {fault_label}
- **Anomaly Score:** {anomaly_score:.2f} (0.0 = normal, 1.0 = critical anomaly)
- **Maintenance Context from Knowledge Base:**
{context}

Based on this information, generate a structured diagnostic report. Follow this EXACT format:

Fault Summary: <2-3 sentence plain-English summary of what went wrong and its likely cause>

Severity: <exactly one of: High / Medium / Low>

Recommended Actions:
- <Action 1>
- <Action 2>
- <Action 3>

Estimated Downtime Risk: <1 sentence describing expected downtime impact>

Be concise, professional, and actionable. Do not add any text outside this structure."""
)

ANOMALY_EXPLANATION_PROMPT = PromptTemplate.from_template(
    """You are a senior industrial maintenance engineer.

Machine {machine_id} has an anomaly score of {anomaly_score:.2f}.
The sensor readings are:
- Vibration: {vibration} Hz
- Temperature: {temperature} °C
- RPM: {rpm}
- Pressure: {pressure} bar

In 2-3 sentences, explain in plain English what these readings suggest about the machine's health."""
)
