from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from datetime import datetime
from typing import Dict, Any, Optional

class SummarySection:
    """Value object representing a section of the summary"""
    def __init__(self, name: str, content: str):
        self.name = name
        self.content = content

class ReportSummarizer:
    """Main class for generating structured summaries from agricultural reports"""
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        self.model_name = model_name
        self._initialize_model()
        self.max_length = 2048
        self.temperature = 0.7
        self.sections = [
            "Market Trends",
            "Price Forecasts",
            "Supply/Demand Analysis",
            "Key Risk Factors"
        ]

    def _initialize_model(self) -> None:
        """Initialize the language model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def _preprocess_text(self, text: str) -> str:
        """Preprocess and truncate input text if necessary"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            text = self.tokenizer.decode(tokens)
        return text

    def _create_prompt(self, text: str) -> str:
        """Create a structured prompt for the model"""
        return f"""Please analyze this agricultural report and provide a structured summary focusing on:
1. Market Trends
2. Price Forecasts
3. Supply/Demand Analysis
4. Key Risk Factors

Report Text:
{text}

Provide a concise, structured analysis:"""

    def _generate_summary(self, prompt: str) -> str:
        """Generate summary using the language model"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=1000,
                    temperature=self.temperature,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            raise RuntimeError(f"Failed to generate summary: {str(e)}")

    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract specific section from the generated summary"""
        try:
            start = text.index(section_name)
            next_section = float('inf')
            
            for section in self.sections:
                if section != section_name:
                    try:
                        pos = text.index(section, start)
                        next_section = min(next_section, pos)
                    except ValueError:
                        continue
            
            section_text = text[start:next_section].strip() if next_section != float('inf') \
                          else text[start:].strip()
            return section_text.replace(section_name, "").strip(":\n -")
        except ValueError:
            return ""

    def summarize(self, text: str) -> Dict[str, Any]:
        """Generate a structured summary from the input text"""
        try:
            preprocessed_text = self._preprocess_text(text)
            prompt = self._create_prompt(preprocessed_text)
            summary = self._generate_summary(prompt)
            
            summary_sections = {
                section.lower().replace(" ", "_"): self._extract_section(summary, section)
                for section in self.sections
            }
            
            return {
                "timestamp": datetime.now().isoformat(),
                "summary": summary_sections,
                "model_info": {
                    "model_name": self.model_name,
                    "temperature": self.temperature
                }
            }
        except Exception as e:
            raise RuntimeError(f"Failed to generate summary: {str(e)}")

    def to_json(self, summary: Dict[str, Any]) -> str:
        """Convert summary to JSON string"""
        return json.dumps(summary, indent=2)