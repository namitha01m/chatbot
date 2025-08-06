import json
import requests
import os
from typing import Tuple, Optional

OLLAMA_API_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "llama3" 

def _generate_llama_response_for_eval(prompt: str) -> str:
    """
    Helper function to interact with the Ollama LLM for evaluation purposes.
    Similar to generate_llama_response in rag_agent.py, but for internal use.
    """
    url = f"{OLLAMA_API_BASE_URL}/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": { 
            "temperature": 0.1, 
            "top_p": 0.9,
            "num_predict": 100 
        }
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60) # Shorter timeout for eval
        response.raise_for_status()
        result = response.json()
        return result["response"].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error during evaluation LLM call: {e}")
        return "ERROR: LLM evaluation failed."

def evaluate_groundedness(response: str, context: str) -> Tuple[float, str]:
    
    if not context.strip():
        return 0.0, "No context provided for groundedness evaluation."

    groundedness_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"You are an expert evaluator. Your task is to assess if a given RESPONSE is FULLY GROUNDED in the provided CONTEXT. "
        f"A response is FULLY GROUNDED if all factual claims in the response can be directly and solely supported by the context. "
        f"If the response contains any information not present in the context, it is NOT fully grounded.\n"
        f"Output your assessment as a JSON object with two keys: 'score' (float, 0.0 for not grounded, 1.0 for fully grounded) and 'reason' (string, explanation).\n"
        f"CONTEXT:\n{context}\n\n"
        f"RESPONSE:\n{response}\n"
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"Is the RESPONSE fully grounded in the CONTEXT? Provide a score and reason as JSON.\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

    llm_eval_response = _generate_llama_response_for_eval(groundedness_prompt)
    
    try:
        
        eval_data = json.loads(llm_eval_response)
        score = float(eval_data.get("score", 0.0))
        reason = eval_data.get("reason", "Evaluation reason not provided by LLM.")
        return score, reason
    except json.JSONDecodeError:
        print(f"Warning: LLM did not return valid JSON for groundedness evaluation: {llm_eval_response}")
        return 0.0, f"LLM evaluation failed to parse JSON. Raw response: {llm_eval_response}"
    except Exception as e:
        print(f"Error processing LLM response for groundedness: {e}")
        return 0.0, f"Error in evaluation: {e}"


def evaluate_relevance(query: str, response: str) -> Tuple[float, str]:
    
    relevance_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"You are an expert evaluator. Your task is to assess if a given RESPONSE is RELEVANT to the provided QUERY. "
        f"A response is RELEVANT if it directly and comprehensively answers the query. "
        f"Output your assessment as a JSON object with two keys: 'score' (float, 0.0 for not relevant, 1.0 for fully relevant) and 'reason' (string, explanation).\n"
        f"QUERY:\n{query}\n\n"
        f"RESPONSE:\n{response}\n"
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"Is the RESPONSE relevant to the QUERY? Provide a score and reason as JSON.\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

    llm_eval_response = _generate_llama_response_for_eval(relevance_prompt)

    try:
        eval_data = json.loads(llm_eval_response)
        score = float(eval_data.get("score", 0.0))
        reason = eval_data.get("reason", "Evaluation reason not provided by LLM.")
        return score, reason
    except json.JSONDecodeError:
        print(f"Warning: LLM did not return valid JSON for relevance evaluation: {llm_eval_response}")
        return 0.0, f"LLM evaluation failed to parse JSON. Raw response: {llm_eval_response}"
    except Exception as e:
        print(f"Error processing LLM response for relevance: {e}")
        return 0.0, f"Error in evaluation: {e}"


