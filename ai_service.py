import os
import streamlit as st
import requests
import json
import re
from sqlalchemy import inspect, Engine
from typing import TypedDict, Tuple, Optional, Dict, Any, List

# Model registry with capabilities
MODELS = {
    # OpenRouter models with vision support
    'openrouter/google/gemini-2.0-flash-exp:free': {
        'description': 'Gemini 2.0 Flash (Free) - Fast with vision support',
        'provider': 'openrouter',
        'supports_json': True,
        'supports_vision': True,
        'best_for': ['sql', 'vision', 'general']
    },
    'openrouter/google/gemini-pro-1.5': {
        'description': 'Gemini Pro 1.5 - Advanced with vision',
        'provider': 'openrouter',
        'supports_json': True,
        'supports_vision': True,
        'best_for': ['sql', 'reasoning', 'vision']
    },
    'openrouter/deepseek/deepseek-chat': {
        'description': 'DeepSeek Chat via OpenRouter - excellent for SQL',
        'provider': 'openrouter',
        'supports_json': True,
        'supports_vision': False,
        'best_for': ['sql', 'code', 'reasoning']
    },
    'openrouter/anthropic/claude-3.5-sonnet': {
        'description': 'Claude 3.5 Sonnet - top-tier reasoning with vision',
        'provider': 'openrouter',
        'supports_json': True,
        'supports_vision': True,
        'best_for': ['sql', 'reasoning', 'analysis', 'vision']
    },
    'openrouter/openai/gpt-4o': {
        'description': 'GPT-4o via OpenRouter - vision capable',
        'provider': 'openrouter',
        'supports_json': True,
        'supports_vision': True,
        'best_for': ['sql', 'reasoning', 'vision']
    },

    # OpenAI models
    'openai/gpt-4-turbo': {
        'description': 'GPT-4 Turbo - powerful general model',
        'provider': 'openai',
        'supports_json': True,
        'supports_vision': True,
        'best_for': ['sql', 'reasoning', 'general']
    },
    'openai/gpt-4o': {
        'description': 'GPT-4o - fast and capable with vision',
        'provider': 'openai',
        'supports_json': True,
        'supports_vision': True,
        'best_for': ['sql', 'reasoning', 'general', 'vision']
    },

    # Anthropic models
    'anthropic/claude-3-5-sonnet-20241022': {
        'description': 'Claude 3.5 Sonnet - excellent reasoning and SQL',
        'provider': 'anthropic',
        'supports_json': False,
        'supports_vision': True,
        'best_for': ['sql', 'reasoning', 'analysis']
    },

    # Google Gemini models
    'google/gemini-1.5-pro': {
        'description': 'Gemini 1.5 Pro - strong analytical capabilities',
        'provider': 'google',
        'supports_json': True,
        'supports_vision': True,
        'best_for': ['sql', 'analysis', 'reasoning']
    },
}

# Default model
DEFAULT_MODEL = 'openrouter/deepseek/deepseek-chat'

# Provider configurations
PROVIDER_CONFIGS = {
    'openrouter': {
        'api_url_env': 'OPENROUTER_API_URL',
        'api_key_env': 'OPENROUTER_API_KEY',
        'default_url': 'https://openrouter.ai/api/v1',
        'key_prefix': 'sk-or-',
        'requires_referer': True,
    },
    'openai': {
        'api_url_env': 'OPENAI_API_URL',
        'api_key_env': 'OPENAI_API_KEY',
        'default_url': 'https://api.openai.com/v1',
        'key_prefix': 'sk-',
        'requires_referer': False,
    },
    'anthropic': {
        'api_url_env': 'ANTHROPIC_API_URL',
        'api_key_env': 'ANTHROPIC_API_KEY',
        'default_url': 'https://api.anthropic.com/v1',
        'key_prefix': 'sk-ant-',
        'requires_referer': False,
    },
    'google': {
        'api_url_env': 'GOOGLE_API_URL',
        'api_key_env': 'GOOGLE_API_KEY',
        'default_url': 'https://generativelanguage.googleapis.com/v1beta',
        'key_prefix': None,
        'requires_referer': False,
    },
}


class AISqlResponse(TypedDict):
    explanation: str
    sqlQuery: str


def parse_json_response(content: str) -> Optional[dict]:
    """Robust JSON parsing with multiple fallback strategies."""
    if not content:
        return None

    if isinstance(content, dict):
        return content

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Extract JSON from code blocks
    json_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{.*\}',
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Find first { and last }
    start = content.find('{')
    end = content.rfind('}')

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def get_configured_provider() -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Detect which provider is configured and return its config."""
    priority_providers = ['openrouter', 'openai', 'anthropic', 'google']

    for provider in priority_providers:
        config = PROVIDER_CONFIGS[provider]
        api_key = os.getenv(config['api_key_env'])

        if api_key:
            api_url = os.getenv(config['api_url_env'], config['default_url'])
            return provider, {
                'api_key': api_key,
                'api_url': api_url,
                'config': config
            }

    return None, None


def get_available_models() -> list[str]:
    """Get list of available models based on configured provider."""
    provider, provider_data = get_configured_provider()

    if not provider:
        return [DEFAULT_MODEL]

    available = [k for k, v in MODELS.items() if v['provider'] == provider]

    if not available:
        return [DEFAULT_MODEL]

    return available


def select_best_model() -> str:
    """Select the best available model for text-to-SQL tasks."""
    try:
        available = get_available_models()

        if not available:
            return DEFAULT_MODEL

        # Prefer models with SQL in best_for
        sql_models = [m for m in available if 'sql' in MODELS.get(m, {}).get('best_for', [])]

        if sql_models:
            model = sql_models[0]
            if model in MODELS:
                st.info(f"Using: {model} - {MODELS[model]['description']}")
            return model

        return available[0]
    except Exception as e:
        st.warning(f"Model selection error: {e}")
        return DEFAULT_MODEL


@st.cache_data(ttl=600)
def get_database_schema(_engine: Engine, table_whitelist: Optional[List[str]] = None) -> str:
    """Get database schema as CREATE TABLE statements with optional filtering."""
    try:
        inspector = inspect(_engine)
        schema_str = ""
        table_names = inspector.get_table_names()

        if not table_names:
            return "Error: No tables found in the database."

        if table_whitelist:
            table_names = [t for t in table_names if t in table_whitelist]
            if not table_names:
                return f"Error: No tables match whitelist: {table_whitelist}"

        max_tables = int(os.getenv("MAX_SCHEMA_TABLES", "50"))
        if len(table_names) > max_tables:
            st.warning(f"⚠️ Large schema detected: showing {max_tables} of {len(table_names)} tables.")
            table_names = table_names[:max_tables]

        for table_name in table_names:
            schema_str += f"CREATE TABLE {table_name} (\n"
            columns = inspector.get_columns(table_name)
            for i, column in enumerate(columns):
                col_name = column['name']
                col_type = str(column['type'])
                schema_str += f"  {col_name} {col_type}"
                if i < len(columns) - 1:
                    schema_str += ",\n"
            schema_str += "\n);\n\n"

        return schema_str
    except Exception as e:
        return f"Error inspecting schema: {e}"


def call_openrouter_api(
    model: str,
    messages: list,
    api_key: str,
    api_url: str,
    image_url: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Call OpenRouter API with optional image support."""
    try:
        endpoint = api_url.rstrip('/')
        if not endpoint.endswith('/chat/completions'):
            if not endpoint.endswith('/v1'):
                endpoint += '/v1'
            endpoint += '/chat/completions'

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://github.com/text-to-sql"),
            "X-Title": os.getenv("OPENROUTER_TITLE", "Text-to-SQL Agent")
        }

        # Extract model name after 'openrouter/'
        model_name = model.replace('openrouter/', '')

        # Prepare messages with image if provided
        prepared_messages = []
        for msg in messages:
            if msg["role"] == "user" and image_url:
                # Add image to user message
                prepared_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": msg["content"]},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                })
            else:
                prepared_messages.append(msg)

        payload = {
            "model": model_name,
            "temperature": 0.3,
            "messages": prepared_messages,
            "response_format": {"type": "json_object"},
        }

        resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        return content, None

    except requests.exceptions.HTTPError as e:
        error_detail = e.response.text if e.response else str(e)
        return None, f"OpenRouter API error: {error_detail}"
    except Exception as e:
        return None, f"OpenRouter error: {str(e)}"


def call_openai_api(model: str, messages: list, api_key: str, api_url: str) -> Tuple[Optional[str], Optional[str]]:
    """Call OpenAI API."""
    try:
        endpoint = api_url.rstrip('/')
        if not endpoint.endswith('/chat/completions'):
            if not endpoint.endswith('/v1'):
                endpoint += '/v1'
            endpoint += '/chat/completions'

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        model_name = model.replace('openai/', '')

        payload = {
            "model": model_name,
            "temperature": 0.3,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

        resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        return content, None

    except requests.exceptions.HTTPError as e:
        error_detail = e.response.text if e.response else str(e)
        return None, f"OpenAI API error: {error_detail}"
    except Exception as e:
        return None, f"OpenAI error: {str(e)}"


def call_anthropic_api(model: str, messages: list, api_key: str, api_url: str) -> Tuple[Optional[str], Optional[str]]:
    """Call Anthropic API."""
    try:
        endpoint = api_url.rstrip('/')
        if not endpoint.endswith('/messages'):
            endpoint += '/messages'

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        model_name = model.replace('anthropic/', '')

        system_msg = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg += msg["content"] + "\n"
            else:
                user_messages.append(msg)

        system_msg += "\nRespond with a valid JSON object containing 'explanation' and 'sqlQuery' fields."

        payload = {
            "model": model_name,
            "max_tokens": 2048,
            "temperature": 0.3,
            "system": system_msg,
            "messages": user_messages,
        }

        resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        content_blocks = data.get("content", [])
        if content_blocks and isinstance(content_blocks, list):
            content = content_blocks[0].get("text", "")
            return content, None

        return None, "Anthropic API returned unexpected format"

    except requests.exceptions.HTTPError as e:
        error_detail = e.response.text if e.response else str(e)
        return None, f"Anthropic API error: {error_detail}"
    except Exception as e:
        return None, f"Anthropic error: {str(e)}"


def call_google_api(model: str, messages: list, api_key: str, api_url: str) -> Tuple[Optional[str], Optional[str]]:
    """Call Google Gemini API."""
    try:
        model_name = model.replace('google/', '')
        endpoint = f"{api_url.rstrip('/')}/models/{model_name}:generateContent?key={api_key}"

        headers = {
            "Content-Type": "application/json",
        }

        contents = []
        system_instruction = ""

        for msg in messages:
            if msg["role"] == "system":
                system_instruction += msg["content"] + "\n"
            else:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.3,
                "responseMimeType": "application/json",
            }
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{
                    "text": system_instruction + "\nRespond with a valid JSON object containing 'explanation' and 'sqlQuery' fields."
                }]
            }

        resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return content, None

        return None, "Google API returned no candidates"

    except requests.exceptions.HTTPError as e:
        error_detail = e.response.text if e.response else str(e)
        return None, f"Google API error: {error_detail}"
    except Exception as e:
        return None, f"Google error: {str(e)}"


def get_ai_response(
    user_prompt: str,
    schema: str,
    image_url: Optional[str] = None
) -> Tuple[Optional[AISqlResponse], Optional[str]]:
    """Send request to configured AI provider and get SQL response."""
    try:
        provider, provider_data = get_configured_provider()

        if not provider:
            return None, (
                "No AI API configured. Please set one of the following in your .env file:\n"
                "- OPENROUTER_API_KEY for OpenRouter (https://openrouter.ai/keys)\n"
                "- OPENAI_API_KEY for OpenAI (https://platform.openai.com/api-keys)\n"
                "- ANTHROPIC_API_KEY for Anthropic (https://console.anthropic.com/)\n"
                "- GOOGLE_API_KEY for Google Gemini (https://makersuite.google.com/app/apikey)"
            )

        model = select_best_model()

        system_msg = (
            "You are an expert Text-to-SQL agent. Analyze the user's natural language query "
            "and the database schema, then generate a single, valid, SELECT-only PostgreSQL query. "
            "Provide a brief one-sentence explanation of what the query does. "
            "Respond with a JSON object containing exactly two fields: 'explanation' and 'sqlQuery'."
            f"\n\nDatabase Schema:\n{schema}"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]

        api_key = provider_data['api_key']
        api_url = provider_data['api_url']

        if provider == 'openrouter':
            content, error = call_openrouter_api(model, messages, api_key, api_url, image_url)
        elif provider == 'openai':
            content, error = call_openai_api(model, messages, api_key, api_url)
        elif provider == 'anthropic':
            content, error = call_anthropic_api(model, messages, api_key, api_url)
        elif provider == 'google':
            content, error = call_google_api(model, messages, api_key, api_url)
        else:
            return None, f"Unsupported provider: {provider}"

        if error:
            return None, error

        if not content:
            return None, "API returned empty response"

        parsed = parse_json_response(content)

        if not parsed:
            return None, f"Could not parse valid JSON from response: {content[:200]}..."

        if "explanation" not in parsed or "sqlQuery" not in parsed:
            return None, f"Missing required fields. Got: {list(parsed.keys())}"

        return {
            "explanation": parsed["explanation"],
            "sqlQuery": parsed["sqlQuery"]
        }, None

    except Exception as e:
        return None, f"Unexpected error: {str(e)}"