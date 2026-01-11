"""
LLM client implementations for various providers.

This module provides unified interfaces for querying different LLM providers:
- OpenAI (GPT models)
- Google Gemini
- AWS Bedrock
- Anthropic (Claude models)
"""

import json
import traceback
from typing import Dict, Optional

import boto3
import google.generativeai as genai
from anthropic import Anthropic
from botocore.exceptions import ClientError
from google.generativeai.types import FunctionDeclaration, HarmBlockThreshold, HarmCategory, Tool
from openai import OpenAI

from utils.config import (
    AWS_REGION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P
)


def query_openai_model(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: Optional[str] = None,
    llm_params: Optional[Dict] = None,
    max_retries: int = 3,
    retry_delay: int = 5
) -> Optional[str]:
    """
    Query OpenAI model using the OpenAI Python client.

    Args:
        system_prompt: System prompt for the model
        user_prompt: User prompt for the model
        model: Model name (e.g., 'gpt-4o')
        api_key: OpenAI API key
        llm_params: Dictionary with parameters like temperature, max_tokens, etc.
        max_retries: Maximum number of retries (unused, kept for API consistency)
        retry_delay: Delay between retries (unused, kept for API consistency)

    Returns:
        Model response as string, or None if request fails
    """
    try:
        client = OpenAI(api_key=api_key)

        params = {
            'temperature': DEFAULT_TEMPERATURE,
            'max_tokens': DEFAULT_MAX_TOKENS,
            'top_p': DEFAULT_TOP_P,
            **(llm_params or {})
        }

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=params['temperature'],
            max_tokens=params['max_tokens'],
            top_p=params['top_p']
        )

        content = response.choices[0].message.content
        if content is not None:
            return content.strip()

        print("Warning: Received empty content from OpenAI API")
        return None

    except Exception as e:
        print(f"Error querying OpenAI model '{model}': {e}")
        return None


def query_gemini_model(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: Optional[str] = None,
    llm_params: Optional[Dict] = None,
    max_retries: int = 3,
    retry_delay: int = 5
) -> Optional[str]:
    """
    Query a Google Gemini model.

    Args:
        system_prompt: System instruction for the model
        user_prompt: User prompt for the model
        model: Model name (e.g., 'gemini-2.5-pro')
        api_key: Google Gemini API key
        llm_params: Dictionary with parameters like temperature, max_tokens, etc.
        max_retries: Maximum number of retries (unused, kept for API consistency)
        retry_delay: Delay between retries (unused, kept for API consistency)

    Returns:
        Model response as string, or None if request fails
    """
    if not api_key:
        print("ERROR: Google Gemini API key not provided.")
        return None

    try:
        genai.configure(api_key=api_key)

        params = llm_params or {}
        generation_config = genai.GenerationConfig(
            temperature=params.get('temperature', DEFAULT_TEMPERATURE),
            top_p=params.get('top_p', DEFAULT_TOP_P),
            max_output_tokens=params.get('max_tokens', DEFAULT_MAX_TOKENS)
        )

        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt
        )

        tool_config = {
            "function_calling_config": {
                "mode": "NONE"
            }
        }

        response = gemini_model.generate_content(
            user_prompt,
            generation_config=generation_config,
            tool_config=tool_config
        )

        if not response.parts:
            print(f"WARN: Gemini response for model '{model}' was empty or blocked. "
                  f"Feedback: {response.prompt_feedback}")
            return None

        return response.text.strip()

    except Exception as e:
        print(f"ERROR: Exception in query_gemini_model for model '{model}'. "
              f"Type: {type(e).__name__}, Details: {e}")
        traceback.print_exc()
        return None


def query_anthropic_model(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: Optional[str] = None,
    llm_params: Optional[Dict] = None,
    max_retries: int = 3,
    retry_delay: int = 5
) -> Optional[str]:
    """
    Query Anthropic Claude model using the Anthropic Python client.

    Args:
        system_prompt: System prompt for the model
        user_prompt: User prompt for the model
        model: Model name (e.g., 'claude-3-5-sonnet-20241022')
        api_key: Anthropic API key
        llm_params: Dictionary with parameters like temperature, max_tokens, etc.
        max_retries: Maximum number of retries (unused, kept for API consistency)
        retry_delay: Delay between retries (unused, kept for API consistency)

    Returns:
        Model response as string, or None if request fails
    """
    if not api_key:
        print("ERROR: Anthropic API key not provided.")
        return None

    try:
        client = Anthropic(api_key=api_key)

        params = {
            'temperature': DEFAULT_TEMPERATURE,
            'max_tokens': DEFAULT_MAX_TOKENS,
            'top_p': DEFAULT_TOP_P,
            **(llm_params or {})
        }

        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=params['temperature'],
            max_tokens=params['max_tokens'],
            top_p=params['top_p']
        )

        if response.content and len(response.content) > 0:
            return response.content[0].text.strip()

        print("Warning: Received empty content from Anthropic API")
        return None

    except Exception as e:
        print(f"Error querying Anthropic model '{model}': {e}")
        traceback.print_exc()
        return None


def _create_bedrock_client(api_keys: Optional[Dict] = None) -> boto3.client:
    """
    Create AWS Bedrock client with optional credentials.

    Args:
        api_keys: Dictionary containing AWS credentials

    Returns:
        Configured boto3 Bedrock client
    """
    client_args = {"region_name": AWS_REGION}

    if api_keys and api_keys.get('aws_access_key_id') and api_keys.get('aws_secret_access_key'):
        client_args['aws_access_key_id'] = api_keys['aws_access_key_id']
        client_args['aws_secret_access_key'] = api_keys['aws_secret_access_key']

    return boto3.client("bedrock-runtime", **client_args)


def query_aws_bedrock_model(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_keys: Optional[Dict] = None,
    llm_params: Optional[Dict] = None,
    max_retries: int = 3,
    retry_delay: int = 5
) -> Optional[str]:
    """
    Query AWS Bedrock models with retry logic.

    Args:
        system_prompt: System prompt for the model
        user_prompt: User prompt for the model
        model: Model ARN or ID
        api_keys: Dictionary containing AWS credentials
        llm_params: Dictionary with parameters like temperature, max_tokens, etc.
        max_retries: Maximum number of retries
        retry_delay: Delay in seconds between retries

    Returns:
        Model response as string, or None if all retries fail
    """
    import time

    params = llm_params or {}

    for attempt in range(1, max_retries + 1):
        try:
            client = _create_bedrock_client(api_keys)

            messages = [{"role": "user", "content": [{"text": user_prompt}]}]
            system_messages = [{"text": system_prompt}]
            inference_config = {
                'maxTokens': params.get('max_tokens', DEFAULT_MAX_TOKENS),
                'temperature': params.get('temperature', DEFAULT_TEMPERATURE),
                'topP': params.get('top_p', DEFAULT_TOP_P),
            }

            response = client.converse(
                modelId=model,
                messages=messages,
                system=system_messages,
                inferenceConfig=inference_config
            )

            response_text = response["output"]["message"]["content"][0]["text"]
            return response_text.strip()

        except (ClientError, Exception) as e:
            if isinstance(e, ClientError) and 'ThrottlingException' in e.response.get('Error', {}).get('Code', ''):
                print(f"WARN: Bedrock API throttling detected for '{model}'. Reason: {e}")
            else:
                print(f"ERROR: Error querying Bedrock model '{model}'. Reason: {e}")

            if attempt < max_retries:
                print(f"Retrying in {retry_delay} seconds... (Attempt {attempt}/{max_retries})")
                time.sleep(retry_delay)

    print(f"Failed to get response from Bedrock model '{model}' after {max_retries} attempts.")
    return None
