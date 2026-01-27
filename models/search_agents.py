"""
Search agent implementations for LLM providers with web search capabilities.

This module provides search-enabled agents for different LLM providers:
- OpenAI with function calling
- Google Gemini with tool use
- AWS Bedrock with tool specification
- Anthropic Claude with tool use
"""

import json
import traceback
from typing import Dict, Optional

import boto3
import google.generativeai as genai
from anthropic import Anthropic
from google.generativeai.types import FunctionDeclaration, HarmBlockThreshold, HarmCategory, Tool
from openai import OpenAI

from config import DEFAULT_MAX_TOKENS
from models.llm_clients import _create_bedrock_client


def perform_web_search(query: str, serper_api_key: str) -> str:
    """
    Perform a web search using the Serper API.

    Args:
        query: The search query
        serper_api_key: API key for Serper

    Returns:
        A formatted string of search results or an error message
    """
    import requests

    if not serper_api_key:
        return "Error: Serper API key is not configured."

    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers=headers,
            data=payload,
            timeout=30
        )
        response.raise_for_status()

        search_results = response.json()
        output = ""

        # Add answer box if available
        if "answerBox" in search_results:
            answer_box = search_results["answerBox"]
            title = answer_box.get("title", "")
            snippet = answer_box.get("snippet", answer_box.get("answer", ""))
            output += f"Answer Box:\nTitle: {title}\nSnippet: {snippet}\n\n"

        # Add top organic search results
        if "organic" in search_results:
            output += "Organic Search Results:\n"
            for i, result in enumerate(search_results["organic"][:5], 1):
                title = result.get("title", "")
                link = result.get("link", "")
                snippet = result.get("snippet", "")
                output += f"{i}. Title: {title}\n   Link: {link}\n   Snippet: {snippet}\n\n"

        return output if output else "No results found."

    except requests.exceptions.RequestException as e:
        return f"Error performing web search: {e}"


def run_openai_search_agent(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    serper_api_key: str,
    llm_params: Optional[Dict] = None
) -> Optional[str]:
    """
    Run an agentic workflow with OpenAI that enables web search.

    Args:
        system_prompt: System prompt for the model
        user_prompt: User prompt for the model
        model: Model name (e.g., 'gpt-4o')
        api_key: OpenAI API key
        serper_api_key: Serper API key for web search
        llm_params: Dictionary with parameters (unused, kept for API consistency)

    Returns:
        Model response as string, or None if request fails
    """
    try:
        client = OpenAI(api_key=api_key)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "perform_web_search",
                    "description": "Performs a web search to find up-to-date information on a given topic.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to use."
                            }
                        },
                        "required": ["query"],
                    },
                }
            }
        ]

        print("INFO: Making initial call to OpenAI to plan")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        if response_message.tool_calls:
            print(f"INFO: OpenAI model requested {len(response_message.tool_calls)} tool call(s).")

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name

                if function_name == "perform_web_search":
                    function_args = json.loads(tool_call.function.arguments)
                    query = function_args.get("query")

                    print(f"INFO: Executing web search for query: '{query}'")
                    search_results = perform_web_search(query, serper_api_key)

                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": search_results,
                    })

            print("INFO: Making second call to OpenAI with all tool results")
            final_response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            return final_response.choices[0].message.content.strip()

        print("INFO: OpenAI model answered directly without searching.")
        return response_message.content.strip()

    except Exception as e:
        print(f"ERROR: Exception in run_openai_search_agent for model '{model}': {e}")
        traceback.print_exc()
        return None


def run_gemini_search_agent(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    serper_api_key: str,
    llm_params: Optional[Dict] = None
) -> Optional[str]:
    """
    Run an agentic workflow with Google Gemini that enables web search.

    Args:
        system_prompt: System prompt for the model
        user_prompt: User prompt for the model
        model: Model name (e.g., 'gemini-2.5-pro')
        api_key: Google Gemini API key
        serper_api_key: Serper API key for web search
        llm_params: Dictionary with parameters (unused, kept for API consistency)

    Returns:
        Model response as string, or None if request fails
    """
    try:
        genai.configure(api_key=api_key)

        web_search_tool = Tool(
            function_declarations=[
                FunctionDeclaration(
                    name="perform_web_search",
                    description="Performs a web search to find up-to-date information on a given topic.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to use."
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
        )

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt,
            tools=[web_search_tool],
            safety_settings=safety_settings
        )

        chat = gemini_model.start_chat()

        print("INFO: Making initial call to Gemini to plan...")
        response = chat.send_message(user_prompt)

        try:
            function_call = response.parts[0].function_call
            if function_call.name == "perform_web_search":
                print("INFO: Gemini model decided to use the web_search tool.")
                query = function_call.args['query']

                print(f"INFO: Executing web search for query: '{query}'")
                search_results_str = perform_web_search(query, serper_api_key)

                print("INFO: Making second call to Gemini with search results...")
                final_response = chat.send_message(
                    [
                        {
                            "function_response": {
                                "name": "perform_web_search",
                                "response": {"content": search_results_str}
                            }
                        }
                    ]
                )

                return final_response.text.strip()
        except (AttributeError, IndexError):
            pass

        print("INFO: Gemini model answered directly without searching.")
        return response.text.strip()

    except Exception as e:
        print(f"ERROR: Exception in run_gemini_search_agent for model '{model}': {e}")
        traceback.print_exc()
        return None


def run_bedrock_search_agent(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_keys: Dict,
    serper_api_key: str,
    llm_params: Optional[Dict] = None
) -> Optional[str]:
    """
    Run an agentic workflow with AWS Bedrock that enables web search.

    Args:
        system_prompt: System prompt for the model
        user_prompt: User prompt for the model
        model: Model ARN or ID
        api_keys: Dictionary containing AWS credentials
        serper_api_key: Serper API key for web search
        llm_params: Dictionary with parameters (unused, kept for API consistency)

    Returns:
        Model response as string, or None if request fails
    """
    try:
        client = _create_bedrock_client(api_keys)

        messages = [{"role": "user", "content": [{"text": user_prompt}]}]

        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "perform_web_search",
                        "description": "Performs a web search to find up-to-date information on a given topic.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The search query to use."
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    }
                }
            ]
        }

        print("INFO: Making initial call to Bedrock to plan")
        response = client.converse(
            modelId=model,
            messages=messages,
            system=[{"text": system_prompt}],
            toolConfig=tool_config
        )

        response_message = response['output']['message']
        messages.append(response_message)

        stop_reason = response.get('stopReason')
        if stop_reason == 'tool_use':
            print("INFO: Bedrock model decided to use the web_search tool.")
            tool_request = next((content for content in response_message['content'] if 'toolUse' in content), None)

            if tool_request:
                tool_use_id = tool_request['toolUse']['toolUseId']
                tool_name = tool_request['toolUse']['name']

                if tool_name == "perform_web_search":
                    tool_input = tool_request['toolUse']['input']
                    query = tool_input.get("query")

                    print(f"INFO: Executing web search for query: '{query}'")
                    search_results = perform_web_search(query, serper_api_key)

                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": tool_use_id,
                                    "content": [{"text": search_results}]
                                }
                            }
                        ]
                    })

                    print("INFO: Making second call to Bedrock with search results")
                    final_response = client.converse(
                        modelId=model,
                        messages=messages,
                        system=[{"text": system_prompt}],
                        toolConfig=tool_config
                    )
                    return final_response['output']['message']['content'][0]['text'].strip()

        print("INFO: Bedrock model answered directly without searching.")
        return response_message['content'][0]['text'].strip()

    except Exception as e:
        print(f"ERROR: Exception in run_bedrock_search_agent for model '{model}': {e}")
        traceback.print_exc()
        return None


def run_anthropic_search_agent(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    serper_api_key: str,
    llm_params: Optional[Dict] = None
) -> Optional[str]:
    """
    Run an agentic workflow with Anthropic Claude that enables web search.

    Args:
        system_prompt: System prompt for the model
        user_prompt: User prompt for the model
        model: Model name (e.g., 'claude-3-5-sonnet-20241022')
        api_key: Anthropic API key
        serper_api_key: Serper API key for web search
        llm_params: Dictionary with parameters (max_tokens, temperature, etc.)

    Returns:
        Model response as string, or None if request fails
    """
    try:
        # Extract max_tokens from llm_params or use default
        max_tokens = DEFAULT_MAX_TOKENS
        if llm_params and 'max_tokens' in llm_params:
            max_tokens = llm_params['max_tokens']

        client = Anthropic(api_key=api_key)

        tools = [
            {
                "name": "perform_web_search",
                "description": "Performs a web search to find up-to-date information on a given topic.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to use."
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

        print("INFO: Making initial call to Claude to plan")
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            max_tokens=max_tokens
        )

        if response.stop_reason == "tool_use":
            print("INFO: Claude model decided to use the web_search tool.")

            tool_use_block = next((block for block in response.content if block.type == "tool_use"), None)

            if tool_use_block and tool_use_block.name == "perform_web_search":
                query = tool_use_block.input.get("query")

                print(f"INFO: Executing web search for query: '{query}'")
                search_results = perform_web_search(query, serper_api_key)

                print("INFO: Making second call to Claude with search results")
                final_response = client.messages.create(
                    model=model,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": response.content},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_block.id,
                                    "content": search_results
                                }
                            ]
                        }
                    ],
                    tools=tools,
                    max_tokens=max_tokens
                )

                text_blocks = [block.text for block in final_response.content if hasattr(block, 'text')]
                return ' '.join(text_blocks).strip() if text_blocks else None

        print("INFO: Claude model answered directly without searching.")
        text_blocks = [block.text for block in response.content if hasattr(block, 'text')]
        return ' '.join(text_blocks).strip() if text_blocks else None

    except Exception as e:
        print(f"ERROR: Exception in run_anthropic_search_agent for model '{model}': {e}")
        traceback.print_exc()
        return None
