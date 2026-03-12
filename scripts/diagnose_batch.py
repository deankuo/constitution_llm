#!/usr/bin/env python3
"""
Diagnose batch vs sync differences for Gemini.

Sends the same prompt via sync and batch paths, then compares:
1. Whether system_instruction is applied in batch
2. Whether thinking tokens are generated in batch
3. Whether response content quality differs
"""

import os
import time
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_sync(api_key: str, model: str, system_prompt: str, user_prompt: str):
    """Run via sync GeminiLLM (the old google.generativeai SDK)."""
    from models.llm_clients import GeminiLLM

    llm = GeminiLLM(model=model, api_key=api_key)
    response = llm.call(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0,
    )
    return {
        "content": response.content,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "thinking_tokens": response.thinking_tokens,
    }


def run_batch(api_key: str, model: str, system_prompt: str, user_prompt: str):
    """Run via batch API (the new google.genai SDK)."""
    from google import genai

    client = genai.Client(api_key=api_key)

    # Format A: system_instruction inside config (current code)
    request_a = {
        "contents": [
            {"parts": [{"text": user_prompt}], "role": "user"}
        ],
        "config": {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "temperature": 0,
        },
    }

    # Format B: system_instruction at top level (per issue #1190)
    request_b = {
        "contents": [
            {"parts": [{"text": user_prompt}], "role": "user"}
        ],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "config": {
            "temperature": 0,
        },
    }

    results = {}
    for label, req in [("batch_config", request_a), ("batch_toplevel", request_b)]:
        print(f"\n  Submitting {label}...")
        job = client.batches.create(
            model=model,
            src=[req],
            config={"display_name": f"diag-{label}-{int(time.time())}"},
        )
        print(f"  Job: {job.name} — polling...")

        while True:
            job = client.batches.get(name=job.name)
            state = str(job.state)
            if "SUCCEEDED" in state or "COMPLETED" in state:
                break
            if "FAILED" in state or "CANCELLED" in state:
                print(f"  Job {label} ended with state: {state}")
                results[label] = {"error": state}
                break
            time.sleep(5)
        else:
            continue  # only reached if while didn't break to error

        # Extract response
        try:
            inlined = list(job.dest.inlined_responses)
            resp = inlined[0].response
            text = ""
            if hasattr(resp, "text"):
                try:
                    text = resp.text
                except (ValueError, AttributeError):
                    pass
            if not text and hasattr(resp, "candidates"):
                try:
                    text = resp.candidates[0].content.parts[0].text
                except (IndexError, AttributeError):
                    pass
            if not text:
                text = str(resp)

            usage = getattr(resp, "usage_metadata", None)
            results[label] = {
                "content": text,
                "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
                "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
                "thinking_tokens": getattr(usage, "thoughts_token_count", 0) or 0,
            }
        except Exception as e:
            results[label] = {"error": str(e)}

    return results


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable")
        return

    model = "gemini-2.5-pro"

    # Test prompt: system instruction tells the model to always prefix with a keyword
    system_prompt = (
        "You are a classifier. You MUST start your response with the exact "
        'string "SYS_OK:" before any other output. This is mandatory.'
    )
    user_prompt = (
        'Classify the sovereignty of the Roman Republic under Julius Caesar '
        '(-49 to -44). Respond with a JSON object: '
        '{"sovereign": "0" or "1", "reasoning": "brief", "confidence_score": 1-100}'
    )

    print("=" * 70)
    print("BATCH vs SYNC DIAGNOSTIC")
    print("=" * 70)

    # Sync
    print("\n[1/3] Running SYNC call...")
    sync_result = run_sync(api_key, model, system_prompt, user_prompt)
    print(f"  Content starts with: {sync_result['content'][:80]}...")
    print(f"  Thinking tokens: {sync_result['thinking_tokens']}")
    print(f"  Has SYS_OK prefix: {'SYS_OK:' in sync_result['content'][:20]}")

    # Batch
    print("\n[2/3] Running BATCH calls (config format + top-level format)...")
    batch_results = run_batch(api_key, model, system_prompt, user_prompt)

    # Compare
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    for label, result in [("sync", sync_result)] + list(batch_results.items()):
        print(f"\n--- {label} ---")
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue
        content = result["content"]
        print(f"  Has SYS_OK prefix: {'SYS_OK:' in content[:20]}")
        print(f"  Thinking tokens:   {result['thinking_tokens']}")
        print(f"  Input tokens:      {result['input_tokens']}")
        print(f"  Output tokens:     {result['output_tokens']}")
        print(f"  Content preview:   {content[:120]}...")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
If SYS_OK prefix is present  → system_instruction IS applied
If SYS_OK prefix is missing  → system_instruction IS IGNORED (root cause!)
If thinking_tokens = 0       → thinking is disabled in batch (quality impact)
Compare batch_config vs batch_toplevel to see which format works.
""")


if __name__ == "__main__":
    main()
