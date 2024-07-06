"""
This module provides an AI assistant powered by Anthropic's Claude-3.5-Sonnet model.
It includes functionalities for creating project structures, writing and editing code,
debugging, and more.
"""

import argparse
import base64
import difflib
import io
import json
import os
import re
import subprocess

import pygments.util
from anthropic import Anthropic
from colorama import Fore, Style, init
from openai import OpenAI

import boto3
from botocore.config import Config
from PIL import Image
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import get_lexer_by_name
from tavily import TavilyClient
from prompts import SYSTEM_PROMPT
from tools import CLAUDE_TOOLS

# Initialize colorama
init()

# Color constants
USER_COLOR = Fore.WHITE
CLAUDE_COLOR = Fore.BLUE
TOOL_COLOR = Fore.YELLOW
RESULT_COLOR = Fore.GREEN

# Add these constants at the top of the file
CONTINUATION_EXIT_PHRASE = "AUTOMODE_COMPLETE"
MAX_CONTINUATION_ITERATIONS = 25

# Initialize the Anthropic client
# read the key from ./anthropic.key
with open("./anthropic.key", "r", encoding="utf-8") as secret_key:
    anthropic_key = secret_key.read().strip()

anthropic_client = Anthropic(api_key=anthropic_key)

# Initialize the OpenAI client
with open("./openai.key", "r", encoding="utf-8") as secret_key:
    openai_key = secret_key.read().strip()

openai_client = OpenAI(api_key=openai_key)

# Initialize the Tavily client
# read the key from ./tv.key
with open("./tavily.key", "r", encoding="utf-8") as secret_key:
    tv_key = secret_key.read().strip()

tavily = TavilyClient(api_key=tv_key)

# Initialize the AWS Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    config=Config(region_name="us-east-1")
)

# Set up the conversation memory
conversation_history = []

# automode flag
AUTOMODE = False

OPENAI_TOOLS = []
BEDROCK_TOOLS = []

def get_system_prompt(current_iteration=None, max_iterations=None):
    automode_status = "You are currently in automode." if AUTOMODE else "You are not in automode."
    iteration_info = ""
    if current_iteration is not None and max_iterations is not None:
        iteration_info = f"You are currently on iteration {current_iteration} out of {max_iterations} in automode."
    return SYSTEM_PROMPT.format(automode_status=automode_status, iteration_info=iteration_info)

def print_colored(text, color):
    print(f"{color}{text}{Style.RESET_ALL}")

def print_code(code, language):
    try:
        lexer = get_lexer_by_name(language, stripall=True)
        formatted_code = highlight(code, lexer, TerminalFormatter())
        print(formatted_code)
    except pygments.util.ClassNotFound:
        print_colored(f"Code (language: {language}):\n{code}", CLAUDE_COLOR)

def create_folder(path):
    try:
        os.makedirs(path, exist_ok=True)
        return f"Folder created: {path}"
    except Exception as e:
        return f"Error creating folder: {str(e)}"

def create_file(path, content=""):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"File created: {path}"
    except Exception as e:
        return f"Error creating file: {str(e)}"

def generate_and_apply_diff(original_content, new_content, path):
    diff = list(difflib.unified_diff(
        original_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=3
    ))

    if not diff:
        return "No changes detected."

    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(new_content)
        return f"Changes applied to {path}:\n" + ''.join(diff)
    except Exception as e:
        return f"Error applying changes: {str(e)}"

def write_to_file(path, content):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            result = generate_and_apply_diff(original_content, content, path)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            result = f"New file created and content written to: {path}"
        return result
    except Exception as e:
        return f"Error writing to file: {str(e)}"

def apply_patch(pwd, patch):
    # write patch content to a tmp file
    tmp_file = "/tmp/patch.diff"
    tmp_file_corrected = "/tmp/patch2.diff"
    try:
        with open(tmp_file, 'w', encoding='utf-8') as f:
            f.write(patch)
        # user recountdiff to fix up line numbers
        with open(tmp_file_corrected, 'w', encoding='utf-8') as f:
            subprocess.run(["recountdiff", tmp_file], stdout=f, check=True, cwd=pwd)
        with open(tmp_file_corrected, 'r', encoding='utf-8') as f:
            result = subprocess.run(["patch", "-p0", "-f", "--fuzz", "10"], stdin=f, cwd=pwd)
            if result.returncode != 0:
                return f"Error applying patch: <stderr>\n```{result.stderr}```\n</stderr>\n<stdout>\n```{result.stdout}```\n</stdout>"
        return "Patch applied successfully."
    except Exception as e:
        return f"Error applying patch: {str(e)}"

def read_file(path):
    result = subprocess.run(["cat", "-n", path], capture_output=True, text=True)
    if result.returncode != 0:
        return f"Error reading file: {result.stderr}"
    return result.stdout

def list_files(path="."):
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files: {str(e)}"

def tavily_search(query):
    try:
        response = tavily.qna_search(query=query, search_depth="advanced")
        return response
    except Exception as e:
        return f"Error performing search: {str(e)}"

def run_build_command(command, pwd):
    result = subprocess.run(command, cwd=pwd, capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        return f"Error running build: <stderr>\n```{result.stderr}```\n</stderr>\n<stdout>\n```{result.stdout}```\n</stdout>"
    return result.stdout

def end_automode():
    global AUTOMODE
    AUTOMODE = False


def execute_tool(tool_name, tool_input):
    if tool_name == "create_folder":
        return create_folder(tool_input["path"])
    elif tool_name == "create_file":
        return create_file(tool_input["path"], tool_input.get("content", ""))
    elif tool_name == "apply_patch":
        return apply_patch(tool_input["pwd"], tool_input["patch"])
    elif tool_name == "read_file":
        return read_file(tool_input["path"])
    elif tool_name == "list_files":
        return list_files(tool_input.get("path", "."))
    elif tool_name == "tavily_search":
        return tavily_search(tool_input["query"])
    elif tool_name == "run_build_command":
        return run_build_command(tool_input["command"], tool_input["pwd"])
    elif tool_name == "end_automode":
        return end_automode()
    else:
        return f"Unknown tool: {tool_name}"

def encode_image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            max_size = (1024, 1024)
            img.thumbnail(max_size, Image.DEFAULT_STRATEGY)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    except Exception as e:
        return f"Error encoding image: {str(e)}"

def parse_goals(response):
    goals = re.findall(r'Goal \d+: (.+)', response)
    return goals

def execute_goals(goals):
    global AUTOMODE
    for i, goal in enumerate(goals, 1):
        print_colored(f"\nExecuting Goal {i}: {goal}", TOOL_COLOR)
        response, _ = chat_with_llm(f"Continue working on goal: {goal}")
        if CONTINUATION_EXIT_PHRASE in response:
            AUTOMODE = False
            print_colored("Exiting automode.", TOOL_COLOR)
            break

def chat_with_llm(user_input, image_path=None, current_iteration=None, max_iterations=None, model="claude-3-5-sonnet-20240620", debug=False):
    global conversation_history

    # Create a new list for the current conversation
    current_conversation = []

    if image_path:
        print_colored(f"Processing image at path: {image_path}", TOOL_COLOR)
        image_base64 = encode_image_to_base64(image_path)

        if image_base64.startswith("Error"):
            print_colored(f"Error encoding image: {image_base64}", TOOL_COLOR)
            return "I'm sorry, there was an error processing the image. Please try again.", False

        image_message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_base64
                    }
                },
                {
                    "type": "text",
                    "text": f"User input for image: {user_input}"
                }
            ]
        }
        current_conversation.append(image_message)
        print_colored("Image message added to conversation history", TOOL_COLOR)
    elif len(conversation_history) == 0 or conversation_history[-1]["role"] != "user":
        current_conversation.append({"role": "user", "content": user_input})

    # Combine the previous conversation history with the current conversation
    messages = conversation_history + current_conversation

    try:
        if model.startswith("claude"):
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=4000,
                system=get_system_prompt(current_iteration, max_iterations),
                messages=messages,
                tools=CLAUDE_TOOLS,
                tool_choice={"type": "auto"}
            )
        elif model.startswith("gpt"):
            openai_messages = [{"role": "system", "content": get_system_prompt(current_iteration, max_iterations)}]
            for msg in messages:
                new_msg = {
                    "role": msg["role"],
                }
                if "tool_calls" in msg:
                    new_msg["tool_calls"] = msg["tool_calls"]
                if "tool_call_id" in msg:
                    new_msg["tool_call_id"] = msg["tool_call_id"]
                if "content" in msg:
                    new_msg["content"] = msg["content"]
                openai_messages.append(new_msg)

            if debug:
                debug_messages = [{"role": m["role"]} for m in openai_messages]
                print(f"OpenAI messages: {json.dumps(debug_messages, indent=2)}", TOOL_COLOR)
            response = openai_client.chat.completions.create(
                model=model,
                messages=openai_messages,
                max_tokens=4000,
                tools=OPENAI_TOOLS,
                tool_choice="auto")
        elif model == "bedrock-claude":
            bedrock_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            response = bedrock_client.converse(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                messages=bedrock_messages,
                system=get_system_prompt(current_iteration, max_iterations),
                tool_config=BEDROCK_TOOLS,
                inference_config=json.dumps({
                    "maxTokens": 4000,
                    "temperature": 0.7,
                    "topP": 0.999,
                })
            )
        else:
            raise ValueError(f"Unsupported model: {model}")
    except Exception as e:
        print_colored(f"Error calling LLM API: {str(e)}", TOOL_COLOR)
        return "I'm sorry, there was an error communicating with the AI. Please try again.", False

    assistant_response = ""
    exit_continuation = False

    if model.startswith("claude"):
        for content_block in response.content:
            if content_block.type == "text":
                assistant_response += content_block.text
                if CONTINUATION_EXIT_PHRASE in content_block.text:
                    exit_continuation = True
            elif content_block.type == "tool_use":
                tool_name = content_block.name
                tool_input = content_block.input
                tool_use_id = content_block.id

                print_colored(f"\nTool Used: {tool_name}", TOOL_COLOR)
                print_colored(f"\nTool Input: {tool_input}", TOOL_COLOR)

                result = execute_tool(tool_name, tool_input)
                print_colored(f"\nTool Result:\n<result>\n```{result}```\n</result>", RESULT_COLOR)

                # Add the tool use to the current conversation
                current_conversation.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": tool_name,
                            "input": tool_input
                        }
                    ]
                })

                # Add the tool result to the current conversation
                current_conversation.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": result
                        }
                    ]
                })

                # Update the messages with the new tool use and result
                messages = conversation_history + current_conversation

                try:
                    tool_response = anthropic_client.messages.create(
                        model=model,
                        max_tokens=4000,
                        system=get_system_prompt(current_iteration, max_iterations),
                        messages=messages,
                        tools=CLAUDE_TOOLS,
                        tool_choice={"type": "auto"}
                    )

                    for tool_content_block in tool_response.content:
                        if tool_content_block.type == "text":
                            assistant_response += tool_content_block.text
                except Exception as e:
                    print_colored(f"Error in tool response: {str(e)}", TOOL_COLOR)
                    assistant_response += "\nI encountered an error while processing the tool result. Please try again."
        current_conversation.append({
            "role": "assistant",
            "content": assistant_response
        })
    elif model.startswith("gpt"):
        msg = response.choices[0].message
        if msg.tool_calls:
            tool_name = msg.tool_calls[0].function.name
            tool_input = eval(msg.tool_calls[0].function.arguments)
            tool_use_id = msg.tool_calls[0].id
            current_conversation.append({
                "role": msg.role,
                "tool_call_id": tool_use_id,
                "tool_calls": msg.tool_calls
            })
            print_colored(f"\nTool Used: {tool_name}", TOOL_COLOR)
            print_colored(f"\nTool Input: {tool_input}", TOOL_COLOR)
            result = execute_tool(tool_name, tool_input)
            print_colored(f"\nTool Result:\n<result>\n```{result}```\n</result>", RESULT_COLOR)
            # add the tool use to the current conversation
            current_conversation.append({
                "role": "tool",
                "tool_call_id": tool_use_id,
                "name": tool_name,
                "content": result
            })
        elif msg.content:
            current_conversation.append({
                "role": msg.role,
                "content": msg.content
            })
            if CONTINUATION_EXIT_PHRASE in msg.content:
                exit_continuation = True

    elif model == "bedrock-claude":
        assistant_response = json.loads(response.body.read()).completion
        current_conversation.append({
            "role": "assistant",
            "content": assistant_response
        })
        if CONTINUATION_EXIT_PHRASE in assistant_response:
            exit_continuation = True

    # Update the global conversation history
    conversation_history = messages + current_conversation

    return assistant_response, exit_continuation

def process_and_display_response(response):
    if response.startswith("Error") or response.startswith("I'm sorry"):
        print_colored(response, TOOL_COLOR)
    else:
        if "```" in response:
            parts = response.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    print_colored(part, CLAUDE_COLOR)
                else:
                    lines = part.split('\n')
                    language = lines[0].strip() if lines else ""
                    code = '\n'.join(lines[1:]) if len(lines) > 1 else ""

                    if language and code:
                        print_code(code, language)
                    elif code:
                        print_colored(f"Code:\n{code}", CLAUDE_COLOR)
                    else:
                        print_colored(part, CLAUDE_COLOR)
        else:
            print_colored(response, CLAUDE_COLOR)

# This function transforms the CLAUDE_TOOLS object into an openai-compatible object
def get_openai_tools(tools):
    openai_tools = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "parameters": tool["input_schema"],
                "description": tool["description"]
            }
        })
    return openai_tools

def get_bedrock_tools(tools):
    bedrock_tools = []
    for tool in tools:
        bedrock_tools.append({
            "toolSpec": {
                "name": tool["name"],
                "inputSchema": {
                    "json": tool["input_schema"]
                },
                "description": tool["description"]
            }
        })
    return {
        "tools": bedrock_tools,
        "tool_choice": "auto"
    }

def main():
    global AUTOMODE, OPENAI_TOOLS, BEDROCK_TOOLS, conversation_history

    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--automode", type=int, help="Enter automode with a specific number of iterations")
    parser.add_argument("--prompt", type=str, help="The prompt to start the conversation with")
    parser.add_argument("--pwd", type=str, help="The path to the directory where the project is located")
    parser.add_argument("--file", type=str, help="A path to a file to be added to the initial input")
    parser.add_argument("--build-command", type=str, help="The command to build the project")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20240620", help="The LLM model to use (claude-3-5-sonnet-20240620, gpt-4o, gpt-4-turbo, gpt-3.5-turbo, bedrock-claude)")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    args = parser.parse_args()

    if args.debug:
        print_colored("Debug mode enabled.", TOOL_COLOR)

    # Generate an openai-compatible TOOLS object by transforming CLAUDE_TOOLS
    OPENAI_TOOLS = get_openai_tools(CLAUDE_TOOLS)
    BEDROCK_TOOLS = get_bedrock_tools(CLAUDE_TOOLS)

    if args.automode:
        AUTOMODE = True
        max_iterations = args.automode
        print_colored(f"Entering automode with {max_iterations} iterations. Press Ctrl+C to exit automode at any time.", TOOL_COLOR)

        if args.prompt:
            user_input = args.prompt
        else:
            user_input = input(f"\n{USER_COLOR}You: {Style.RESET_ALL}")

        if args.pwd:
            user_input += f"\nThe project is located at '{args.pwd}'"
        if args.file:
            full_path = os.path.join(args.pwd, args.file)
            with open(full_path, 'r') as f:
                user_input += f"\nA file that is important to this task is '{args.file}' and the content is:\n<content>```\n{f.read()}\n```</content>"
        if args.build_command:
            try:
                result = execute_tool("run_build_command", {"command": args.build_command, "pwd": args.pwd})
                user_input += f"\nThe tool to use to build the project is 'run_build_command' and the current result is:\n<result>```\n{result}\n```</result>"
                user_input += "\nRun the build tool after changes to the project are complete, then fix any errors."
            except Exception as e:
                print_colored(f"Error executing tool: {str(e)}", TOOL_COLOR)
                user_input += "\nI encountered an error while executing the tool. Please try again."

        iteration_count = 0
        try:
            while AUTOMODE and iteration_count < max_iterations:
                response, exit_continuation = chat_with_llm(user_input, current_iteration=iteration_count+1, max_iterations=max_iterations, model=args.model, debug=args.debug)
                process_and_display_response(response)

                if exit_continuation or CONTINUATION_EXIT_PHRASE in response:
                    print_colored("Automode completed.", TOOL_COLOR)
                    AUTOMODE = False
                else:
                    print_colored(f"Continuation iteration {iteration_count + 1} completed.", TOOL_COLOR)
                    print_colored("Press Ctrl+C to exit automode.", TOOL_COLOR)
                    user_input = "Continue with the next step or exit automode if the task is completed and the build passes."

                iteration_count += 1   

                if iteration_count >= max_iterations:
                    print_colored("Max iterations reached. Exiting automode.", TOOL_COLOR)
                    AUTOMODE = False
        except KeyboardInterrupt:
            print_colored("\nAutomode interrupted by user. Exiting automode.", TOOL_COLOR)
            AUTOMODE = False
            if conversation_history and conversation_history[-1]["role"] == "user":
                conversation_history.append({"role": "assistant", "content": "Automode interrupted. How can I assist you further?"})
    else:
        print_colored("Welcome to the Claude-3.5-Sonnet Engineer Chat with Image Support!", CLAUDE_COLOR)
        print_colored("Type 'exit' to end the conversation.", CLAUDE_COLOR)
        print_colored("Type 'image' to include an image in your message.", CLAUDE_COLOR)
        print_colored("Type 'automode [number]' to enter Autonomous mode with a specific number of iterations.", CLAUDE_COLOR)
        print_colored("While in automode, press Ctrl+C at any time to exit the automode to return to regular chat.", CLAUDE_COLOR)

    while True:
        if args.prompt:
            user_input = args.prompt
            args.prompt = None
        else:
            user_input = input(f"\n{USER_COLOR}You: {Style.RESET_ALL}")

        if user_input.lower() == 'exit':
            print_colored("Thank you for chatting. Goodbye!", CLAUDE_COLOR)
            break

        if user_input.lower() == 'image':
            image_path = input(f"{USER_COLOR}Drag and drop your image here: {Style.RESET_ALL}").strip().replace("'", "")

            if os.path.isfile(image_path):
                user_input = input(f"{USER_COLOR}You (prompt for image): {Style.RESET_ALL}")
                response, _ = chat_with_llm(user_input, image_path, debug=args.debug)
                process_and_display_response(response)
            else:
                print_colored("Invalid image path. Please try again.", CLAUDE_COLOR)
                continue
        elif user_input.lower().startswith('automode'):
            try:
                parts = user_input.split()
                if len(parts) > 1 and parts[1].isdigit():
                    max_iterations = int(parts[1])
                else:
                    max_iterations = MAX_CONTINUATION_ITERATIONS

                AUTOMODE = True
                print_colored(f"Entering automode with {max_iterations} iterations. Press Ctrl+C to exit automode at any time.", TOOL_COLOR)
                print_colored("Press Ctrl+C at any time to exit the automode loop.", TOOL_COLOR)
                user_input = input(f"\n{USER_COLOR}You: {Style.RESET_ALL}")

                iteration_count = 0
                try:
                    while AUTOMODE and iteration_count < max_iterations:
                        response, exit_continuation = chat_with_llm(user_input, current_iteration=iteration_count+1, max_iterations=max_iterations, model=args.model, debug=args.debug)
                        process_and_display_response(response)

                        if exit_continuation or CONTINUATION_EXIT_PHRASE in response:
                            print_colored("Automode completed.", TOOL_COLOR)
                            AUTOMODE = False
                        else:
                            print_colored(f"Continuation iteration {iteration_count + 1} completed.", TOOL_COLOR)
                            print_colored("Press Ctrl+C to exit automode.", TOOL_COLOR)
                            user_input = "Continue with the next step."

                        iteration_count += 1

                        if iteration_count >= max_iterations:
                            print_colored("Max iterations reached. Exiting automode.", TOOL_COLOR)
                            AUTOMODE = False
                except KeyboardInterrupt:
                    print_colored("\nAutomode interrupted by user. Exiting automode.", TOOL_COLOR)
                    AUTOMODE = False
                    # Ensure the conversation history ends with an assistant message
                    if conversation_history and conversation_history[-1]["role"] == "user":
                        conversation_history.append({"role": "assistant", "content": "Automode interrupted. How can I assist you further?"})
            except KeyboardInterrupt:
                print_colored("\nAutomode interrupted by user. Exiting automode.", TOOL_COLOR)
                AUTOMODE = False
                # Ensure the conversation history ends with an assistant message
                if conversation_history and conversation_history[-1]["role"] == "user":
                    conversation_history.append({"role": "assistant", "content": "Automode interrupted. How can I assist you further?"})

            print_colored("Exited automode. Returning to regular chat.", TOOL_COLOR)
        else:
            response, _ = chat_with_llm(user_input, model=args.model, debug=args.debug)
            process_and_display_response(response)

if __name__ == "__main__":
    main()
