#!/usr/bin/env python3

import os
import sys
import asyncio
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.logging import RichHandler
import logging
import pyperclip
import click
from anthropic import Anthropic
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

# -------------------------------
# Configuration Constants
# -------------------------------

EXCLUDE_PATTERNS = [
    "*~", "*.tmp", "*.log", "*.o", "*.out", "*.class", "*.DS_Store",
    "*.swp", "*.bak", "*.pyc", "*.wasm", "*.d.ts", "package-lock.json"
]
EXCLUDE_DIRS = [
    "node_modules", "debug", "deps", "target", "pkg", "refs", ".git"
]
MAX_TOKENS = 4000  # Adjust based on Claude's token limit
TOKEN_ESTIMATE_PER_CHAR = 0.25  # 1 token ≈ 4 characters
BATCH_MAX_TOKENS = 3000  # Tokens reserved for prompt and response
OUTPUT_FILE = "relevant_code.txt"
LOG_FILE = "copy_context.log"

# -------------------------------
# Initialize Environment
# -------------------------------

# Load environment variables from .env file
load_dotenv()

# Initialize Rich console
console = Console()

# Setup Rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("copy_context")

# -------------------------------
# Utility Functions
# -------------------------------

def estimate_tokens(char_count):
    """
    Estimate the number of tokens based on character count.
    """
    return int(char_count * TOKEN_ESTIMATE_PER_CHAR)

def find_files(directory):
    """
    Find all files in the specified directory excluding patterns and directories.
    """
    logger.debug(f"Searching for files in '{directory}' with exclusions.")
    cmd = ['find', str(directory), '-type', 'f']

    # Add excluded file patterns
    for pattern in EXCLUDE_PATTERNS:
        cmd += ['!', '-name', pattern]

    # Add excluded directories
    for d in EXCLUDE_DIRS:
        cmd += ['!', '-path', f"*/{d}/*"]

    cmd += ['-print0']

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        files = result.stdout.split(b'\0')
        files = [f.decode('utf-8') for f in files if f]
        logger.info(f"Found {len(files)} files to process.")
        return files
    except subprocess.CalledProcessError as e:
        logger.error(f"Error finding files: {e.stderr.decode('utf-8')}")
        sys.exit(1)

def retry(exception_to_check, tries=4, delay=3, backoff=2):
    """
    Retry decorator for handling transient errors.
    """
    def deco_retry(f):
        @wraps(f)
        async def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return await f(*args, **kwargs)
                except exception_to_check as e:
                    logger.warning(f"{e}, Retrying in {mdelay} seconds...")
                    await asyncio.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            # Last attempt
            return await f(*args, **kwargs)
        return f_retry
    return deco_retry

# -------------------------------
# Anthropic API Interaction
# -------------------------------

async def process_with_claude(anthropic_client: Anthropic, prompt: str):
    """
    Send the prompt to Anthropic's Claude API and return the response.
    """
    try:
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",  # Ensure this matches your intended model
            system='SYSTEM_PROMPT',
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            stop_sequences=["\n\nHuman:"],
            stream=True
        )

        full_response = ""
        for chunk in response:
            if hasattr(chunk, 'type') and chunk.type == 'content_block_delta':
                if hasattr(chunk.delta, 'text'):
                    full_response += chunk.delta.text
        return full_response
    except Exception as e:
        logger.exception(f"Exception during API call: {e}")
        return ""

@retry(Exception, tries=4, delay=3, backoff=2)
async def process_with_claude_async(anthropic_client: Anthropic, prompt: str, executor):
    """
    Asynchronous wrapper to process the prompt with retries.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: asyncio.run(process_with_claude(anthropic_client, prompt)))

# -------------------------------
# Main Processing Function
# -------------------------------

async def main(directory, user_prompt, concurrency, verbose):
    """
    Main asynchronous function to process files and interact with Claude's API.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    files = find_files(directory)
    total_files = len(files)
    if total_files == 0:
        console.print("[yellow]No files found to process.[/yellow]")
        sys.exit(0)

    output_path = Path(OUTPUT_FILE)
    output_path.write_text("")  # Clear previous content

    token_count = 0
    batch_content = ""
    processed_files = 0

    # Initialize Anthropic client
    api_key = os.getenv('CLAUDE_API_KEY')
    if not api_key:
        console.print("[red]Error: CLAUDE_API_KEY not found in environment variables.[/red]")
        sys.exit(1)
    anthropic_client = Anthropic(api_key=api_key)

    # Initialize ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=concurrency)

    # Define System Prompt with detailed instructions
    SYSTEM_PROMPT = (
        "You are an assistant that analyzes code files to answer specific questions from developers.\n"
        "When analyzing the code, you should:\n"
        "1. Understand the context and goal of the developer's question.\n"
        "2. Break down the code into meaningful sections.\n"
        "3. Provide explanations of key parts related to the developer's question.\n"
        "4. Offer suggestions for improvements, where relevant, and explain your reasoning.\n"
        "5. Stitch together an answer that addresses the developer's question in relation to the code provided.\n"
    )

    # User Message Template
    USER_MESSAGE_TEMPLATE = (
        "Question: {question}\n\n"
        "Please analyze the following code snippets and provide detailed insights and suggestions based on the question above."
    )

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("[cyan]Processing Files...", total=total_files)
        for file in files:
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.read()
            except Exception as e:
                logger.error(f"Error reading file {file}: {str(e)}")
                progress.update(task, advance=1)
                continue

            char_count = len(file_content)
            file_token_count = estimate_tokens(char_count)

            if token_count + file_token_count > BATCH_MAX_TOKENS:
                # Prepare the prompt for the current batch
                user_message = USER_MESSAGE_TEMPLATE.format(question=user_prompt) + f"\n\nCode Snippets:\n{batch_content.strip()}"
                full_prompt = SYSTEM_PROMPT + "\n\n" + user_message

                # Process current batch asynchronously
                relevant_content = await process_with_claude_async(anthropic_client, full_prompt, executor)
                if relevant_content:
                    output_path.write_text(output_path.read_text() + relevant_content + "\n\n")
                batch_content = ""
                token_count = 0

            batch_content += f"{file_content}\n\n"
            token_count += file_token_count
            processed_files += 1
            progress.update(task, advance=1)

        # Process any remaining content
        if batch_content:
            user_message = USER_MESSAGE_TEMPLATE.format(question=user_prompt) + f"\n\nCode Snippets:\n{batch_content.strip()}"
            full_prompt = SYSTEM_PROMPT + "\n\n" + user_message
            relevant_content = await process_with_claude_async(anthropic_client, full_prompt, executor)
            if relevant_content:
                output_path.write_text(output_path.read_text() + relevant_content + "\n\n")

    # Shutdown the executor
    executor.shutdown(wait=True)

    # Copy to clipboard
    try:
        content_to_copy = output_path.read_text()
        pyperclip.copy(content_to_copy)
        console.print(f"\n[green]🎉 Successfully processed {processed_files} file(s).[/green]")
        console.print(f"[green]Relevant code snippets have been saved to '{OUTPUT_FILE}' and copied to the clipboard![/green]")
    except Exception as e:
        logger.exception(f"Error copying to clipboard: {str(e)}")
        console.print("[red]Processed content saved, but failed to copy to clipboard. Check the log for details.[/red]")

# -------------------------------
# CLI Entry Point
# -------------------------------

@click.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), default='.')
@click.option('--concurrency', default=10, show_default=True, help='Number of concurrent API requests.')
@click.option('--verbose', is_flag=True, help='Enable verbose output for debugging.')
def cli_main(directory, concurrency, verbose):
    """
    Efficiently process and filter code files based on a user prompt using Claude's API.

    DIRECTORY is the directory to process. Defaults to the current directory.
    """
    # Prompt the user for input
    user_prompt = console.input("[bold cyan]Enter your prompt for Claude:[/bold cyan] ").strip()
    if not user_prompt:
        console.print("[red]Prompt cannot be empty. Exiting.[/red]")
        sys.exit(1)

    try:
        asyncio.run(main(directory, user_prompt, concurrency=concurrency, verbose=verbose))
    except Exception as e:
        logger.exception(f"Unhandled exception: {str(e)}")
        console.print("[red]An unexpected error occurred. Check the log for details.[/red]")
        sys.exit(1)

if __name__ == "__main__":
    cli_main()
