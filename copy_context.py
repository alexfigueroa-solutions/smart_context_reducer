#!/usr/bin/env python3

import os
import sys
import asyncio
import aiohttp
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from colorama import Fore, Style, init
import logging
import pyperclip

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()

# Configuration Constants
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

# Setup Logging
logging.basicConfig(
    filename=LOG_FILE,
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Function to estimate tokens
def estimate_tokens(char_count):
    return char_count * TOKEN_ESTIMATE_PER_CHAR

# Function to find files with exclusions
def find_files(directory):
    logging.info(f"Searching for files in '{directory}' with exclusions.")
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
        logging.info(f"Found {len(files)} files to process.")
        return files
    except subprocess.CalledProcessError as e:
        logging.error(f"Error finding files: {e.stderr.decode('utf-8')}")
        print(Fore.RED + "Error finding files. Check the log for details.")
        sys.exit(1)

# Function to interact with Claude's API
async def process_with_claude(session, prompt, content):
    api_endpoint = os.getenv('CLAUDE_API_ENDPOINT')
    api_key = os.getenv('CLAUDE_API_KEY')

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    payload = {
        'prompt': prompt,
        'content': content,
        'max_tokens': 1500  # Adjust as needed
    }

    try:
        async with session.post(api_endpoint, json=payload, headers=headers) as response:
            if response.status != 200:
                text = await response.text()
                logging.error(f"API Error {response.status}: {text}")
                return ""
            data = await response.json()
            processed_content = data.get('processed_content', '')
            return processed_content
    except Exception as e:
        logging.error(f"Exception during API call: {str(e)}")
        return ""

# Asynchronous main processing function
async def main(directory, user_prompt):
    files = find_files(directory)
    total_files = len(files)
    if total_files == 0:
        print(Fore.YELLOW + "No files found to process.")
        sys.exit(0)

    output_path = Path(OUTPUT_FILE)
    output_path.write_text("")  # Clear previous content

    token_count = 0
    batch_content = ""
    copied_files = 0

    connector = aiohttp.TCPConnector(limit=10)  # Limit concurrent connections
    async with aiohttp.ClientSession(connector=connector) as session:
        for file in tqdm(files, desc="Processing Files", unit="file"):
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.read()
            except Exception as e:
                logging.error(f"Error reading file {file}: {str(e)}")
                continue

            char_count = len(file_content)
            file_token_count = estimate_tokens(char_count)

            if token_count + file_token_count > BATCH_MAX_TOKENS:
                # Process current batch
                relevant_content = await process_with_claude(session, user_prompt, batch_content)
                if relevant_content:
                    output_path.write_text(output_path.read_text() + relevant_content + "\n\n")
                    copied_files += 1
                batch_content = ""
                token_count = 0

            batch_content += file_content + "\n\n"
            token_count += file_token_count
            copied_files += 1

        # Process any remaining content
        if batch_content:
            relevant_content = await process_with_claude(session, user_prompt, batch_content)
            if relevant_content:
                output_path.write_text(output_path.read_text() + relevant_content + "\n\n")

    # Copy to clipboard
    try:
        content_to_copy = output_path.read_text()
        pyperclip.copy(content_to_copy)
        print(Fore.GREEN + f"\n🎉 Successfully processed {copied_files} file(s).")
        print(Fore.GREEN + f"Relevant code snippets have been saved to '{OUTPUT_FILE}' and copied to the clipboard!")
    except Exception as e:
        logging.error(f"Error copying to clipboard: {str(e)}")
        print(Fore.RED + "Processed content saved, but failed to copy to clipboard. Check the log for details.")

# Synchronous CLI Entry Point
def cli_main():
    parser = argparse.ArgumentParser(
        description="Efficiently process and filter code files based on a user prompt using Claude's API."
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default=os.getcwd(),
        help='Directory to copy files from (default: current directory)'
    )
    args = parser.parse_args()

    # Prompt the user
    print(Fore.CYAN + "==============================================")
    print(Fore.CYAN + "         🔥 OP File Processing Script 🔥")
    print(Fore.CYAN + "==============================================\n")
    user_prompt = input("Enter your prompt for Claude: ").strip()
    if not user_prompt:
        print(Fore.RED + "Prompt cannot be empty. Exiting.")
        sys.exit(1)

    # Run the main asynchronous function
    try:
        asyncio.run(main(args.directory, user_prompt))
    except KeyboardInterrupt:
        print(Fore.RED + "\nProcess interrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unhandled exception: {str(e)}")
        print(Fore.RED + "An unexpected error occurred. Check the log for details.")
        sys.exit(1)

if __name__ == "__main__":
    cli_main()
