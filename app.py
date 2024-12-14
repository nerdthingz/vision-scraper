from langchain_community.tools import TavilySearchResults
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from datetime import date
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import ast
import re
import json
import sys
import base64
import tempfile
import io
from typing import Dict, List
import textwrap
import requests
from langchain.prompts import PromptTemplate
from contextlib import redirect_stdout, redirect_stderr
import time
from io import StringIO
import traceback
from selenium import *
from bs4 import BeautifulSoup
from bs4 import *
from PIL import Image
import os
import gradio as gr
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

@tool
def web_scrape(input_str: str) -> str:
    '''Scrapes code based on the objective. Please clarify the objective, considering web scraping will be done. Tell exactly what needs to be scraped.'''

    match = re.search(r"scrape (.+?) - (https?://\S+)", input_str)
    if not match:
        return "Error: Input format is incorrect. Use 'scrape <query> - <url>'."

    query, url = match.groups()

    # Regex to capture additional info
    pattern_2 = r"\(the user gave some image that has been analysed and this is all about it related to the objective(.*?)\)"
    additional_info = re.findall(pattern_2, input_str, re.DOTALL)
    additional_info = " ".join(additional_info).strip() if additional_info else ""

    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        return f"Error fetching the webpage: {e}"

    soup = BeautifulSoup(response.content, "html.parser")

    # Extract a clean structure of the page
    structure = {}
    for tag in ["header", "main", "footer", "article", "section", "aside", "nav"]:
        elements = soup.find_all(tag)
        if elements:
            structure[tag] = [str(element)[:500] for element in elements[:3]]  # Limit to 500 chars for clarity

    body_content = str(soup.body)[:3000] if soup.body else "No body content available."

    # Construct the prompt
    layout_prompt_template = (
        f"You are a web scraping assistant. Extract the following data accurately:\n"
        f"Query: '{query}'\n"
        f"URL: {url}\n\n"
        "Here is the HTML structure of the page:\n"
        f"{structure}\n\n"
        "Here is a snippet of the body content:\n"
        f"{body_content}\n\n"
        "Generate Python code to extract the required information.\n"
        "Make sure the output defines a 'result' or 'scraped_text' variable, and only include relevant information.\n"
        "Avoid including irrelevant sections like ads or footers.\n"
        "Write the code between CODE_START and CODE_END.\n"
    )
    if additional_info:
        layout_prompt_template += f"\nSome additional info that can help is: {additional_info}\n"


    for attempt in range(5):
        try:
            llm_response = model.invoke(layout_prompt_template)
            if not llm_response:
                return "Error: No response from LLM."

            code_match = re.search(r"CODE_START\n(.*?)CODE_END", llm_response.content, re.DOTALL)
            if not code_match:
                return "Error: Could not extract valid code from LLM response."

            python_code = code_match.group(1).strip()

            local_scope = {}
            exec(python_code, {"__builtins__": __builtins__}, local_scope)
            if "result" in local_scope:
                return json.dumps(local_scope["result"], indent=4)
            elif "scraped_text" in local_scope:
                return json.dumps(local_scope["scraped_text"], indent=4)
        except Exception as e:
            traceback.print_exc()
            continue

    return "Sorry, can't scrape. Tried 5 times but couldn't succeed."

def image_scraping_assistant(text, image, url=None):
    structure = {}
    body_content = "No content available."

    if url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            for tag in ["header", "main", "footer", "article", "section", "aside", "nav"]:
                elements = soup.find_all(tag)
                if elements:
                    structure[tag] = [str(element)[:500] for element in elements[:3]]

            body_content = str(soup.body)[:3000] if soup.body else "No body content available."
        except requests.exceptions.RequestException as e:
            return f"Error fetching URL: {e}"

    layout_prompt_template = (
        f"You are an expert image analyst. The provided image is most likely related to a web page. "
        f"Provide small description of the image relative to the query and steps to scrape the information if applicable. "
        f" write your descrition within DESCRIPTION_STARTS and DESCRIPTION_ENDS "
        f"If the query does not involve scraping, simply describe the image concisely. Do not include complete code. "
        f"Single-line code snippets may be provided only if absolutely necessary.\n"
        f"\nQuery: '{text}'\n"
        f"HTML Structure of the Page: {structure}\n"
        f"Snippet of Body Content: {body_content}\n"
        f"Output should have liitle description but should mainly contain steps on how to scrape the specific information and no need to tell how to load the page or what libraie to use to scrape just tell how to scrape the query"
    )

    messages = [{"role": "user", "content": layout_prompt_template}]

    if image:
        try:
            # Handle image provided via Gradio (file path or numpy array)
            if isinstance(image, str):  # File path
                with open(image, "rb") as img_file:
                    image_data = base64.b64encode(img_file.read()).decode("utf-8")
            elif isinstance(image, np.ndarray):  # NumPy array
                pil_image = Image.fromarray(image)
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                    pil_image.save(temp_file.name)
                    temp_path = temp_file.name
                with open(temp_path, "rb") as img_file:
                    image_data = base64.b64encode(img_file.read()).decode("utf-8")
                os.remove(temp_path)
            else:
                raise ValueError("Unsupported image type. Provide a file path or NumPy array.")

            messages.append({"role": "user", "content": [
                {"type": "text", "text": "Image related to the query:"},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_data}"}
            ]})
        except Exception as e:
            return f"Error processing image: {e}"

    try:
        response = model.invoke(messages)
        response_str = io.StringIO(str(response))
        pattern = r"DESCRIPTION_STARTS(.*?)DESCRIPTION_ENDS"
        matches = re.findall(pattern, response_str.getvalue(), re.DOTALL)
        return matches
    except Exception as e:
        error_message = f"Error in image_scraping_assistant: {str(e)}"
        print(error_message)
        return "no additional info"
      
def get_api_keys():
    with gr.Row():
        tavily_key = gr.Textbox(type="password", label="Tavily API Key")
        google_key = gr.Textbox(type="password", label="Google AI API Key")
        groq_key = gr.Textbox(type="password", label="GROQ API Key")
    submit_btn = gr.Button("Submit API Keys")
    status_text = gr.Textbox(label="Status")

    def submit_keys(tavily, google, groq):
        os.environ["TAVILY_API_KEY"] = tavily
        os.environ["GOOGLE_API_KEY"] = google
        os.environ["GROQ_API_KEY"] = groq
        global search_tool, llm, model
        search_tool = TavilySearchResults(
            max_results=1,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=False,
        )
        llm = ChatGroq(model="llama3-70b-8192")
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            max_tokens=None,
            timeout=None,
            max_retries=3,
        )
        return "API keys set, environment variables updated, and tools initialized!"

    submit_btn.click(
        submit_keys,
        inputs=[tavily_key, google_key, groq_key],
        outputs=status_text,
    )

    return status_text   

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def run_agent(text_input, image_input=None):
    if not text_input:
        return [["system", "Please provide text input."]]

    try:
        today = date.today().strftime("%Y-%m-%d")

        # Define tools
        web_scraper_tool = Tool(
            name="WebScraper",
            func=web_scrape,  # Ensure web_scrape is defined elsewhere
            description=""" Scrapes a webpage. Provide the objective and the URL do mention the exact objective that the user asked for make sure you send the input to the function in a correct way that is query first and then url next
                          make sure you provide the query in a correct manner the input format is scrape objective - url
                          if the tool fails once it is recommned you try it 5 times again  before moving on and stay true to the user objective """)

        web_searcher = Tool(
            name="Search",
            func=lambda query: search_tool.run(f"{query} as of {today}"),  # Ensure search_tool is defined
            description="Useful for when you need to answer questions about general information. Input should be a search query. The search will include today's date to ensure the most current information. But before answering, make sure you have the complete information so it's recommended to run it 3 times."
        )

        tools = [web_searcher, web_scraper_tool]

        # Initialize the agent with persistent memory
        agent = initialize_agent(
            tools,
            llm,
            agent="conversational-react-description",
            verbose=False,
            memory=memory,
            handle_parsing_errors=True
        )

        # Extract URL from input
        url_pattern = r"(https?://\S+)"
        match = re.search(url_pattern, text_input)
        found_link = match.group(0) if match else None

        # Handle image input
        internal_text_input = text_input
        if image_input and found_link:
            image_info = image_scraping_assistant(text_input, image_input, found_link)
            additional_prompt = (f"(The user gave an image that has been analyzed, and this is all about it related to the objective: {image_info}) ")
            internal_text_input += additional_prompt

        # Run the agent and update memory
        input_payload = {"input": internal_text_input}
        response = agent.invoke(input_payload)

        # Extract agent response
        agent_response = response.get('output', "No output from agent.")

        # Add only the current messages to memory
        memory.chat_memory.add_user_message(text_input)
        memory.chat_memory.add_ai_message(agent_response)

        # Format the current user and agent messages for Gradio
        formatted_chat = [
            ["user", text_input],
            ["agent", agent_response]
        ]

        return formatted_chat

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        traceback_details = traceback.format_exc()
        print(f"{error_message}\n{traceback_details}")
        return [["error", error_message]]

# Gradio Interface
with gr.Blocks() as demo:
    api_keys_iface = get_api_keys()
    text = gr.Textbox(label="Enter your text here (required if related to scraping the first word should be scrape)")
    image = gr.Image(type="filepath", label="Optional Image")
    run_btn = gr.Button("Run")
    output = gr.Chatbot()

    # Button click to run the agent
    run_btn.click(
        run_agent,
        inputs=[text, image],
        outputs=output,
    )

demo.launch()
